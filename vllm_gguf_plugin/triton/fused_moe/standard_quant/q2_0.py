"""Fused MoE kernel for GGML type 42 (Q2_0).

Q2_0 packs 64 two-bit codes into 16 bytes preceded by one fp16 scale, giving
an 18-byte block. Code ``j`` occupies bits ``[2*(j%4), 2*(j%4)+2)`` of byte
``j//4``; four codes fit exactly per byte. Codes are offset binary, decoded as
``(code - 1) * d``. The layout is sequential, so activations load
sequentially.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q2_0
from ..utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 18
BLOCK_SIZE = 64
CODE_BITS = 2
SCALE_BYTES = 2

# The shared default BLOCK_N=128 targets 32-element blocks; this format's
# 64-element block doubles the weight tile, so halve the N tile to stay
# inside the 64 KiB LDS budget on gfx1151.
MOE_BLOCK_N = 64


@triton.jit
def _q2_0_moe_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    top_k,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_output, offs_token, token_mask = load_moe_token_info(
        sorted_token_ids_ptr, pid_m, top_k, num_valid_tokens, BLOCK_M=BLOCK_M
    )
    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < n
    offs_k = tl.arange(0, BLOCK_SIZE)
    byte_index = offs_k // 4
    bit_shift = (offs_k % 4) * CODE_BITS

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        base = w_row_ptrs + kb * BLOCK_BYTES

        lo = tl.load(base, mask=n_mask, other=0).to(tl.uint16)
        hi = tl.load(base + 1, mask=n_mask, other=0).to(tl.uint16)
        scale = tl.cast(lo | (hi << 8), tl.float16, bitcast=True).to(tl.float32)

        packed = tl.load(
            base[:, None] + SCALE_BYTES + byte_index[None, :],
            mask=n_mask[:, None],
            other=0,
        ).to(tl.int32)
        codes = (packed >> bit_shift[None, :]) & 0x3
        w_tile = (codes.to(tl.float32) - 1.0) * scale[:, None]

        x_tile = load_moe_x_chunk(
            x_ptr,
            stride_xm,
            stride_xk,
            offs_token,
            token_mask,
            kb * BLOCK_SIZE,
            CHUNK=BLOCK_SIZE,
        )
        acc = tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


def ggml_moe_q2_0_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the Q2_0 fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _q2_0_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q2_0,
        block_n=MOE_BLOCK_N,
    )
