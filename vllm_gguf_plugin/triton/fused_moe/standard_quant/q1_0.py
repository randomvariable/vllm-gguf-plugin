"""Fused MoE kernel for GGML type 41 (Q1_0).

Q1_0 packs 128 one-bit codes into 16 bytes preceded by one fp16 scale, giving
an 18-byte block. Bit ``j`` selects ``+d`` when set and ``-d`` when clear,
LSB-first within each byte. The layout is sequential (code ``j`` -> output
``j``), so activations load sequentially.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q1_0
from ..utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 18
BLOCK_SIZE = 128
SCALE_BYTES = 2

# The shared default BLOCK_N=128 targets 32-element blocks. This format's
# 128-element block makes the weight tile 4x larger, overflowing LDS
# (67584 > 65536 on gfx1151), so the N tile shrinks to compensate.
MOE_BLOCK_N = 32


@triton.jit
def _q1_0_moe_kernel(
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
    byte_index = offs_k // 8
    bit_shift = offs_k % 8

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
        bits = (packed >> bit_shift[None, :]) & 1
        w_tile = tl.where(bits == 1, scale[:, None], -scale[:, None])

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


def ggml_moe_q1_0_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the Q1_0 fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _q1_0_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q1_0,
        block_n=MOE_BLOCK_N,
    )
