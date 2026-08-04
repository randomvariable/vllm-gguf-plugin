"""Fused MoE kernel for GGML type 107 (Q2_0_ROCMFPX).

Type 107 packs 32 2-bit codes into 8 payload bytes plus two UE4M3 half-scale
bytes (8 and 9), giving a 10-byte block. Code ``j`` occupies bits
``[2j, 2j+2)``; four codes fit exactly in each byte, so no code straddles a
byte boundary. The first scale covers outputs 0..15 and the second covers
16..31. Reserved scale bytes ``0x7F..0xFF`` decode to zero.

The layout is sequential (code ``j`` -> output ``j``), so activations are
loaded sequentially, matching the type-103 kernel rather than the nibble-paired
type-100/101 kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q2_0_ROCMFPX
from ...utils_rocmfpx_decode import decode_ue4m3_scale
from ..utils import (
    load_moe_token_info,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 10
BLOCK_SIZE = 32
CODE_BITS = 2
PAYLOAD_BYTES = 8


@triton.jit
def _codebook(code):
    """Type-107 codebook: [-4, -1, 1, 4]."""
    return tl.where(
        code == 0,
        -4.0,
        tl.where(code == 1, -1.0, tl.where(code == 2, 1.0, 4.0)),
    )


@triton.jit
def _q2_0_rocmfpx_moe_kernel(
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
    offs_output, offs_token, token_mask = load_moe_token_info(
        sorted_token_ids_ptr,
        tl.program_id(0),
        top_k,
        num_valid_tokens,
        BLOCK_M=BLOCK_M,
    )

    expert = tl.load(expert_ids_ptr + tl.program_id(0))
    if expert < 0:
        return

    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_code = tl.arange(0, BLOCK_SIZE)

    # Four 2-bit codes per byte, no straddling.
    byte_index = offs_code // 4
    bit_shift = (offs_code % 4) * CODE_BITS
    scale_index = offs_code // 16

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks

        x_ptrs = (
            x_ptr
            + offs_token[:, None, None] * stride_xm
            + (cur_kb[None, :, None] * BLOCK_SIZE + offs_code[None, None, :])
            * stride_xk
        )
        x_tile = tl.load(
            x_ptrs,
            mask=token_mask[:, None, None] & kb_mask[None, :, None],
            other=0.0,
        )
        x_tile = tl.reshape(x_tile, (BLOCK_M, BLOCK_K_BLOCKS * BLOCK_SIZE))
        x_dtype = x_tile.dtype

        block_base = w_row_ptrs[:, :, None] + cur_kb[None, :, None] * BLOCK_BYTES
        tile_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]

        packed = tl.load(
            block_base + byte_index[None, None, :], mask=tile_mask, other=0
        ).to(tl.int32)
        codes = (packed >> bit_shift[None, None, :]) & 0x3

        scale_byte = tl.load(
            block_base + PAYLOAD_BYTES + scale_index[None, None, :],
            mask=tile_mask,
            other=0,
        )
        w_tile = _codebook(codes) * decode_ue4m3_scale(scale_byte)
        w_tile = tl.reshape(w_tile, (BLOCK_N, BLOCK_K_BLOCKS * BLOCK_SIZE)).to(x_dtype)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_q2_0_rocmfpx_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the type-107 ROCmFPX fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _q2_0_rocmfpx_moe_kernel,
        w,
        x,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q2_0_ROCMFPX,
    )
