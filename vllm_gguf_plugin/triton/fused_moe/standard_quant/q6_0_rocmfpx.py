"""Fused MoE kernel for GGML type 102 (Q6_0_ROCMFPX).

Type 102 packs 32 6-bit sign-magnitude codes into 24 payload bytes plus two
UE4M3 half-scale bytes (24 and 25), giving a 26-byte block. Code ``j`` occupies
bits ``[6j, 6j+6)``, so codes may straddle a byte boundary and the decode reads
two adjacent bytes. The first scale covers outputs 0..15 and the second covers
16..31. Reserved scale bytes ``0x7F..0xFF`` decode to zero.

Codes are sign-magnitude, not two's complement: bit 5 is the sign and bits 0..4
are the magnitude. A set sign bit with zero magnitude encodes -32 rather than
negative zero.

The layout is sequential (code ``j`` -> output ``j``), so activations are
loaded sequentially rather than via the nibble-paired helper.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q6_0_ROCMFPX
from ...utils_rocmfpx_decode import decode_ue4m3_scale
from ..utils import (
    load_moe_token_info,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 26
BLOCK_SIZE = 32
CODE_BITS = 6
PAYLOAD_BYTES = 24


@triton.jit
def _q6_0_rocmfpx_moe_kernel(
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

    bit_offset = offs_code * CODE_BITS
    byte_index = bit_offset // 8
    bit_shift = bit_offset % 8
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

        low_byte = tl.load(
            block_base + byte_index[None, None, :], mask=tile_mask, other=0
        ).to(tl.int32)
        # The high byte only matters when a code straddles the boundary;
        # otherwise the shifted bits fall outside the 6-bit mask.
        high_byte = tl.load(
            block_base + tl.minimum(byte_index + 1, PAYLOAD_BYTES - 1)[None, None, :],
            mask=tile_mask,
            other=0,
        ).to(tl.int32)
        codes = (
            (low_byte >> bit_shift[None, None, :])
            | (high_byte << (8 - bit_shift)[None, None, :])
        ) & 0x3F

        magnitude = codes & 31
        # Sign bit set with zero magnitude encodes -32, not negative zero.
        signed = tl.where(
            (codes & 32) != 0,
            -tl.where(magnitude != 0, magnitude, 32),
            magnitude,
        ).to(tl.float32)

        scale_byte = tl.load(
            block_base + PAYLOAD_BYTES + scale_index[None, None, :],
            mask=tile_mask,
            other=0,
        )
        w_tile = signed * decode_ue4m3_scale(scale_byte)
        w_tile = tl.reshape(w_tile, (BLOCK_N, BLOCK_K_BLOCKS * BLOCK_SIZE)).to(x_dtype)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_q6_0_rocmfpx_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the type-102 ROCmFPX fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _q6_0_rocmfpx_moe_kernel,
        w,
        x,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q6_0_ROCMFPX,
    )
