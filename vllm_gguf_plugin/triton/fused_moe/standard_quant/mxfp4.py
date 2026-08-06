# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel for GGML type 39 (MXFP4).

MXFP4 packs 32 weights into 17 bytes: 16 packed 4-bit code bytes followed by
one trailing E8M0 scale byte. Low nibbles decode to logical weights 0..15 and
high nibbles to 16..31, so both halves are emitted contiguously and share the
one scale.

This is structurally ROCmFP4_FAST (type 101) with two constants changed: the
codebook's eighth entry is 12 rather than 10, and the scale is E8M0 -- a pure
power of two -- rather than UE4M3.

The codebook and scale decode are shared with the dense GEMM lane so the two
cannot drift apart.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.standard_quant.mxfp4 import _codebook, _e8m0_scale
from ...gemm.utils import GGML_TYPE_MXFP4
from ..utils import load_moe_token_info, run_triton_fused_moe_kernel

BLOCK_BYTES = 17
BLOCK_SIZE = 32
QS_BYTES = 16


@triton.jit
def _mxfp4_moe_kernel(
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
    offs_nibble = tl.arange(0, QS_BYTES)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks

        # Load X as two contiguous 16-value halves to match the weight layout.
        # load_moe_x_tile interleaves the low/high K split for paired layouts
        # and would pair weights against the wrong activations.
        x_base = (
            offs_token[:, None, None] * stride_xm
            + (cur_kb[None, :, None] * BLOCK_SIZE) * stride_xk
        )
        x_low_ptrs = x_ptr + x_base + offs_nibble[None, None, :] * stride_xk
        x_high_ptrs = (
            x_ptr + x_base + (offs_nibble[None, None, :] + QS_BYTES) * stride_xk
        )
        tile_mask = token_mask[:, None, None] & kb_mask[None, :, None]
        x_low = tl.load(x_low_ptrs, mask=tile_mask, other=0.0)
        x_high = tl.load(x_high_ptrs, mask=tile_mask, other=0.0)
        x_tile = tl.reshape(
            tl.join(x_low, x_high), (BLOCK_M, BLOCK_K_BLOCKS * BLOCK_SIZE)
        )
        x_dtype = x_tile.dtype

        code_ptrs = (
            w_row_ptrs[:, :, None]
            + cur_kb[None, :, None] * BLOCK_BYTES
            + offs_nibble[None, None, :]
        )
        code_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]
        packed = tl.load(code_ptrs, mask=code_mask, other=0).to(tl.int32)
        low_codes = _codebook(packed & 0xF)
        high_codes = _codebook(packed >> 4)

        # One trailing E8M0 scale byte per block, applied to all 32 weights.
        scale_ptrs = w_row_ptrs + cur_kb[None, :] * BLOCK_BYTES + QS_BYTES
        scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
        scale = _e8m0_scale(tl.load(scale_ptrs, mask=scale_mask, other=0)).to(x_dtype)

        w_low = low_codes * scale[:, :, None]
        w_high = high_codes * scale[:, :, None]
        # tl.dot requires matching operand dtypes; the codebook returns float32
        # regardless of the scale's dtype, so cast after the product.
        w_tile = tl.reshape(
            tl.join(w_low, w_high), (BLOCK_N, BLOCK_K_BLOCKS * BLOCK_SIZE)
        ).to(x_dtype)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_mxfp4_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the MXFP4 fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _mxfp4_moe_kernel,
        w,
        x,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_MXFP4,
    )
