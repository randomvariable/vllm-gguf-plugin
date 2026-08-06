# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel for GGML type 40 (NVFP4).

NVFP4 packs 64 weights into 36 bytes: four leading UE4M3 scale bytes followed
by 32 payload bytes. Each 16-weight sub-block owns one scale and eight payload
bytes, and within a sub-block low nibbles fill the first eight outputs and high
nibbles the second eight.

Because each sub-block carries its own scale, the reduction is done one
sub-block at a time rather than as one 64-wide tile.

The codebook and scale decode are shared with the dense GEMM lane so the two
cannot drift apart.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.standard_quant.nvfp4 import _codebook, _ue4m3_scale
from ...gemm.utils import GGML_TYPE_NVFP4
from ..utils import load_moe_token_info, run_triton_fused_moe_kernel

BLOCK_BYTES = 36
BLOCK_SIZE = 64
SUB_SIZE = 16
SUB_COUNT = BLOCK_SIZE // SUB_SIZE
SCALE_BYTES = SUB_COUNT
SUB_PAYLOAD_BYTES = SUB_SIZE // 2


@triton.jit
def _nvfp4_moe_kernel(
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
    offs_j = tl.arange(0, SUB_PAYLOAD_BYTES)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks

        # Each sub-block has its own scale, so it is decoded and reduced
        # separately rather than as one 64-wide tile.
        for sub in range(0, SUB_COUNT):
            k_base = cur_kb * BLOCK_SIZE + sub * SUB_SIZE

            x_base = offs_token[:, None, None] * stride_xm
            x_low_ptrs = (
                x_ptr
                + x_base
                + (k_base[None, :, None] + offs_j[None, None, :]) * stride_xk
            )
            x_high_ptrs = (
                x_ptr
                + x_base
                + (k_base[None, :, None] + SUB_PAYLOAD_BYTES + offs_j[None, None, :])
                * stride_xk
            )
            tile_mask = token_mask[:, None, None] & kb_mask[None, :, None]
            x_low = tl.load(x_low_ptrs, mask=tile_mask, other=0.0)
            x_high = tl.load(x_high_ptrs, mask=tile_mask, other=0.0)
            x_tile = tl.reshape(
                tl.join(x_low, x_high), (BLOCK_M, BLOCK_K_BLOCKS * SUB_SIZE)
            )
            x_dtype = x_tile.dtype

            code_ptrs = (
                w_row_ptrs[:, :, None]
                + cur_kb[None, :, None] * BLOCK_BYTES
                + SCALE_BYTES
                + sub * SUB_PAYLOAD_BYTES
                + offs_j[None, None, :]
            )
            code_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]
            packed = tl.load(code_ptrs, mask=code_mask, other=0).to(tl.int32)
            low_codes = _codebook(packed & 0xF)
            high_codes = _codebook(packed >> 4)

            scale_ptrs = w_row_ptrs + cur_kb[None, :] * BLOCK_BYTES + sub
            scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
            scale = _ue4m3_scale(
                tl.load(scale_ptrs, mask=scale_mask, other=0).to(tl.int32)
            ).to(x_dtype)

            w_low = low_codes * scale[:, :, None]
            w_high = high_codes * scale[:, :, None]
            # tl.dot requires matching operand dtypes; the codebook returns
            # float32 regardless of the scale's dtype, so cast after the product.
            w_tile = tl.reshape(
                tl.join(w_low, w_high), (BLOCK_N, BLOCK_K_BLOCKS * SUB_SIZE)
            ).to(x_dtype)

            acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_nvfp4_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the NVFP4 fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _nvfp4_moe_kernel,
        w,
        x,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_NVFP4,
    )
