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
PAYLOAD_BYTES = BLOCK_SIZE // 2

# The shared default BLOCK_N=128 targets 32-element blocks; this format's
# 64-element block doubles the weight tile, so halve the N tile to stay inside
# the 64 KiB LDS budget on gfx1151. Q2_0 does the same for the same reason.
MOE_BLOCK_N = 64


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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    # One lane per payload byte. Byte p belongs to sub-block p // 8 and carries
    # the codes for outputs sub*16 + (p % 8) and sub*16 + 8 + (p % 8).
    #
    # The whole block is reduced in one 64-wide dot rather than four 16-wide
    # ones, with each byte's scale gathered alongside it. Four narrow dots is
    # the more obvious shape and is correct on gfx1151, but mispairs operands
    # on sm_121, so the wide form is the portable one.
    offs_p = tl.arange(0, PAYLOAD_BYTES)
    sub_of_p = offs_p // SUB_PAYLOAD_BYTES
    x_low_idx = sub_of_p * SUB_SIZE + (offs_p % SUB_PAYLOAD_BYTES)
    x_high_idx = x_low_idx + SUB_PAYLOAD_BYTES

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks

        x_base = offs_token[:, None, None] * stride_xm
        k_base = cur_kb * BLOCK_SIZE
        x_low_ptrs = (
            x_ptr
            + x_base
            + (k_base[None, :, None] + x_low_idx[None, None, :]) * stride_xk
        )
        x_high_ptrs = (
            x_ptr
            + x_base
            + (k_base[None, :, None] + x_high_idx[None, None, :]) * stride_xk
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
            + SCALE_BYTES
            + offs_p[None, None, :]
        )
        code_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]
        packed = tl.load(code_ptrs, mask=code_mask, other=0).to(tl.int32)

        # Gather per byte, so one tile carries all four sub-block scales.
        scale_ptrs = (
            w_row_ptrs[:, :, None]
            + cur_kb[None, :, None] * BLOCK_BYTES
            + sub_of_p[None, None, :]
        )
        scale = _ue4m3_scale(tl.load(scale_ptrs, mask=code_mask, other=0).to(tl.int32))

        w_low = _codebook(packed & 0xF) * scale
        w_high = _codebook(packed >> 4) * scale
        # tl.dot requires matching operand dtypes; the codebook returns float32
        # regardless of the scale's dtype, so cast after the product.
        w_tile = tl.reshape(
            tl.join(w_low, w_high), (BLOCK_N, BLOCK_K_BLOCKS * BLOCK_SIZE)
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
        block_n=MOE_BLOCK_N,
    )
