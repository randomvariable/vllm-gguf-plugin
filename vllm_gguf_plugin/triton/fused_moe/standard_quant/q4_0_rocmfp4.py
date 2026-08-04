"""Fused MoE kernel for GGML type 100 (Q4_0_ROCMFP4).

Type 100 packs 32 weights into 18 bytes: 16 packed 4-bit code bytes followed by
two UE4M3 half-scale bytes. The low nibble of each packed byte decodes via
Codebook10 to the first 16 weights; the high nibble decodes to the next 16.
The two halves are emitted contiguously (low-then-high), not interleaved, and
each half is scaled by its own UE4M3 byte. Reserved scale bytes ``0x7F..0xFF``
decode to zero. This matches the authoritative ABI (memory #5323) and the
validated dense GEMM kernel in ``gemm/standard_quant/q4_0_rocmfp4.py``.

The scale decode is shared with the dense lane via ``decode_ue4m3_scale`` so
the two cannot drift apart.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q4_0_ROCMFP4
from ...utils_rocmfpx_decode import decode_ue4m3_scale
from ..utils import (
    load_moe_token_info,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 18
BLOCK_SIZE = 32

# Codebook10 values (memory #5013): [0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10].
# The kernel inlines the lookup via _q4_0_rocmfp4_codebook (tl.where chain)
# because Triton has no module-level constexpr tensor literals.


@triton.jit
def _q4_0_rocmfp4_codebook(index):
    """Codebook10 lookup: [0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10]."""
    # Bits 0..2 select the magnitude (0..7), bit 3 selects the sign.
    mag = index & 7
    sign = (index >> 3) & 1
    magnitude = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            1.0,
            tl.where(
                mag == 2,
                2.0,
                tl.where(
                    mag == 3,
                    3.0,
                    tl.where(
                        mag == 4,
                        4.0,
                        tl.where(
                            mag == 5,
                            6.0,
                            tl.where(mag == 6, 8.0, 10.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where(sign != 0, -magnitude, magnitude)


@triton.jit
def _q4_0_rocmfp4_moe_kernel(
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
    offs_nibble = tl.arange(0, 16)  # 16 packed bytes per block -> low/high halves

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Expert weight base: [expert, output_row, packed_cols].
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks

        # Load X as two contiguous 16-value halves (low then high), matching
        # the weight layout. load_moe_x_tile interleaves a low/high split for
        # Q4_0/Q8_0-style paired layouts, which would pair weights against the
        # wrong activations here (same trap that bit type 103).
        x_base = (
            offs_token[:, None, None] * stride_xm
            + (cur_kb[None, :, None] * BLOCK_SIZE) * stride_xk
        )
        x_low_ptrs = x_ptr + x_base + offs_nibble[None, None, :] * stride_xk
        x_high_ptrs = x_ptr + x_base + (offs_nibble[None, None, :] + 16) * stride_xk
        tile_mask = token_mask[:, None, None] & kb_mask[None, :, None]
        x_low = tl.load(x_low_ptrs, mask=tile_mask, other=0.0)
        x_high = tl.load(x_high_ptrs, mask=tile_mask, other=0.0)
        x_tile = tl.reshape(
            tl.join(x_low, x_high), (BLOCK_M, BLOCK_K_BLOCKS * BLOCK_SIZE)
        )
        x_dtype = x_tile.dtype

        # Load 16 packed code bytes per block.
        code_ptrs = (
            w_row_ptrs[:, :, None]
            + cur_kb[None, :, None] * BLOCK_BYTES
            + offs_nibble[None, None, :]
        )
        code_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]
        packed = tl.load(code_ptrs, mask=code_mask, other=0).to(tl.int32)
        low_codes = _q4_0_rocmfp4_codebook(packed & 0xF)
        high_codes = _q4_0_rocmfp4_codebook(packed >> 4)

        # Two UE4M3 half-scale bytes per block: byte 16 scales the low half,
        # byte 17 scales the high half. Reserved bytes 0x7F..0xFF decode to 0.
        low_scale_ptrs = w_row_ptrs + cur_kb[None, :] * BLOCK_BYTES + 16
        high_scale_ptrs = w_row_ptrs + cur_kb[None, :] * BLOCK_BYTES + 17
        scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
        low_scale = decode_ue4m3_scale(
            tl.load(low_scale_ptrs, mask=scale_mask, other=0)
        ).to(x_dtype)
        high_scale = decode_ue4m3_scale(
            tl.load(high_scale_ptrs, mask=scale_mask, other=0)
        ).to(x_dtype)

        # Weight tile: contiguous low half (16) then high half (16) per block.
        w_low = low_codes * low_scale[:, :, None]
        w_high = high_codes * high_scale[:, :, None]
        w_tile = tl.reshape(
            tl.join(w_low, w_high), (BLOCK_N, BLOCK_K_BLOCKS * BLOCK_SIZE)
        )

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_q4_0_rocmfp4_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the type-100 ROCmFPX fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _q4_0_rocmfp4_moe_kernel,
        w,
        x,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q4_0_ROCMFP4,
    )
