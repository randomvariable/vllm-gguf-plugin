"""Fused MoE kernel for GGML type 103 (Q8_0_ROCMFPX).

Type 103 packs 32 signed int8 codes into 32 bytes plus one UE4M3 scale byte
(byte 32), giving a 33-byte block. The scale decode is shared  the dense
GEMM kernel via ``decode_ue4m3_scale`` so the two lanes cannot drift apart.
Reserved scale bytes ``0x7F..0xFF`` decode to zero.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q8_0_ROCMFPX
from ...utils_rocmfpx_decode import decode_ue4m3_scale
from ..utils import (
    load_moe_token_info,
    load_moe_x_tile,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 33
BLOCK_SIZE = 32


@triton.jit
def _q8_0_rocmfpx_moe_kernel(
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Expert weight base: [expert, output_row, packed_cols].
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        x_tile, cur_kb, kb_mask = load_moe_x_tile(
            x_ptr,
            num_k_blocks,
            stride_xm,
            stride_xk,
            offs_token,
            token_mask,
            kb_start,
            offs_kb,
            offs_code,
            BLOCK_M=BLOCK_M,
            BLOCK_K_BLOCKS=BLOCK_K_BLOCKS,
        )
        x_dtype = x_tile.dtype

        # Load 32 int8 codes per packed block. load_moe_x_tile builds x_tile as
        # (BLOCK_M, BLOCK_K_BLOCKS * 32), so the weight tile must match K layout.
        code_ptrs = (
            w_row_ptrs[:, :, None]
            + cur_kb[None, :, None] * BLOCK_BYTES
            + offs_code[None, None, :]
        )
        code_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]
        packed = tl.load(code_ptrs, mask=code_mask, other=0)
        # Reinterpret stored byte as a signed int8 code.
        codes = tl.where(packed < 128, packed, packed - 256).to(tl.float32)

        # One UE4M3 scale byte per block, shared across all 32 codes.
        scale_ptrs = w_row_ptrs[:, None] + cur_kb[None, :] * BLOCK_BYTES + 32
        scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
        scale_byte = tl.load(scale_ptrs, mask=scale_mask, other=0)
        scale = decode_ue4m3_scale(scale_byte).to(x_dtype)  # (BLOCK_N, BLOCK_K_BLOCKS)

        # Weight tile: codes (BLOCK_N, BLOCK_K_BLOCKS, 32) * scale.
        w_tile = codes * scale[:, :, None]
        w_tile = tl.reshape(w_tile, (BLOCK_N, BLOCK_K_BLOCKS * BLOCK_SIZE))

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_q8_0_rocmfpx_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return run_triton_fused_moe_kernel(
        _q8_0_rocmfpx_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q8_0_ROCMFPX,
    )
