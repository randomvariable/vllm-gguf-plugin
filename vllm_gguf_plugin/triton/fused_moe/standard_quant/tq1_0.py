"""Fused MoE kernel for GGML type 34 (TQ1_0).

TQ1_0 is the base-3 ternary format: 256 trits packed five-per-byte (3^5 = 243
< 256) into ``qs[48]``, plus a four-per-byte tail in ``qh[4]``, followed by one
fp16 scale -- 54 bytes total.

Decode, from ``dequantize_row_tq1_0``::

    q = (byte * pow3[plane]) & 0xFF  # uint8 truncation isolates the trit
    xi = (q * 3) >> 8
    y = (xi - 1) * d

The ``& 0xFF`` is load-bearing: the quantiser rescales each byte by a ceiling
division so truncated multiplication recovers exactly one trit per plane.

Output ordering walks three segments::

    qs[0:32]  x 5 planes -> outputs   0..159
    qs[32:48] x 5 planes -> outputs 160..239
    qh[0:4]   x 4 planes -> outputs 240..255
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_TQ1_0
from ..utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    run_triton_fused_moe_kernel,
)

BLOCK_BYTES = 54
BLOCK_SIZE = 256
QS_BYTES = 48
QH_BYTES = 4
PAYLOAD_BYTES = QS_BYTES + QH_BYTES

# QK_K=256 is 8x the 32-element block the shared defaults target. Decoding a
# whole block at once would need BLOCK_N * 256 * 4 bytes of LDS (131 KiB at the
# default BLOCK_N=128, over the 64 KiB limit on gfx1151) *and* hand tl.dot a
# K=256 tile, which has no matching wmma intrinsic at BLOCK_M=4 -- the kernel
# silently drops to FMA. Decoding one K-chunk at a time fixes both.
K_CHUNK = 64
CHUNKS_PER_BLOCK = 256 // K_CHUNK


@triton.jit
def _pow3(plane):
    """Small base-3 table: [1, 3, 9, 27, 81] for planes 0..4."""
    return tl.where(
        plane == 0,
        1,
        tl.where(
            plane == 1,
            3,
            tl.where(plane == 2, 9, tl.where(plane == 3, 27, 81)),
        ),
    )


@triton.jit
def _tq1_0_moe_kernel(
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
    offs_c = tl.arange(0, K_CHUNK)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        base = w_row_ptrs + kb * BLOCK_BYTES

        lo = tl.load(base + PAYLOAD_BYTES, mask=n_mask, other=0).to(tl.uint16)
        hi = tl.load(base + PAYLOAD_BYTES + 1, mask=n_mask, other=0).to(tl.uint16)
        scale = tl.cast(lo | (hi << 8), tl.float16, bitcast=True).to(tl.float32)

        for chunk in range(CHUNKS_PER_BLOCK):
            offs_k = chunk * K_CHUNK + offs_c

            # Segment-dependent byte index and plane, see module docstring.
            in_seg0 = offs_k < 160
            in_seg1 = (offs_k >= 160) & (offs_k < 240)
            k1 = offs_k - 160
            k2 = offs_k - 240
            byte_index = tl.where(
                in_seg0,
                offs_k % 32,
                tl.where(in_seg1, 32 + (k1 % 16), QS_BYTES + (k2 % 4)),
            )
            plane = tl.where(
                in_seg0,
                offs_k // 32,
                tl.where(in_seg1, k1 // 16, k2 // 4),
            )
            plane_mul = _pow3(plane)

            packed = tl.load(
                base[:, None] + byte_index[None, :],
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            # uint8 truncation then (q * 3) >> 8 recovers the trit.
            q = (packed * plane_mul[None, :]) & 0xFF
            xi = (q * 3) >> 8
            w_tile = (xi.to(tl.float32) - 1.0) * scale[:, None]

            x_tile = load_moe_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_token,
                token_mask,
                kb * BLOCK_SIZE + chunk * K_CHUNK,
                CHUNK=K_CHUNK,
            )
            acc = tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


def ggml_moe_tq1_0_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Run the TQ1_0 fused MoE kernel."""
    return run_triton_fused_moe_kernel(
        _tq1_0_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_TQ1_0,
    )
