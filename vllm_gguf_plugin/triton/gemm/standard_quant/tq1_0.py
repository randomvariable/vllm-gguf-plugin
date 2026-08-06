"""Dedicated dense GEMM for GGML type 34 (TQ1_0).

TQ1_0 is the base-3 ternary format: 256 trits packed five-per-byte (3^5 = 243
< 256) into ``qs[48]``, plus a four-per-byte tail in ``qh[4]``, followed by one
fp16 scale -- 54 bytes total.

Decode, from llama.cpp's ``dequantize_row_tq1_0``::

    q = (byte * pow3[plane]) & 0xFF  # uint8 truncation isolates the trit
    xi = (q * 3) >> 8
    y = (xi - 1) * d

The ``& 0xFF`` is load-bearing: the quantiser rescales each packed byte with a
ceiling division ``(q * 256 + 242) / 243`` so that truncated multiplication by
``pow3[plane]`` recovers exactly one trit per plane.

Output ordering walks three segments::

    qs[0:32]  x 5 planes -> outputs   0..159
    qs[32:48] x 5 planes -> outputs 160..239
    qh[0:4]   x 4 planes -> outputs 240..255
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 54
BLOCK_SIZE = 256
QS_BYTES = 48
QH_BYTES = 4
PAYLOAD_BYTES = QS_BYTES + QH_BYTES
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

GEMM_BLOCK_M = 32
GEMM_BLOCK_N = 32
NUM_WARPS = 4
NUM_STAGES = 2


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
def _load_fp16_scale(base_ptr, mask):
    """Load the trailing little-endian fp16 scale from a block."""
    lo = tl.load(base_ptr + PAYLOAD_BYTES, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(base_ptr + PAYLOAD_BYTES + 1, mask=mask, other=0).to(tl.uint16)
    return tl.cast(lo | (hi << 8), tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _tq1_0_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    m,
    n,
    num_blocks,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE)

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

    for block in range(num_blocks):
        base = w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES
        row_mask = offs_n[:, None] < n

        scale = _load_fp16_scale(base, row_mask)

        packed = tl.load(
            base + byte_index[None, :],
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        # uint8 truncation then (q * 3) >> 8 recovers the trit.
        q = (packed * plane_mul[None, :]) & 0xFF
        xi = (q * 3) >> 8
        w_tile = (xi.to(tl.float32) - 1.0) * scale

        x_tile = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + (block * BLOCK_SIZE + offs_k)[None, :] * stride_xk,
            mask=(offs_m[:, None] < m),
            other=0.0,
        )
        acc = tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)), acc=acc)

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def ggml_gemm_tq1_0_triton(
    W: torch.Tensor,
    X: torch.Tensor,
    row: int,
) -> torch.Tensor:
    """Dense GEMM for TQ1_0 packed weights."""
    if W.dim() != 2:
        raise ValueError(f"TQ1_0 weights must be 2D, got {W.dim()}D")
    if W.dtype is not torch.uint8:
        raise TypeError(f"TQ1_0 weights must be torch.uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Triton TQ1_0 kernel supports torch.float16, torch.bfloat16, and "
            f"torch.float32 activations, got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if any(size <= 0 for size in X.shape):
        raise ValueError(f"X dimensions must be positive, got {tuple(X.shape)}")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    if W.shape[1] % BLOCK_BYTES != 0:
        raise ValueError(
            f"Invalid TQ1_0 row width {W.shape[1]}: must be divisible by {BLOCK_BYTES}"
        )
    if not W.is_cuda or not X.is_cuda:
        raise ValueError("TQ1_0 Triton kernel requires CUDA tensors")

    num_blocks = W.shape[1] // BLOCK_BYTES
    hidden_size = num_blocks * BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match "
            f"TQ1_0 weight width {hidden_size}"
        )

    W = W.contiguous()
    x_shape = X.shape
    x_2d = X.reshape(-1, hidden_size).contiguous()
    y_2d = torch.empty((x_2d.shape[0], row), device=X.device, dtype=X.dtype)

    grid = (
        triton.cdiv(x_2d.shape[0], GEMM_BLOCK_M),
        triton.cdiv(row, GEMM_BLOCK_N),
    )
    _tq1_0_gemm_kernel[grid](
        x_2d,
        W,
        y_2d,
        x_2d.shape[0],
        row,
        num_blocks,
        x_2d.stride(0),
        x_2d.stride(1),
        W.stride(0),
        y_2d.stride(0),
        y_2d.stride(1),
        BLOCK_M=GEMM_BLOCK_M,
        BLOCK_N=GEMM_BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    if X.dim() == 2:
        return y_2d
    return y_2d.view(*x_shape[:-1], row)
