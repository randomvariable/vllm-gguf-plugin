"""Dedicated dense GEMM for GGML type 41 (Q1_0).

Q1_0 is a 1-bit format: 128 codes packed into 16 bytes preceded by one fp16
scale, giving an 18-byte block. Bit ``j`` of the payload selects ``+d`` when
set and ``-d`` when clear, LSB-first within each byte. The scale leads the
block, unlike the K-quants where it trails.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 18
BLOCK_SIZE = 128
SCALE_BYTES = 2
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

GEMM_BLOCK_M = 32
GEMM_BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _load_fp16_scale(base_ptr, mask):
    """Load the leading little-endian fp16 scale from a block."""
    lo = tl.load(base_ptr + 0, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(base_ptr + 1, mask=mask, other=0).to(tl.uint16)
    return tl.cast(lo | (hi << 8), tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _q1_0_gemm_kernel(
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
    byte_index = offs_k // 8
    bit_shift = offs_k % 8

    for block in range(num_blocks):
        base = w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES
        row_mask = offs_n[:, None] < n

        scale = _load_fp16_scale(base, row_mask)

        packed = tl.load(
            base + SCALE_BYTES + byte_index[None, :],
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        bits = (packed >> bit_shift[None, :]) & 1
        w_tile = tl.where(bits == 1, scale, -scale)

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


def ggml_gemm_q1_0_triton(
    W: torch.Tensor,
    X: torch.Tensor,
    row: int,
) -> torch.Tensor:
    """Dense GEMM for Q1_0 packed weights."""
    if W.dim() != 2:
        raise ValueError(f"Q1_0 weights must be 2D, got {W.dim()}D")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Q1_0 weights must be torch.uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Triton Q1_0 kernel supports torch.float16, torch.bfloat16, and "
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
            f"Invalid Q1_0 row width {W.shape[1]}: must be divisible by {BLOCK_BYTES}"
        )
    if not W.is_cuda or not X.is_cuda:
        raise ValueError("Q1_0 Triton kernel requires CUDA tensors")

    num_blocks = W.shape[1] // BLOCK_BYTES
    hidden_size = num_blocks * BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match "
            f"Q1_0 weight width {hidden_size}"
        )

    W = W.contiguous()
    x_shape = X.shape
    x_2d = X.reshape(-1, hidden_size).contiguous()
    y_2d = torch.empty((x_2d.shape[0], row), device=X.device, dtype=X.dtype)

    grid = (
        triton.cdiv(x_2d.shape[0], GEMM_BLOCK_M),
        triton.cdiv(row, GEMM_BLOCK_N),
    )
    _q1_0_gemm_kernel[grid](
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
