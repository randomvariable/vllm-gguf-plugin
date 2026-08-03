"""Dedicated dense GEMM for GGML type 103 (Q8_0_ROCMFPX).

Type 103 packs 32 signed int8 codes into 32 bytes plus one UE4M3 scale
byte (byte 32), giving a 33-byte block. Unlike types 100/101 the codes
are stored contiguously with no nibble packing and a single scale covers
all 32 weights. Reserved scale bytes ``0x7F..0xFF`` decode to zero.
"""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 33
BLOCK_SIZE = 32
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Launch configuration measured on gfx1151 (Radeon 8060S) at rows=11008,
# hidden=4096, M=1. The 33-byte block is the widest ROCmFPX layout, so the
# per-row strided reads need extra warps to keep enough loads in flight:
# num_warps=1 measured 1847us (25.2 GB/s) versus 598us (77.7 GB/s) here.
LAUNCH_BLOCK_M = 16
LAUNCH_BLOCK_N = 16
LAUNCH_NUM_WARPS = 4
LAUNCH_NUM_STAGES = 2


@triton.jit
def _gemm_kernel(
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

    for block in range(num_blocks):
        packed = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES + offs_k[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.int32)
        # Reinterpret the stored byte as a signed int8 code.
        codes = tl.where(packed < 128, packed, packed - 256).to(tl.float32)

        scale_byte = tl.load(
            w_ptr + offs_n * stride_wn + block * BLOCK_BYTES + 32,
            mask=offs_n < n,
            other=0,
        ).to(tl.int32)
        exponent = (scale_byte >> 3) & 0xF
        mantissa = scale_byte & 7
        scale = tl.where(
            scale_byte > 0x7E,
            0.0,
            tl.where(
                exponent == 0,
                mantissa.to(tl.float32) / 1024.0,
                (1.0 + mantissa.to(tl.float32) / 8.0)
                * tl.exp2(exponent.to(tl.float32) - 8.0),
            ),
        )
        w_tile = codes * scale[:, None]

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


def _decode_cpu(weights: torch.Tensor) -> torch.Tensor:
    blocks = weights.reshape(weights.shape[0], -1, BLOCK_BYTES).to(torch.int32)
    packed = blocks[..., :32]
    codes = torch.where(packed < 128, packed, packed - 256).to(torch.float32)
    scale_bytes = blocks[..., 32:33]
    exponent = (scale_bytes >> 3) & 0xF
    mantissa = scale_bytes & 7
    scales = torch.where(
        scale_bytes > 0x7E,
        torch.zeros_like(scale_bytes, dtype=torch.float32),
        torch.where(
            exponent == 0,
            mantissa.float() / 1024.0,
            (1 + mantissa.float() / 8) * torch.pow(2.0, exponent.float() - 8),
        ),
    )
    return (codes * scales).reshape(weights.shape[0], -1)


def ggml_gemm_q8_0_rocmfpx_triton(
    W: torch.Tensor, X: torch.Tensor, row: int
) -> torch.Tensor:
    if W.dim() != 2:
        raise ValueError(f"Q8_0_ROCMFPX weights must be 2D, got {W.dim()}D")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Q8_0_ROCMFPX weights must be uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Q8_0_ROCMFPX supports float16, bfloat16, and float32 activations, "
            f"got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if any(size <= 0 for size in X.shape):
        raise ValueError("Q8_0_ROCMFPX activations must have positive dimensions")
    if W.shape[0] <= 0 or W.shape[1] <= 0:
        raise ValueError("Q8_0_ROCMFPX weights need positive rows and width")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    if W.shape[1] % BLOCK_BYTES:
        raise ValueError(f"Q8_0_ROCMFPX row width must be divisible by {BLOCK_BYTES}")
    hidden_size = W.shape[1] // BLOCK_BYTES * BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match weight hidden size "
            f"{hidden_size}"
        )
    if W.device != X.device:
        raise ValueError(
            "Q8_0_ROCMFPX weights and activations must be on the same device"
        )

    W = W.contiguous()
    x_shape = X.shape
    X = X.reshape(-1, hidden_size).contiguous()
    if not X.is_cuda:
        return (X.float() @ _decode_cpu(W).T).to(X.dtype).view(*x_shape[:-1], row)

    Y = torch.empty((X.shape[0], row), device=X.device, dtype=X.dtype)
    _gemm_kernel[
        (triton.cdiv(X.shape[0], LAUNCH_BLOCK_M), triton.cdiv(row, LAUNCH_BLOCK_N))
    ](
        X,
        W,
        Y,
        X.shape[0],
        row,
        hidden_size // BLOCK_SIZE,
        X.stride(0),
        X.stride(1),
        W.stride(0),
        Y.stride(0),
        Y.stride(1),
        BLOCK_M=LAUNCH_BLOCK_M,
        BLOCK_N=LAUNCH_BLOCK_N,
        num_warps=LAUNCH_NUM_WARPS,
        num_stages=LAUNCH_NUM_STAGES,
    )
    return Y.view(*x_shape[:-1], row)
