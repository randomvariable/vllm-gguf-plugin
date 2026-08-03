"""Dedicated dense GEMM for GGML type 107 (Q2_0_ROCMFPX).

Type 107 packs 32 2-bit codes into 8 bytes followed by two UE4M3
half-scales (bytes 8 and 9), giving a 10-byte block. Code ``j`` occupies
bits ``[2j, 2j+2)`` of the payload; four codes fit exactly in each byte so
no code straddles a byte boundary. The first scale covers outputs 0..15,
the second covers 16..31. Reserved scale bytes ``0x7F..0xFF`` decode to
zero.
"""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 10
BLOCK_SIZE = 32
CODE_BITS = 2
PAYLOAD_BYTES = 8
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _codebook(code):
    # [-4, -1, 1, 4]
    return tl.where(
        code == 0,
        -4.0,
        tl.where(code == 1, -1.0, tl.where(code == 2, 1.0, 4.0)),
    )


@triton.jit
def _decode_scale(scale_byte):
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    return tl.where(
        scale_byte > 0x7E,
        0.0,
        tl.where(
            exponent == 0,
            mantissa.to(tl.float32) / 1024.0,
            (1.0 + mantissa.to(tl.float32) / 8.0)
            * tl.exp2(exponent.to(tl.float32) - 8.0),
        ),
    )


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
    byte_index = offs_k // 4
    bit_shift = (offs_k % 4) * CODE_BITS

    for block in range(num_blocks):
        base = w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES
        packed = tl.load(
            base + byte_index[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.int32)
        codes = (packed >> bit_shift[None, :]) & 0x3

        scale_byte = tl.load(
            base + PAYLOAD_BYTES + (offs_k // 16)[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.int32)
        w_tile = _codebook(codes) * _decode_scale(scale_byte)

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
    payload = blocks[..., :PAYLOAD_BYTES]

    offs = torch.arange(BLOCK_SIZE, device=weights.device)
    byte_index = offs // 4
    bit_shift = (offs % 4) * CODE_BITS
    codes = (payload[..., byte_index] >> bit_shift) & 0x3

    codebook = torch.tensor([-4, -1, 1, 4], dtype=torch.float32, device=weights.device)
    values = codebook[codes.long()]

    scale_bytes = blocks[..., PAYLOAD_BYTES : PAYLOAD_BYTES + 2]
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
    ).repeat_interleave(16, dim=-1)
    return (values * scales).reshape(weights.shape[0], -1)


def ggml_gemm_q2_0_rocmfpx_triton(
    W: torch.Tensor, X: torch.Tensor, row: int
) -> torch.Tensor:
    if W.dim() != 2:
        raise ValueError(f"Q2_0_ROCMFPX weights must be 2D, got {W.dim()}D")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Q2_0_ROCMFPX weights must be uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Q2_0_ROCMFPX supports float16, bfloat16, and float32 activations, "
            f"got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if any(size <= 0 for size in X.shape):
        raise ValueError("Q2_0_ROCMFPX activations must have positive dimensions")
    if W.shape[0] <= 0 or W.shape[1] <= 0:
        raise ValueError("Q2_0_ROCMFPX weights must have positive dimensions")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    if W.shape[1] % BLOCK_BYTES:
        raise ValueError(f"Q2_0_ROCMFPX row width must be divisible by {BLOCK_BYTES}")
    hidden_size = W.shape[1] // BLOCK_BYTES * BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match weight hidden size "
            f"{hidden_size}"
        )
    if W.device != X.device:
        raise ValueError(
            "Q2_0_ROCMFPX weights and activations must be on the same device"
        )

    W = W.contiguous()
    x_shape = X.shape
    X = X.reshape(-1, hidden_size).contiguous()
    if not X.is_cuda:
        return (X.float() @ _decode_cpu(W).T).to(X.dtype).view(*x_shape[:-1], row)

    Y = torch.empty((X.shape[0], row), device=X.device, dtype=X.dtype)
    _gemm_kernel[(triton.cdiv(X.shape[0], 16), triton.cdiv(row, 16))](
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
        BLOCK_M=16,
        BLOCK_N=16,
        num_warps=1,
        num_stages=2,
    )
    return Y.view(*x_shape[:-1], row)
