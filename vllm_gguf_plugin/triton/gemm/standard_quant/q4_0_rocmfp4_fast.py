"""Dedicated GEMM for GGML type 101 (ROCmFP4_FAST)."""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 17
BLOCK_SIZE = 32
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _codebook(code):
    return tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            1.0,
            tl.where(
                code == 2,
                2.0,
                tl.where(
                    code == 3,
                    3.0,
                    tl.where(
                        code == 4,
                        4.0,
                        tl.where(
                            code == 5,
                            6.0,
                            tl.where(
                                code == 6,
                                8.0,
                                tl.where(
                                    code == 7,
                                    10.0,
                                    tl.where(
                                        code == 8,
                                        0.0,
                                        tl.where(
                                            code == 9,
                                            -1.0,
                                            tl.where(
                                                code == 10,
                                                -2.0,
                                                tl.where(
                                                    code == 11,
                                                    -3.0,
                                                    tl.where(
                                                        code == 12,
                                                        -4.0,
                                                        tl.where(
                                                            code == 13,
                                                            -6.0,
                                                            tl.where(
                                                                code == 14,
                                                                -8.0,
                                                                -10.0,
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
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
    offs_k = tl.arange(0, 16)

    for block in range(0, num_blocks):
        packed_ptr = (
            w_ptr
            + offs_n[:, None] * stride_wn
            + block * BLOCK_BYTES
            + offs_k[None, :]
        )
        packed = tl.load(packed_ptr, mask=(offs_n[:, None] < n), other=0).to(tl.uint8)
        low = packed & 0xF
        high = packed >> 4
        scale_byte = tl.load(
            w_ptr + offs_n * stride_wn + block * BLOCK_BYTES + 16,
            mask=offs_n < n,
            other=0,
        ).to(tl.uint8)
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
        w_tile = tl.join(_codebook(low), _codebook(high)) * scale[:, None, None]
        x_low = tl.load(
            x_ptr + offs_m[:, None] * stride_xm
            + (block * BLOCK_SIZE + offs_k)[None, :] * stride_xk,
            mask=(offs_m[:, None] < m),
            other=0.0,
        )
        x_high = tl.load(
            x_ptr + offs_m[:, None] * stride_xm
            + (block * BLOCK_SIZE + offs_k + 16)[None, :] * stride_xk,
            mask=(offs_m[:, None] < m),
            other=0.0,
        )
        x_tile = tl.reshape(tl.join(x_low, x_high), (BLOCK_M, BLOCK_SIZE))
        w_tile = tl.reshape(w_tile, (BLOCK_N, BLOCK_SIZE)).to(x_tile.dtype)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def _decode_cpu(weights: torch.Tensor) -> torch.Tensor:
    blocks = weights.reshape(weights.shape[0], -1, BLOCK_BYTES)
    packed = blocks[..., :16]
    codes = torch.cat((packed & 0xF, packed >> 4), dim=-1).to(torch.float32)
    codebook = torch.tensor(
        (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10),
        dtype=torch.float32,
        device=weights.device,
    )
    codes = codebook[codes.to(torch.long)]
    scale_byte = blocks[..., 16].to(torch.int32)
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    scale = torch.where(
        scale_byte > 0x7E,
        torch.zeros_like(scale_byte, dtype=torch.float32),
        torch.where(
            exponent == 0,
            mantissa.float() / 1024.0,
            (1.0 + mantissa.float() / 8.0) * torch.pow(2.0, exponent.float() - 8.0),
        ),
    )
    return (codes * scale[..., None]).reshape(weights.shape[0], -1)


def ggml_gemm_q4_0_rocmfp4_fast_triton(
    W: torch.Tensor, X: torch.Tensor, row: int
) -> torch.Tensor:
    if W.dim() != 2:
        raise ValueError(f"ROCmFP4_FAST weights must be 2D, got {W.dim()}D")
    if W.dtype is not torch.uint8:
        raise TypeError(f"ROCmFP4_FAST weights must be uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "ROCmFP4_FAST supports float16, bfloat16, and float32 activations, "
            f"got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if any(size <= 0 for size in X.shape):
        raise ValueError("ROCmFP4_FAST activations must have positive dimensions")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    if W.shape[0] <= 0 or W.shape[1] <= 0:
        raise ValueError("ROCmFP4_FAST weights must have positive rows and width")
    if W.shape[1] % BLOCK_BYTES:
        raise ValueError(f"ROCmFP4_FAST row width must be divisible by {BLOCK_BYTES}")
    hidden_size = W.shape[1] // BLOCK_BYTES * BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match weight hidden size "
            f"{hidden_size}"
        )
    if W.device != X.device:
        raise ValueError(
            "ROCmFP4_FAST weights and activations must be on the same device"
        )

    W = W.contiguous()
    x_shape = X.shape
    X = X.reshape(-1, hidden_size).contiguous()
    if not X.is_cuda:
        return (X.float() @ _decode_cpu(W).T).to(X.dtype).view(*x_shape[:-1], row)

    Y = torch.empty((X.shape[0], row), device=X.device, dtype=X.dtype)
    _gemm_kernel[(triton.cdiv(X.shape[0], 16), triton.cdiv(row, 16))](
        X, W, Y, X.shape[0], row, hidden_size // BLOCK_SIZE,
        X.stride(0), X.stride(1), W.stride(0), Y.stride(0), Y.stride(1),
        BLOCK_M=16, BLOCK_N=16, num_warps=1, num_stages=2,
    )
    return Y.view(*x_shape[:-1], row)
