"""Dedicated dense GEMM for GGML type 104 (Q3_0_ROCMFPX).

Type 104 packs 32 3-bit codes into 12 bytes followed by two UE4M3
half-scales (bytes 12 and 13), giving a 14-byte block. Code ``j`` occupies
bits ``[3j, 3j+3)`` of the 12-byte payload, so codes may straddle a byte
boundary. The first scale covers outputs 0..15 and the second covers
outputs 16..31. Reserved scale bytes ``0x7F..0xFF`` decode to zero.
"""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 14
BLOCK_SIZE = 32
CODE_BITS = 3
PAYLOAD_BYTES = 12
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Launch configuration measured on gfx1151 (Radeon 8060S) at rows=11008,
# hidden=4096, M=1. Widening BLOCK_N to 32 measured 225us (87.8 GB/s) versus
# 245us (80.7 GB/s) with BLOCK_N=16.
LAUNCH_BLOCK_M = 16
LAUNCH_BLOCK_N = 32
LAUNCH_NUM_WARPS = 1
LAUNCH_NUM_STAGES = 2


@triton.jit
def _codebook(code):
    # [0, 1, 2, 4, 0, -1, -2, -4]
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
                    4.0,
                    tl.where(
                        code == 4,
                        0.0,
                        tl.where(
                            code == 5,
                            -1.0,
                            tl.where(code == 6, -2.0, -4.0),
                        ),
                    ),
                ),
            ),
        ),
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
    bit_offset = offs_k * CODE_BITS
    byte_index = bit_offset // 8
    bit_shift = bit_offset % 8

    for block in range(num_blocks):
        base = w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES
        low_byte = tl.load(
            base + byte_index[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.int32)
        # The high byte only contributes when a code straddles a boundary;
        # otherwise its shifted bits fall outside the 3-bit mask.
        high_byte = tl.load(
            base + tl.minimum(byte_index + 1, PAYLOAD_BYTES - 1)[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.int32)
        codes = (
            (low_byte >> bit_shift[None, :]) | (high_byte << (8 - bit_shift)[None, :])
        ) & 0x7

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
    bit_offset = offs * CODE_BITS
    byte_index = bit_offset // 8
    bit_shift = bit_offset % 8
    low = payload[..., byte_index]
    high = payload[..., torch.clamp(byte_index + 1, max=PAYLOAD_BYTES - 1)]
    codes = ((low >> bit_shift) | (high << (8 - bit_shift))) & 0x7

    codebook = torch.tensor(
        [0, 1, 2, 4, 0, -1, -2, -4], dtype=torch.float32, device=weights.device
    )
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


def ggml_gemm_q3_0_rocmfpx_triton(
    W: torch.Tensor, X: torch.Tensor, row: int
) -> torch.Tensor:
    if W.dim() != 2:
        raise ValueError(f"Q3_0_ROCMFPX weights must be 2D, got {W.dim()}D")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Q3_0_ROCMFPX weights must be uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Q3_0_ROCMFPX supports float16, bfloat16, and float32 activations, "
            f"got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if any(size <= 0 for size in X.shape):
        raise ValueError("Q3_0_ROCMFPX activations must have positive dimensions")
    if W.shape[0] <= 0 or W.shape[1] <= 0:
        raise ValueError("Q3_0_ROCMFPX weights must have positive dimensions")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    if W.shape[1] % BLOCK_BYTES:
        raise ValueError(f"Q3_0_ROCMFPX row width must be divisible by {BLOCK_BYTES}")
    hidden_size = W.shape[1] // BLOCK_BYTES * BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match weight hidden size "
            f"{hidden_size}"
        )
    if W.device != X.device:
        raise ValueError(
            "Q3_0_ROCMFPX weights and activations must be on the same device"
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
