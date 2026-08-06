# SPDX-License-Identifier: Apache-2.0
"""Dedicated dense GEMM for GGML type 39 (MXFP4).

Structurally this is ROCmFP4_FAST (type 101) with two constants changed: the
block is 17 bytes holding 32 E2M1 codes plus one trailing scale byte, low
nibbles decoding outputs 0..15 and high nibbles 16..31. What differs is the
codebook's eighth entry (12 rather than 10) and the scale encoding -- E8M0,
a pure power of two, where ROCmFPX uses UE4M3.

The E8M0 byte decodes as ``2 ** (byte - 128)``. llama.cpp writes that as
``bits = (byte - 1) << 23``: the exponent field sits one below the byte and
fp32 subtracts its own bias of 127, which folds in the halving that matches
``kvalues_fp4`` being stored at twice the true E2M1 values.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 17
BLOCK_SIZE = 32
QS_BYTES = 16
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

GEMM_BLOCK_M = 32
GEMM_BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2

# kvalues_fp4 from ggml-common.h, shared with NVFP4. Stored at twice the true
# E2M1 values, which the scale decoders undo.
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


@triton.jit
def _codebook(code):
    """Map a 4-bit E2M1 index to its (doubled) value."""
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
                                    12.0,
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
                                                                -12.0,
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
def _e8m0_scale(scale_byte):
    """Decode an E8M0 scale byte to ``2 ** (byte - 128)``.

    Bytes 0 and 1 encode subnormal patterns around 1e-38. Computing them with
    the same exponential is correct to the limits of float32 and avoids a
    branch that would only ever fire on degenerate weights.
    """
    return tl.exp2(scale_byte.to(tl.float32) - 128.0)


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
    offs_k = tl.arange(0, QS_BYTES)

    for block in range(0, num_blocks):
        packed_ptr = (
            w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES + offs_k[None, :]
        )
        packed = tl.load(packed_ptr, mask=(offs_n[:, None] < n), other=0).to(tl.uint8)
        low = packed & 0xF
        high = packed >> 4

        scale_byte = tl.load(
            w_ptr + offs_n * stride_wn + block * BLOCK_BYTES + QS_BYTES,
            mask=offs_n < n,
            other=0,
        ).to(tl.uint8)
        scale = _e8m0_scale(scale_byte)

        # tl.join yields (BLOCK_N, 16, 2), so the scale must broadcast at rank
        # three or it only reaches the final pair dimension.
        w_tile = tl.join(_codebook(low), _codebook(high)) * scale[:, None, None]

        x_low = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + ((block * BLOCK_SIZE + offs_k)[None, :]) * stride_xk,
            mask=(offs_m[:, None] < m),
            other=0.0,
        )
        x_high = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + ((block * BLOCK_SIZE + offs_k + QS_BYTES)[None, :]) * stride_xk,
            mask=(offs_m[:, None] < m),
            other=0.0,
        )
        x_tile = tl.reshape(tl.join(x_low, x_high), (BLOCK_M, BLOCK_SIZE))
        # tl.dot requires matching operand dtypes.
        w_tile = tl.reshape(w_tile, (BLOCK_N, BLOCK_SIZE)).to(x_tile.dtype)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def _decode_cpu(weights: torch.Tensor) -> torch.Tensor:
    """Reference decode for hosts without a Triton runtime."""
    blocks = weights.reshape(weights.shape[0], -1, BLOCK_BYTES)
    packed = blocks[..., :QS_BYTES]
    codes = torch.cat((packed & 0xF, packed >> 4), dim=-1).to(torch.long)
    codebook = torch.tensor(CODEBOOK, dtype=torch.float32, device=weights.device)
    values = codebook[codes]
    scale_byte = blocks[..., QS_BYTES].to(torch.float32)
    scale = torch.pow(2.0, scale_byte - 128.0)
    return (values * scale[..., None]).reshape(weights.shape[0], -1)


def ggml_gemm_mxfp4_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    """Dense GEMM for MXFP4-packed weights.

    ``W`` is ``[row, blocks * 17]`` packed uint8; ``X`` is ``[..., hidden]``
    with hidden a multiple of 32. Returns ``[..., row]`` in ``X``'s dtype.
    """
    if W.dim() != 2:
        raise ValueError(f"packed weights must be 2-D, got {W.dim()}-D")
    if W.dtype != torch.uint8:
        raise ValueError(f"packed weights must be uint8, got {W.dtype}")
    if X.dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise ValueError(f"unsupported activation dtype {X.dtype}")
    if any(size <= 0 for size in X.shape):
        raise ValueError(
            f"activation dimensions must be positive, got {tuple(X.shape)}"
        )
    if row != W.shape[0]:
        raise ValueError(f"row {row} does not match weight rows {W.shape[0]}")

    packed_width = W.shape[1]
    if packed_width % BLOCK_BYTES:
        raise ValueError(
            f"packed width {packed_width} is not a multiple of {BLOCK_BYTES}"
        )
    num_blocks = packed_width // BLOCK_BYTES
    hidden = num_blocks * BLOCK_SIZE
    if X.shape[-1] != hidden:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match weight hidden size {hidden}"
        )

    leading = X.shape[:-1]
    x2d = X.reshape(-1, hidden).contiguous()
    w = W.contiguous()

    if not x2d.is_cuda:
        decoded = _decode_cpu(w).to(x2d.dtype)
        return (x2d @ decoded.T).reshape(*leading, row)

    out = torch.empty((x2d.shape[0], row), dtype=x2d.dtype, device=x2d.device)
    grid = (
        triton.cdiv(x2d.shape[0], GEMM_BLOCK_M),
        triton.cdiv(row, GEMM_BLOCK_N),
    )
    _gemm_kernel[grid](
        x2d,
        w,
        out,
        x2d.shape[0],
        row,
        num_blocks,
        x2d.stride(0),
        x2d.stride(1),
        w.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=GEMM_BLOCK_M,
        BLOCK_N=GEMM_BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out.reshape(*leading, row)
