# SPDX-License-Identifier: Apache-2.0
"""Dedicated dense GEMM for GGML type 40 (NVFP4).

NVFP4 shares MXFP4's E2M1 codebook but blocks differently: 64 weights in 36
bytes, as four leading UE4M3 scale bytes followed by 32 payload bytes. Each
16-weight sub-block owns one scale and eight payload bytes, and within a
sub-block low nibbles fill the first eight outputs and high nibbles the second
eight.

The scales lead here, where MXFP4's single scale trails.

A note on the name: NVIDIA's ModelOpt and compressed-tensors NVFP4 also carry
per-tensor global scales. The GGUF encoding does not -- ``quantize_row_nvfp4_ref``
derives each sub-block scale from that sub-block's own absmax and
``dequantize_row_nvfp4`` reads nothing else -- so no companion tensors are
needed here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 36
BLOCK_SIZE = 64
SUB_SIZE = 16
SUB_COUNT = BLOCK_SIZE // SUB_SIZE
SCALE_BYTES = SUB_COUNT
SUB_PAYLOAD_BYTES = SUB_SIZE // 2
PAYLOAD_BYTES = BLOCK_SIZE // 2
SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

GEMM_BLOCK_M = 32
GEMM_BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2

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
def _ue4m3_scale(scale_byte):
    """Decode a UE4M3 scale byte.

    Four exponent bits and three mantissa bits, halved to match the doubled
    codebook, giving an effective bias of 8 -- the same convention as ROCmFPX.
    ``0x00`` and ``0x7F`` both mean zero.
    """
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    return tl.where(
        (scale_byte == 0) | (scale_byte == 0x7F),
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

    # One lane per payload byte. Byte p belongs to sub-block p // 8 and carries
    # the codes for outputs sub*16 + (p % 8) and sub*16 + 8 + (p % 8).
    #
    # The whole block is reduced in one 64-wide dot rather than four 16-wide
    # ones, with each byte's scale gathered alongside it. Four narrow dots is
    # the more obvious shape and is correct on gfx1151, but mispairs operands
    # on sm_121 -- halving the effective K and doubling half the lanes -- so
    # the wide form is the portable one.
    offs_p = tl.arange(0, PAYLOAD_BYTES)
    sub_of_p = offs_p // SUB_PAYLOAD_BYTES
    x_low_idx = sub_of_p * SUB_SIZE + (offs_p % SUB_PAYLOAD_BYTES)
    x_high_idx = x_low_idx + SUB_PAYLOAD_BYTES

    for block in range(0, num_blocks):
        block_base = w_ptr + offs_n[:, None] * stride_wn + block * BLOCK_BYTES

        packed = tl.load(
            block_base + SCALE_BYTES + offs_p[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.uint8)

        # Gather per byte, so one tile carries all four sub-block scales.
        scale_byte = tl.load(
            block_base + sub_of_p[None, :],
            mask=(offs_n[:, None] < n),
            other=0,
        ).to(tl.uint8)
        scale = _ue4m3_scale(scale_byte)

        w_tile = tl.join(
            _codebook(packed & 0xF) * scale, _codebook(packed >> 4) * scale
        )

        x_low = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + ((block * BLOCK_SIZE + x_low_idx)[None, :]) * stride_xk,
            mask=(offs_m[:, None] < m),
            other=0.0,
        )
        x_high = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + ((block * BLOCK_SIZE + x_high_idx)[None, :]) * stride_xk,
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
    """Reference decode for hosts without a Triton runtime."""
    rows = weights.shape[0]
    blocks = weights.reshape(rows, -1, BLOCK_BYTES)
    scale_bytes = blocks[..., :SCALE_BYTES].to(torch.int32)
    payload = blocks[..., SCALE_BYTES:].reshape(rows, -1, SUB_COUNT, SUB_PAYLOAD_BYTES)

    codes = torch.cat((payload & 0xF, payload >> 4), dim=-1).to(torch.long)
    codebook = torch.tensor(CODEBOOK, dtype=torch.float32, device=weights.device)
    values = codebook[codes]

    exponent = (scale_bytes >> 3) & 0xF
    mantissa = scale_bytes & 7
    scale = torch.where(
        (scale_bytes == 0) | (scale_bytes == 0x7F),
        torch.zeros_like(scale_bytes, dtype=torch.float32),
        torch.where(
            exponent == 0,
            mantissa.float() / 1024.0,
            (1.0 + mantissa.float() / 8.0) * torch.pow(2.0, exponent.float() - 8.0),
        ),
    )

    return (values * scale[..., None]).reshape(rows, -1)


def ggml_gemm_nvfp4_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    """Dense GEMM for NVFP4-packed weights.

    ``W`` is ``[row, blocks * 36]`` packed uint8; ``X`` is ``[..., hidden]``
    with hidden a multiple of 64. Returns ``[..., row]`` in ``X``'s dtype.
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
