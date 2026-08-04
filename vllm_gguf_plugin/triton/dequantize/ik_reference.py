"""Portable reference decoders for exact ik_llama.cpp block layouts."""

from __future__ import annotations

import torch

from ..gemm.utils import GGML_TYPE_I2_S, GGML_TYPE_Q6_0

Q6_0_BLOCK_BYTES = 26
I2_S_ROW_SCALE_BYTES = 4


def _validate_input(
    W: torch.Tensor, m: int, n: int, block_qk: int, expected_bytes: int
) -> torch.Tensor:
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if m <= 0 or n <= 0 or n % block_qk:
        raise ValueError(f"Invalid reference dequant shape ({m}, {n})")
    if W.numel() != expected_bytes:
        raise ValueError(
            f"Quantized weights have {W.numel()} bytes, but shape ({m}, {n}) "
            f"requires {expected_bytes} bytes"
        )
    return W.contiguous().reshape(-1)


def dequantize_q6_0_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    raw = _validate_input(W, m, n, 32, m * n // 32 * Q6_0_BLOCK_BYTES)
    blocks = raw.reshape(-1, Q6_0_BLOCK_BYTES)
    # .view(float16) yields (nblocks, 1); without squeeze, d[:, None] is rank 3
    # and broadcasts into an (nblocks, nblocks, 32) outer product.
    d = blocks[:, :2].contiguous().view(torch.float16).squeeze(1).to(torch.float32)
    qh = blocks[:, 2:10].to(torch.int32)
    qs = blocks[:, 10:].to(torch.int32)
    i = torch.arange(16, device=W.device)
    high = qh[:, i.remainder(8)] >> (4 * (i // 8))
    first = ((qs & 0x0F) | ((high << 4) & 0x30)) - 32
    second = ((qs >> 4) | ((high << 2) & 0x30)) - 32
    values = torch.cat((first, second), dim=1)
    return (
        (values.to(torch.float32) * d[:, None]).reshape(m, n).to(dtype or torch.float16)
    )


def dequantize_i2_s_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    row_bytes = n // 4 + I2_S_ROW_SCALE_BYTES
    raw = _validate_input(W, m, n, 128, m * row_bytes)
    rows = raw.reshape(m, row_bytes)
    scale = rows[:, n // 4 :].contiguous().view(torch.float32)
    # Group order is per 128-value block, not per row: native emits
    # y[block*128 + group*32 + j], so planes must be nested inside blocks.
    blocks_per_row = n // 128
    packed = rows[:, : n // 4].to(torch.int32).reshape(m, blocks_per_row, 32)
    planes = torch.stack([((packed >> shift) & 3) - 1 for shift in (6, 4, 2, 0)], dim=2)
    values = planes.reshape(m, n)
    # scale is already (m, 1) from the float32 view; scale[:, None] would make
    # it (m, 1, 1) and broadcast into an (m, m, n) outer product.
    return (values.to(torch.float32) * scale).to(dtype or torch.float16)


REFERENCE_DECODERS = {
    GGML_TYPE_I2_S: dequantize_i2_s_reference,
    GGML_TYPE_Q6_0: dequantize_q6_0_reference,
}
