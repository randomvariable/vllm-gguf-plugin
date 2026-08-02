"""CPU/PyTorch reference decoders for ik_llama BitNet BN formats."""

from __future__ import annotations

import torch

from ...ik_types import (
    GGML_TYPE_IQ1_BN,
    GGML_TYPE_IQ2_BN,
    IQ1_BN_BLOCK_BYTES,
    IQ2_BN_BLOCK_BYTES,
    QK_IQ1BN,
    QK_IQ2BN,
)


def _validate_row_storage(
    W: torch.Tensor,
    m: int,
    n: int,
    qk: int,
    block_bytes: int,
    row_meta_size: int,
) -> tuple[torch.Tensor, int]:
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if m <= 0 or n <= 0:
        raise ValueError(
            f"Reference dequant shape must have positive dimensions, got ({m}, {n})"
        )
    if n % qk:
        raise ValueError(
            f"Reference dequant columns must be divisible by {qk}, got {n}"
        )
    if not W.is_contiguous():
        raise ValueError("Quantized weights must be contiguous")
    blocks_per_row = n // qk
    expected = m * (row_meta_size + blocks_per_row * block_bytes)
    if W.numel() != expected:
        raise ValueError(
            f"Quantized weights must contain exactly {expected} bytes for shape "
            f"({m}, {n}), "
            f"got {W.numel()}"
        )
    return W.reshape(m, row_meta_size + blocks_per_row * block_bytes), blocks_per_row


def dequantize_iq1_bn_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode ``IQ1_BN`` rows with one FP16 scale prefix per row."""
    rows, blocks_per_row = _validate_row_storage(
        W, m, n, QK_IQ1BN, IQ1_BN_BLOCK_BYTES, 2
    )
    scale = rows[:, :2].contiguous().view(torch.float16).to(torch.float32)
    blocks = rows[:, 2:].reshape(m, blocks_per_row, IQ1_BN_BLOCK_BYTES)
    ql = blocks[:, :, :12].to(torch.int32)
    extra = blocks[:, :, 12].to(torch.int32)
    positions = torch.arange(QK_IQ1BN, device=W.device)
    groups = positions // 16
    lanes = positions % 16
    q_index = groups * 3 + torch.minimum(lanes // 5, torch.tensor(2, device=W.device))
    q = ql[:, :, q_index]
    q = torch.where(lanes == 15, extra[:, :, groups], q)
    digit = torch.where(lanes == 15, groups, lanes % 5)
    multipliers = torch.tensor((81, 27, 9, 3, 1), device=W.device, dtype=torch.int32)
    v = q * multipliers[digit]
    ternary = ((v + (v >> 1)) >> 7) - 1
    output = ternary.to(torch.float32) * scale[:, None]
    return output.reshape(m, n).to(dtype or torch.float16)


def dequantize_iq2_bn_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode ``IQ2_BN`` rows with one FP32 scale prefix per row."""
    rows, blocks_per_row = _validate_row_storage(
        W, m, n, QK_IQ2BN, IQ2_BN_BLOCK_BYTES, 4
    )
    scale = rows[:, :4].contiguous().view(torch.float32)
    blocks = rows[:, 4:].reshape(m, blocks_per_row, IQ2_BN_BLOCK_BYTES)
    qs = blocks.to(torch.int32)
    positions = torch.arange(QK_IQ2BN, device=W.device)
    codes = (qs[:, :, positions % 16] >> (2 * (positions // 16))) & 3
    output = (codes.to(torch.float32) - 1) * scale[:, None]
    return output.reshape(m, n).to(dtype or torch.float16)


REFERENCE_DECODERS = {
    GGML_TYPE_IQ1_BN: dequantize_iq1_bn_reference,
    GGML_TYPE_IQ2_BN: dequantize_iq2_bn_reference,
}
