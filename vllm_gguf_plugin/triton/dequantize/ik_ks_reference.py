"""CPU/PyTorch reference decoders for ik_llama KS formats."""

from __future__ import annotations

import torch

from ...ik_types import (
    GGML_TYPE_IQ2_KS,
    GGML_TYPE_IQ3_KS,
    GGML_TYPE_IQ4_KS,
    GGML_TYPE_IQ5_KS,
    QK_IQ2_KS,
    QK_IQ3_KS,
    QK_IQ4_KS,
    QK_IQ5_KS,
)

_IQ2_VALUES = (-31, -13, 1, 17, -26, -8, 6, 22)
_IQ3_VALUES = (-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51)
_IQ4_VALUES = (
    -127,
    -104,
    -83,
    -65,
    -49,
    -35,
    -22,
    -10,
    1,
    13,
    25,
    38,
    53,
    69,
    89,
    113,
    -123,
    -100,
    -79,
    -61,
    -45,
    -31,
    -18,
    -6,
    5,
    17,
    29,
    42,
    57,
    73,
    93,
    117,
)
_IQ5_VALUES = (
    -126,
    -114,
    -103,
    -92,
    -83,
    -74,
    -65,
    -57,
    -50,
    -43,
    -36,
    -30,
    -24,
    -18,
    -12,
    -6,
    -1,
    5,
    11,
    17,
    23,
    29,
    36,
    43,
    51,
    59,
    68,
    77,
    87,
    97,
    109,
    121,
    -124,
    -112,
    -101,
    -90,
    -81,
    -72,
    -63,
    -55,
    -48,
    -41,
    -34,
    -28,
    -22,
    -16,
    -10,
    -4,
    1,
    7,
    13,
    19,
    25,
    31,
    38,
    45,
    53,
    61,
    70,
    79,
    89,
    99,
    111,
    123,
)


def _rows(W: torch.Tensor, m: int, n: int, qk: int, prefix: int, block: int):
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
    blocks = n // qk
    expected = m * (prefix + blocks * block)
    if W.numel() != expected:
        raise ValueError(
            f"Quantized weights must contain exactly {expected} bytes for shape "
            f"({m}, {n}), got {W.numel()}"
        )
    return W.reshape(m, prefix + blocks * block), blocks


def _table(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def dequantize_iq2_ks_reference(W, m, n, dtype=None):
    rows, blocks = _rows(W, m, n, QK_IQ2_KS, 2, 70)
    scale = rows[:, :2].contiguous().view(torch.float16).float().reshape(m)
    data = rows[:, 2:].reshape(m, blocks, 70)
    extra = data[:, :, :2].contiguous().view(torch.int16).to(torch.int32).squeeze(-1)
    scales, qs = data[:, :, 2:6], data[:, :, 6:]
    out = torch.empty((m, blocks, 256), dtype=torch.float32, device=W.device)
    values = _table(_IQ2_VALUES, W.device)
    for ib in range(4):
        ex = extra >> (2 * ib)
        dl1 = scale[:, None] * (
            ((scales[:, :, ib] & 15) | ((ex >> 4) & 16)) - 16
        )
        dl2 = scale[:, None] * (
            ((scales[:, :, ib] >> 4) | ((ex >> 5) & 16)) - 16
        )
        q = qs[:, :, 32 * (ib // 2) : 32 * (ib // 2) + 32].to(torch.int32)
        shift = 4 * (ib & 1)
        out[:, :, 64 * ib : 64 * ib + 32] = dl1[:, :, None] * values[
            4 * (ex & 1)[:, :, None] + ((q >> shift) & 3)
        ]
        out[:, :, 64 * ib + 32 : 64 * ib + 64] = (
            dl2[:, :, None]
            * values[
                4 * ((ex >> 1) & 1)[:, :, None]
                + ((q >> (shift + 2)) & 3)
            ]
        )
    return out.reshape(m, n).to(dtype or torch.float16)


def dequantize_iq3_ks_reference(W, m, n, dtype=None):
    rows, blocks = _rows(W, m, n, QK_IQ3_KS, 2, 102)
    scale = rows[:, :2].contiguous().view(torch.float16).float().reshape(m)
    data = rows[:, 2:].reshape(m, blocks, 102)
    extra = data[:, :, :2].contiguous().view(torch.int16).to(torch.int32)
    scales, qs, qh = data[:, :, 2:6], data[:, :, 6:70], data[:, :, 70:]
    out = torch.empty((m, blocks, 256), dtype=torch.float32, device=W.device)
    values = _table(_IQ3_VALUES, W.device)
    for ib in range(8):
        j = ib & 3
        sc = (scales[:, :, j] >> (4 if ib >= 4 else 0)) & 15
        sc |= ((extra >> (j + (4 if ib >= 4 else 0))) & 1).squeeze(-1) << 4
        cb = (((extra >> (8 + ib)) & 1) * 8).squeeze(-1)
        q = qs[:, :, 32 * (ib // 4) : 32 * (ib // 4) + 32].to(torch.int32)
        code = ((q >> (2 * (ib & 3))) & 3) | (
            ((qh[:, :, :32] >> (4 * (ib // 4) + (ib & 3))) & 1) << 2
        )
        out[:, :, 32 * ib : 32 * ib + 32] = (
            scale[:, None, None]
            * (sc[:, :, None] - 16)
            * values[code + cb[:, :, None]]
        )
    return out.reshape(m, n).to(dtype or torch.float16)


def dequantize_iq4_ks_reference(W, m, n, dtype=None):
    rows, blocks = _rows(W, m, n, QK_IQ4_KS, 4, 136)
    scale = rows[:, :4].contiguous().view(torch.float32).reshape(m)
    data = rows[:, 4:].reshape(m, blocks, 136)
    scales, qs = data[:, :, :8], data[:, :, 8:]
    out = torch.empty((m, blocks, 256), dtype=torch.float32, device=W.device)
    values = _table(_IQ4_VALUES, W.device)
    for ib in range(8):
        s = scales[:, :, ib].to(torch.int32)
        dl = scale[:, None] * ((s & 254) - 127)
        q = qs[:, :, 16 * ib : 16 * ib + 16].to(torch.int32)
        cb = (s & 1) * 16
        out[:, :, 32 * ib : 32 * ib + 16] = (
            dl[:, :, None] * values[(q & 15) + cb[:, :, None]]
        )
        out[:, :, 32 * ib + 16 : 32 * ib + 32] = (
            dl[:, :, None] * values[(q >> 4) + cb[:, :, None]]
        )
    return out.reshape(m, n).to(dtype or torch.float16)


def dequantize_iq5_ks_reference(W, m, n, dtype=None):
    rows, blocks = _rows(W, m, n, QK_IQ5_KS, 4, 168)
    scale = rows[:, :4].contiguous().view(torch.float32).reshape(m)
    data = rows[:, 4:].reshape(m, blocks, 168)
    scales, qs, qh = data[:, :, :8], data[:, :, 8:136], data[:, :, 136:]
    out = torch.empty((m, blocks, 256), dtype=torch.float32, device=W.device)
    values = _table(_IQ5_VALUES, W.device)
    for ib in range(4):
        s1, s2 = (
            scales[:, :, 2 * ib].to(torch.int32),
            scales[:, :, 2 * ib + 1].to(torch.int32),
        )
        q = qs[:, :, 32 * ib : 32 * ib + 32].to(torch.int32)
        high1 = ((qh[:, :, :32] >> (2 * ib)) & 1).to(torch.int32)
        high2 = ((qh[:, :, :32] >> (2 * ib + 1)) & 1).to(torch.int32)
        out[:, :, 64 * ib : 64 * ib + 32] = (
            scale[:, None, None]
            * ((s1 & 254) - 127)[:, :, None]
            * values[(q & 15) + 16 * high1]
        )
        out[:, :, 64 * ib + 32 : 64 * ib + 64] = (
            scale[:, None, None]
            * ((s2 & 254) - 127)[:, :, None]
            * values[(q >> 4) + 16 * high2]
        )
    return out.reshape(m, n).to(dtype or torch.float16)


REFERENCE_DECODERS = {
    GGML_TYPE_IQ2_KS: dequantize_iq2_ks_reference,
    GGML_TYPE_IQ3_KS: dequantize_iq3_ks_reference,
    GGML_TYPE_IQ4_KS: dequantize_iq4_ks_reference,
    GGML_TYPE_IQ5_KS: dequantize_iq5_ks_reference,
}
