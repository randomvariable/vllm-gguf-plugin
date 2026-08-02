from __future__ import annotations

import torch

QK_IQ2_K = 256
IQ2_K_BLOCK_BYTES = 76
QK_IQ3_K = 256
IQ3_K_BLOCK_BYTES = 110
QK_IQ4_K = 256
IQ4_K_BLOCK_BYTES = 144

_IQ2NL_VALUES = torch.tensor([-31, -13, 1, 17, -26, -8, 6, 22], dtype=torch.float32)
_IQ3NL_VALUES = torch.tensor(
    [-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51],
    dtype=torch.float32,
)
_IQ4K_VALUES = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
)


def _validate(W: torch.Tensor, m: int, n: int, block_bytes: int) -> torch.Tensor:
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if not W.is_contiguous():
        raise ValueError("Quantized weights must be contiguous")
    if m <= 0 or n <= 0 or n % 256:
        raise ValueError(f"Invalid reference dequant shape ({m}, {n})")
    expected = m * n // 256 * block_bytes
    if W.numel() != expected:
        raise ValueError(
            f"Quantized weights have {W.numel()} bytes, but shape ({m}, {n}) "
            f"requires {expected} bytes"
        )
    return W.reshape(-1).reshape(-1, block_bytes)


def _finish(
    values: torch.Tensor, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    return values.reshape(m, n).to(dtype or torch.float16)


def dequantize_iq2_k_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    blocks = _validate(W, m, n, IQ2_K_BLOCK_BYTES)
    d = blocks[:, :2].contiguous().view(torch.float16).float()
    extra = blocks[:, 2] | (blocks[:, 3] << 8)
    scales = blocks[:, 4:12]
    qs = blocks[:, 12:]
    values = []
    for ib in range(8):
        packed = qs[:, (ib // 4) * 32 : (ib // 4 + 1) * 32]
        shift = 2 * (ib % 4)
        codes = (packed >> shift) & 3
        codebook = (extra >> (2 * ib)) & 3
        table = _IQ2NL_VALUES.to(W.device)
        first = table[((codebook & 1) * 4)[:, None] + codes[:, :16]]
        second = table[((codebook >> 1) * 4)[:, None] + codes[:, 16:]]
        scale = torch.stack(((scales[:, ib] & 15) - 8, (scales[:, ib] >> 4) - 8), dim=1)
        values.append(
            torch.cat((first * scale[:, 0, None], second * scale[:, 1, None]), dim=1)
            * d[:, None]
        )
    return _finish(torch.cat(values, dim=1), m, n, dtype)


def dequantize_iq3_k_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    blocks = _validate(W, m, n, IQ3_K_BLOCK_BYTES)
    d = blocks[:, :2].contiguous().view(torch.float16).float()
    extra = blocks[:, 2] | (blocks[:, 3] << 8)
    scales_h = blocks[:, 4] | (blocks[:, 5] << 8)
    scales_l, qs, qh = blocks[:, 6:14], blocks[:, 14:78], blocks[:, 78:]
    values = []
    table = _IQ3NL_VALUES.to(W.device)
    for ib in range(8):
        packed = qs[:, (ib // 4) * 32 : (ib // 4 + 1) * 32]
        low = (packed >> (2 * (ib % 4))) & 3
        high = (qh >> ib) & 1
        codes = low | (high << 2)
        book = (extra >> (2 * ib)) & 3
        first = table[((book & 1) * 8)[:, None] + codes[:, :16]]
        second = table[((book >> 1) * 8)[:, None] + codes[:, 16:]]
        sh = scales_h >> (2 * ib)
        s1 = (2 * (scales_l[:, ib] & 15) + 1) * torch.where((sh & 1) != 0, -1, 1)
        s2 = (2 * (scales_l[:, ib] >> 4) + 1) * torch.where((sh & 2) != 0, -1, 1)
        values.append(
            torch.cat((first * s1[:, None], second * s2[:, None]), dim=1) * d[:, None]
        )
    return _finish(torch.cat(values, dim=1), m, n, dtype)


def dequantize_iq4_k_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    blocks = _validate(W, m, n, IQ4_K_BLOCK_BYTES)
    d = blocks[:, :2].contiguous().view(torch.float16).float()
    extra = blocks[:, 2] | (blocks[:, 3] << 8)
    scales_h, scales_l, qs = blocks[:, 4:8], blocks[:, 8:16], blocks[:, 16:]
    table = _IQ4K_VALUES.to(W.device)
    values = []
    for ib in range(8):
        sh = scales_h[:, ib // 2] >> (4 * (ib % 2))
        s1 = (scales_l[:, ib] & 15) | ((sh << 4) & 0x30)
        s2 = (scales_l[:, ib] >> 4) | ((sh << 2) & 0x30)
        s1, s2 = s1 - 32, s2 - 32
        packed = qs[:, ib * 16 : (ib + 1) * 16]
        book = (extra >> (2 * ib)) & 3
        first = table[((book & 1) * 16)[:, None] + (packed & 15)]
        second = table[((book >> 1) * 16)[:, None] + (packed >> 4)]
        values.append(
            torch.cat((first * s1[:, None], second * s2[:, None]), dim=1) * d[:, None]
        )
    return _finish(torch.cat(values, dim=1), m, n, dtype)


REFERENCE_DECODERS = {
    137: dequantize_iq2_k_reference,
    138: dequantize_iq3_k_reference,
    139: dequantize_iq4_k_reference,
}
