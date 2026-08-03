"""CPU/PyTorch reference decoders for ik_llama IQ4_KSS and IQ2_KL."""

from __future__ import annotations

import torch

from ...ik_types import (
    GGML_TYPE_IQ2_KL,
    GGML_TYPE_IQ4_KSS,
    IQ2_KL_BLOCK_BYTES,
    IQ4_KSS_BLOCK_BYTES,
    QK_IQ2_KL,
    QK_IQ4_KSS,
)


def _validate_rows(
    W: torch.Tensor, m: int, n: int, qk: int, block_bytes: int, prefix_bytes: int
) -> tuple[torch.Tensor, int]:
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if m <= 0 or n <= 0 or n % qk:
        raise ValueError(f"Invalid reference dequant shape ({m}, {n})")
    if not W.is_contiguous():
        raise ValueError("Quantized weights must be contiguous")
    blocks_per_row = n // qk
    expected = m * (prefix_bytes + blocks_per_row * (block_bytes - prefix_bytes))
    if W.numel() != expected:
        raise ValueError(
            f"Quantized weights have {W.numel()} bytes, but shape ({m}, {n}) "
            f"requires {expected} bytes"
        )
    return W.reshape(m, expected // m), blocks_per_row


def dequantize_iq4_kss_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode row-prefix IQ4_KSS using upstream XOR/Grey reconstruction."""
    rows, blocks_per_row = _validate_rows(W, m, n, QK_IQ4_KSS, IQ4_KSS_BLOCK_BYTES, 4)
    prefix = rows[:, :4].contiguous().view(torch.float32).reshape(m)
    blocks = rows[:, 4:].reshape(m, blocks_per_row, 128)
    packed = blocks.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    packed = packed.reshape(m, blocks_per_row, 8, 8)
    ls = ((packed & 1) << torch.arange(8, device=W.device)).sum(dim=-1)
    aux = (packed & 0xFFFE) ^ ((packed & 0xFFFE) >> 1)
    aux_bytes = torch.stack((aux & 0xFF, aux >> 8), dim=-1).reshape(
        m, blocks_per_row, 8, 16
    )
    codebook = torch.tensor(
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
        device=W.device,
        dtype=torch.float32,
    )
    indices = torch.cat((aux_bytes & 15, aux_bytes >> 4), dim=-1)
    values = codebook[indices + (ls & 1).unsqueeze(-1) * 16]
    scale = (ls & 254).to(torch.float32) - 127
    output = (values * scale.unsqueeze(-1)).reshape(m, blocks_per_row, 256)
    output = output * prefix[:, None, None]
    return output.reshape(m, n).to(dtype or torch.float16)


def dequantize_iq2_kl_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode row-prefix IQ2_KL scales and packed pair codebook values."""
    rows, blocks_per_row = _validate_rows(W, m, n, QK_IQ2_KL, IQ2_KL_BLOCK_BYTES, 2)
    prefix = rows[:, :2].contiguous().view(torch.float16).to(torch.float32).reshape(m)
    block_bytes = rows[:, 2:].reshape(m, blocks_per_row, 86)
    blocks = block_bytes.to(torch.int64)
    scales_h = (
        block_bytes[:, :, :2]
        .contiguous()
        .view(torch.int16)
        .to(torch.int64)
        .squeeze(-1)
        & 0xFFFF
    )
    scales_l = blocks[:, :, 2:6]
    qs = blocks[:, :, 6:70]
    qh = blocks[:, :, 70:86]
    codebook = torch.tensor(
        [
            0xE9C1,
            0x0DC1,
            0xC1D8,
            0xF6D8,
            0x0DD8,
            0x2FD8,
            0xD8E9,
            0xE9E9,
            0x01E9,
            0x0DE9,
            0x1CE9,
            0xC1F6,
            0x01F6,
            0x0DF6,
            0x2FF6,
            0xE901,
            0xF601,
            0x0101,
            0x0D01,
            0x1C01,
            0xD80D,
            0xE90D,
            0xF60D,
            0x010D,
            0x0D0D,
            0xC11C,
            0xE91C,
            0x011C,
            0x1C1C,
            0x2F1C,
            0xE92F,
            0x0D2F,
        ],
        device=W.device,
        dtype=torch.int32,
    )
    outputs = []
    for ib in range(4):
        sl1 = ((scales_l[:, :, (2 * ib) % 4] >> (4 * (ib // 2))) & 15) | (
            ((scales_h >> (4 * ib)) & 3) << 4
        )
        sl2 = ((scales_l[:, :, (2 * ib + 1) % 4] >> (4 * (ib // 2))) & 15) | (
            ((scales_h >> (4 * ib + 2)) & 3) << 4
        )
        high = (qh >> (2 * ib)) & 1
        indices1 = (qs[:, :, ib * 16 : (ib + 1) * 16] & 15) | (high << 4)
        high = (qh >> (2 * ib + 1)) & 1
        indices2 = (qs[:, :, ib * 16 : (ib + 1) * 16] >> 4) | (high << 4)
        pair1 = codebook[indices1]
        pair2 = codebook[indices2]
        values1 = torch.stack((pair1 & 255, pair1 >> 8), -1)
        values2 = torch.stack((pair2 & 255, pair2 >> 8), -1)
        values1 = torch.where(values1 < 128, values1, values1 - 256).float()
        values2 = torch.where(values2 < 128, values2, values2 - 256).float()
        first = values1.reshape(m, blocks_per_row, 32)
        first = first * (sl1 - 32).float()[:, :, None] * prefix[:, None, None]
        second = values2.reshape(m, blocks_per_row, 32)
        second = second * (sl2 - 32).float()[:, :, None] * prefix[:, None, None]
        outputs.append(torch.cat((first, second), -1))
    return torch.cat(outputs, -1).reshape(m, n).to(dtype or torch.float16)


REFERENCE_DECODERS = {
    GGML_TYPE_IQ4_KSS: dequantize_iq4_kss_reference,
    GGML_TYPE_IQ2_KL: dequantize_iq2_kl_reference,
}
