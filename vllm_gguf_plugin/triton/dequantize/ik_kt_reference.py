"""PyTorch reference decoders for the ik_llama.cpp KT formats."""

from __future__ import annotations

import struct

import torch

from ...ik_types import (
    GGML_TYPE_IQ1_KT,
    GGML_TYPE_IQ2_KT,
    GGML_TYPE_IQ3_KT,
    GGML_TYPE_IQ4_KT,
)

QK = 256
ROW_PREFIX_BYTES = 4
IQ1_KT_PAYLOAD_BYTES = 56
IQ2_KT_PAYLOAD_BYTES = 68
IQ3_KT_PAYLOAD_BYTES = 100
IQ4_KT_PAYLOAD_BYTES = 128
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
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _trellis_int(seed: int) -> list[float]:
    values = []
    seed += 4096
    for _ in range(8):
        seed = (seed * 0xCBAC1FED) & 0xFFFFFFFF
        chunks = [(seed >> shift) & 0x3F for shift in range(0, 32, 8)]
        values.append(abs(-126 + sum(chunks)))
    return values


def _trellis_fp16(seed: int, offset: int, count: int = 8) -> list[float]:
    values = []
    x = seed + offset
    for _ in range(count):
        x = (89226354 * x + 64248484) & 0xFFFFFFFF
        packed = ((x & 0x8FFF8FFF) ^ 0x3B603B60).to_bytes(4, "little")
        values.append(
            abs(
                struct.unpack("<e", packed[:2])[0]
                + struct.unpack("<e", packed[2:])[0]
            )
        )
    return values


def _rows(W: torch.Tensor, m: int, n: int, payload: int) -> list[list[int]]:
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if not W.is_contiguous():
        raise ValueError("Quantized weights must be contiguous")
    if m <= 0 or n <= 0 or n % QK:
        raise ValueError(f"Invalid reference dequant shape ({m}, {n})")
    expected = m * (ROW_PREFIX_BYTES + n // QK * payload)
    if W.numel() != expected:
        raise ValueError(
            f"Quantized weights require exactly {expected} bytes, got {W.numel()}"
        )
    return W.reshape(m, expected // m).tolist()


def _decode(
    W: torch.Tensor, m: int, n: int, dtype, payload: int, kind: int
) -> torch.Tensor:
    if dtype not in _DTYPES:
        raise TypeError(
            "KT reference dtype must be torch.float16, torch.bfloat16, or torch.float32"
        )
    result = []
    for row in _rows(W, m, n, payload):
        row_scale = struct.unpack("<f", bytes(row[:4]))[0]
        values = []
        for block in range(n // QK):
            data = row[4 + block * payload : 4 + (block + 1) * payload]
            if kind == 1:
                for i in range(32):
                    ib, ig = divmod(i, 4)
                    qh = data[40 + (ib % 4) * 4 + ig]
                    sh = data[ib]
                    seed = data[8 + i] | (((qh >> (4 * (ib // 4))) & 15) << 8)
                    seed |= ((sh >> (4 + ig)) & 1) << 12
                    values.extend(
                        _trellis_fp16(seed, 4096),
                    )
                    start = len(values) - 8
                    values[start:] = [
                        v * row_scale * 31.75 * _IQ4_VALUES[sh & 15]
                        for v in values[start:]
                    ]
            elif kind in (2, 3):
                seeds = [
                    int.from_bytes(bytes(data[4 + 2 * i : 6 + 2 * i]), "little")
                    for i in range(32)
                ]
                for half in range(2):
                    for i in range(16):
                        ib, ig = divmod(i, 4)
                        scale_byte = data[ib]
                        nibble = (scale_byte >> (4 * half)) & 15
                        generated = (
                            _trellis_fp16(seeds[half * 16 + i], 4096)
                            if kind == 2
                            else _trellis_int(seeds[half * 16 + i])
                        )
                        d = row_scale * 31.75 if kind == 2 else row_scale
                        factor = d * (_IQ4_VALUES[nibble] if kind == 2 else nibble)
                        for j, value in enumerate(generated):
                            if kind == 3:
                                qh = data[68 + ig * 8 + j]
                                value *= -1 if qh & (1 << (ib + 4 * half)) else 1
                            values.append(value * factor)
            else:
                headers = [
                    struct.unpack_from("<I", bytes(data), 4 * i)[0] for i in range(8)
                ]
                ql, qh = data[32:96], data[96:]
                for group in range(64):
                    header = headers[group // 8]
                    ib, ig = divmod(group, 8)
                    offset = 36864 if header & 1 else 4096
                    scale = row_scale * 31.75 * (((header & 0xFF) >> 1) - 64)
                    seed = ql[group] | (
                        ((qh[group % 32] >> (4 * (group // 32))) & 15) << 8
                    )
                    seed |= ((header >> (8 + 3 * ig)) & 7) << 12
                    values.extend(
                        v * scale for v in _trellis_fp16(seed, offset, count=4)
                    )
        result.append(values)
    return (
        torch.tensor(result, dtype=torch.float32, device=W.device)
        .reshape(m, n)
        .to(dtype)
    )


def dequantize_iq1_kt_reference(W, m, n, dtype=None):
    return _decode(W, m, n, dtype, IQ1_KT_PAYLOAD_BYTES, 1)


def dequantize_iq2_kt_reference(W, m, n, dtype=None):
    return _decode(W, m, n, dtype, IQ2_KT_PAYLOAD_BYTES, 2)


def dequantize_iq3_kt_reference(W, m, n, dtype=None):
    return _decode(W, m, n, dtype, IQ3_KT_PAYLOAD_BYTES, 3)


def dequantize_iq4_kt_reference(W, m, n, dtype=None):
    return _decode(W, m, n, dtype, IQ4_KT_PAYLOAD_BYTES, 4)


REFERENCE_DECODERS = {
    GGML_TYPE_IQ1_KT: dequantize_iq1_kt_reference,
    GGML_TYPE_IQ2_KT: dequantize_iq2_kt_reference,
    GGML_TYPE_IQ3_KT: dequantize_iq3_kt_reference,
    GGML_TYPE_IQ4_KT: dequantize_iq4_kt_reference,
}
