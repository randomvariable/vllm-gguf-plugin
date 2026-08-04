"""Independent ik_llama ABI oracles for reference-decoder tests.

These are hand transcriptions of the authoritative native CUDA kernels in
``vllm_gguf_plugin/csrc/gguf/dequantize.cuh`` (the path validated on real
gfx1151 and GB10 hardware). They deliberately import nothing from the
production Triton/PyTorch decoders: a shared helper would encode the same
assumption on both sides and could never expose a decoding error.

Each function takes one packed block (``bytes``/``bytearray``) and returns a
plain ``list[float]`` of decoded weights, mirroring the C loop structure line
for line rather than vectorising it.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

_CUDA_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "vllm_gguf_plugin"
    / "csrc"
    / "gguf"
    / "dequantize.cuh"
)


def _table(name: str) -> list[int]:
    """Read a codebook straight out of the native source."""
    src = _CUDA_SOURCE.read_text()
    match = re.search(rf"{name}\[\d+\]\s*=\s*\{{(.*?)\}};", src, re.S)
    if match is None:  # pragma: no cover - guards against source drift
        raise AssertionError(f"codebook {name} not found in {_CUDA_SOURCE}")
    return [int(value) for value in re.findall(r"-?\d+", match.group(1))]


KVALUES_IQ2NL = _table("kvalues_iq2nl")
KVALUES_IQ3NL = _table("kvalues_iq3nl")
KVALUES_IQ4K = _table("kvalues_iq4k")
KVALUES_IQ5NL = _table("kvalues_iq5nl")
KVALUES_IQ6NL = _table("kvalues_iq6nl")


def _half(block: bytes) -> float:
    return struct.unpack("<e", bytes(block[0:2]))[0]


def iq2_k(block: bytes) -> list[float]:
    """dequantize_block_iq2_k.

    Scales are ``nibble - 8``; qs advances only every 4th group.
    """
    d = _half(block)
    extra = block[2] | (block[3] << 8)
    scales, qs = block[4:12], block[12:76]
    out = [0.0] * 256
    for ib32 in range(8):
        shift = 2 * (ib32 % 4)
        base = (ib32 // 4) * 32
        dl1 = d * ((scales[ib32] & 0xF) - 8)
        dl2 = d * ((scales[ib32] >> 4) - 8)
        selector = extra >> (2 * ib32)
        v1 = 4 if selector & 1 else 0
        v2 = 4 if selector & 2 else 0
        for j in range(16):
            out[ib32 * 32 + j] = dl1 * KVALUES_IQ2NL[v1 + ((qs[base + j] >> shift) & 3)]
            out[ib32 * 32 + j + 16] = (
                dl2 * KVALUES_IQ2NL[v2 + ((qs[base + j + 16] >> shift) & 3)]
            )
    return out


def iq3_k(block: bytes) -> list[float]:
    """dequantize_block_iq3_k.

    Odd magnitudes ``2*n + 1`` with a sign bit taken from scales_h.
    """
    d = _half(block)
    extra = block[2] | (block[3] << 8)
    scales_h = block[4] | (block[5] << 8)
    scales_l, qs, qh = block[6:14], block[14:78], block[78:110]
    out = [0.0] * 256
    for ib32 in range(8):
        sign_bits = scales_h >> (2 * ib32)
        dl1 = d * ((2 * (scales_l[ib32] & 0xF) + 1) * (-1 if sign_bits & 1 else 1))
        dl2 = d * ((2 * (scales_l[ib32] >> 4) + 1) * (-1 if sign_bits & 2 else 1))
        selector = extra >> (2 * ib32)
        v1 = 8 if selector & 1 else 0
        v2 = 8 if selector & 2 else 0
        shift_l = 2 * (ib32 % 4)
        shift_h = ib32 % 8
        base = (ib32 // 4) * 32
        for j in range(16):
            code1 = ((qs[base + j] >> shift_l) & 3) | (((qh[j] >> shift_h) & 1) << 2)
            code2 = ((qs[base + j + 16] >> shift_l) & 3) | (
                ((qh[j + 16] >> shift_h) & 1) << 2
            )
            out[ib32 * 32 + j] = dl1 * KVALUES_IQ3NL[v1 + code1]
            out[ib32 * 32 + j + 16] = dl2 * KVALUES_IQ3NL[v2 + code2]
    return out


def iq4_k(block: bytes) -> list[float]:
    """dequantize_block_iq4_k.

    6-bit scale split across scales_h/scales_l, biased by -32.
    """
    d = _half(block)
    extra = block[2] | (block[3] << 8)
    scales_h, scales_l, qs = block[4:8], block[8:16], block[16:144]
    out = [0.0] * 256
    for ib32 in range(8):
        sh = scales_h[ib32 // 2] >> (4 * (ib32 % 2))
        dl1 = d * (((scales_l[ib32] & 0xF) | ((sh << 4) & 0x30)) - 32)
        dl2 = d * (((scales_l[ib32] >> 4) | ((sh << 2) & 0x30)) - 32)
        selector = extra >> (2 * ib32)
        v1 = 16 if selector & 1 else 0
        v2 = 16 if selector & 2 else 0
        base = ib32 * 16
        for j in range(16):
            out[ib32 * 32 + j] = dl1 * KVALUES_IQ4K[v1 + (qs[base + j] & 0xF)]
            out[ib32 * 32 + j + 16] = dl2 * KVALUES_IQ4K[v2 + (qs[base + j] >> 4)]
    return out


def iq5_k(block: bytes) -> list[float]:
    """dequantize_block_iq5_k: four sub-scales per 64 values from one scales_h byte."""
    d = _half(block)
    extra_full = block[2] | (block[3] << 8)
    scales_h, scales_l = block[4:8], block[8:16]
    qs, qh = block[16:144], block[144:176]
    out = [0.0] * 256
    for ib32 in range(8):
        ib64, hi = ib32 // 2, ib32 & 1
        shift = 2 * ib64
        sh = scales_h[ib64]
        sl = scales_l[2 * ib64 + hi]
        high1 = ((sh >> 0) & 0x30) if hi else ((sh << 4) & 0x30)
        high2 = ((sh >> 2) & 0x30) if hi else ((sh << 2) & 0x30)
        dl1 = d * (((sl & 0xF) | high1) - 32)
        dl2 = d * (((sl >> 4) | high2) - 32)
        selector = extra_full >> (4 * ib64 + 2 * hi)
        v1 = (selector & 1) << 5
        v2 = (selector & 2) << 4
        source = qs[32 * ib64 : 32 * ib64 + 32]
        for j in range(16):
            if hi:
                q1 = (source[j] >> 4) | (((qh[j] >> shift) & 2) << 3)
                q2 = (source[j + 16] >> 4) | (((qh[j + 16] >> shift) & 2) << 3)
            else:
                q1 = (source[j] & 0xF) | (((qh[j] >> shift) & 1) << 4)
                q2 = (source[j + 16] & 0xF) | (((qh[j + 16] >> shift) & 1) << 4)
            out[ib32 * 32 + j] = dl1 * KVALUES_IQ5NL[v1 + q1]
            out[ib32 * 32 + j + 16] = dl2 * KVALUES_IQ5NL[v2 + q2]
    return out


def iq6_k(block: bytes) -> list[float]:
    """dequantize_block_iq6_k: signed int8 scales and a 128-entry codebook."""
    d = _half(block)
    extra_full = block[2] | (block[3] << 8)
    scales = [value - 256 if value > 127 else value for value in block[4:20]]
    qs, qh = block[20:148], block[148:212]
    out = [0.0] * 256
    for ib32 in range(8):
        ib64, hi = ib32 // 2, ib32 & 1
        shift = 4 * (ib64 & 1)
        dl1 = d * scales[4 * ib64 + 2 * hi]
        dl2 = d * scales[4 * ib64 + 2 * hi + 1]
        selector = extra_full >> (4 * ib64 + 2 * hi)
        source = qs[32 * ib64 : 32 * ib64 + 32]
        high = qh[32 * (ib64 // 2) : 32 * (ib64 // 2) + 32]
        for j in range(16):
            if hi:
                q1 = (source[j] >> 4) | (((high[j] >> shift) & 0x0C) << 2)
                q2 = (source[j + 16] >> 4) | (((high[j + 16] >> shift) & 0x0C) << 2)
            else:
                q1 = (source[j] & 0xF) | (((high[j] >> shift) & 0x03) << 4)
                q2 = (source[j + 16] & 0xF) | (((high[j + 16] >> shift) & 0x03) << 4)
            out[ib32 * 32 + j] = dl1 * KVALUES_IQ6NL[q1 + (64 if selector & 1 else 0)]
            out[ib32 * 32 + j + 16] = (
                dl2 * KVALUES_IQ6NL[q2 + (64 if selector & 2 else 0)]
            )
    return out


def q6_0(block: bytes) -> list[float]:
    """dequantize_block_q6_0: low nibbles fill y[0:16], high nibbles y[16:32]."""
    d = _half(block)
    qh, qs = block[2:10], block[10:26]
    out = [0.0] * 32
    for j in range(16):
        h = qh[j % 8] >> (4 * (j // 8))
        out[j] = d * (((qs[j] & 0x0F) | ((h << 4) & 0x30)) - 32)
        out[j + 16] = d * (((qs[j] >> 4) | ((h << 2) & 0x30)) - 32)
    return out


def i2_s_row(row: bytes, n: int) -> list[float]:
    """dequantize_block_i2_s: MSB-first groups nested inside each 128-value block."""
    scale = struct.unpack("<f", bytes(row[n // 4 : n // 4 + 4]))[0]
    out = [0.0] * n
    for block_index in range(n // 128):
        for group in range(4):
            for j in range(32):
                packed = row[block_index * 32 + j]
                code = (packed >> (6 - 2 * group)) & 3
                out[block_index * 128 + group * 32 + j] = scale * (code - 1)
    return out
