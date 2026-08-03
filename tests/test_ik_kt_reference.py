"""Pinned scalar-reference ABI tests for ik_llama.cpp KT trellis formats.

Expected values are reconstructed locally from the scalar decoder contract. Do
not import production trellis helpers here: bit packing and output placement are
part of the ABI these tests protect.
"""

from __future__ import annotations

import struct

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from vllm_gguf_plugin.ik_types import (
    GGML_TYPE_IQ1_KT,
    GGML_TYPE_IQ2_KT,
    GGML_TYPE_IQ3_KT,
    GGML_TYPE_IQ4_KT,
)
from vllm_gguf_plugin.triton.dequantize import interface
from vllm_gguf_plugin.triton.dequantize.ik_kt_reference import (
    IQ1_KT_PAYLOAD_BYTES,
    IQ2_KT_PAYLOAD_BYTES,
    IQ3_KT_PAYLOAD_BYTES,
    IQ4_KT_PAYLOAD_BYTES,
    REFERENCE_DECODERS,
    dequantize_iq1_kt_reference,
    dequantize_iq2_kt_reference,
    dequantize_iq3_kt_reference,
    dequantize_iq4_kt_reference,
)

QK = 256
IQ123_ROW_PREFIX_BYTES = 4
IQ4_ROW_PREFIX_BYTES = 4
FORMATS = (
    (
        GGML_TYPE_IQ1_KT,
        IQ123_ROW_PREFIX_BYTES,
        IQ1_KT_PAYLOAD_BYTES,
        dequantize_iq1_kt_reference,
    ),
    (
        GGML_TYPE_IQ2_KT,
        IQ123_ROW_PREFIX_BYTES,
        IQ2_KT_PAYLOAD_BYTES,
        dequantize_iq2_kt_reference,
    ),
    (
        GGML_TYPE_IQ3_KT,
        IQ123_ROW_PREFIX_BYTES,
        IQ3_KT_PAYLOAD_BYTES,
        dequantize_iq3_kt_reference,
    ),
    (
        GGML_TYPE_IQ4_KT,
        IQ4_ROW_PREFIX_BYTES,
        IQ4_KT_PAYLOAD_BYTES,
        dequantize_iq4_kt_reference,
    ),
)


def _trellis_fp16(seed: int, count: int = 8) -> list[float]:
    """Scalar ``kt_set_values_fp16`` with its ABI offset applied first."""
    return _trellis_fp16_with_offset(seed, 4096, count)


def _trellis_fp16_with_offset(seed: int, offset: int, count: int = 8) -> list[float]:
    values = []
    seed += offset
    for _ in range(count):
        seed = (89226354 * seed + 64248484) & 0xFFFFFFFF
        packed = ((seed & 0x8FFF8FFF) ^ 0x3B603B60).to_bytes(4, "little")
        values.append(
            abs(struct.unpack("<e", packed[:2])[0] + struct.unpack("<e", packed[2:])[0])
        )
    return values


def _trellis_int(seed: int) -> list[float]:
    values = []
    seed += 4096
    for _ in range(8):
        seed = (seed * 0xCBAC1FED) & 0xFFFFFFFF
        values.append(
            abs(float(-126 + sum((seed >> shift) & 0x3F for shift in range(0, 32, 8))))
        )
    return values


def _row(*prefix: float, payloads: list[bytes]) -> torch.Tensor:
    raw = struct.pack("<" + "f" * len(prefix), *prefix) + b"".join(payloads)
    return torch.tensor(list(raw), dtype=torch.uint8)


def _iq1_payload() -> bytes:
    payload = bytearray(IQ1_KT_PAYLOAD_BYTES)
    payload[:8] = bytes([0x11, 0, 0, 0, 0, 0, 0, 0])
    payload[8:40] = bytes(range(1, 33))
    payload[40] = 0x0A
    # sh[0] low nibble selects -104; bit 4 contributes seed bit 12.
    payload[48] = 0x11
    return bytes(payload)


def _iq2_payload() -> bytes:
    payload = bytearray(IQ2_KT_PAYLOAD_BYTES)
    payload[:4] = bytes([0x21, 0, 0, 0])
    for index in range(32):
        payload[4 + 2 * index : 6 + 2 * index] = (index + 1).to_bytes(2, "little")
    return bytes(payload)


def _iq3_payload() -> bytes:
    payload = bytearray(IQ3_KT_PAYLOAD_BYTES)
    payload[:4] = bytes([0x21, 0, 0, 0])
    for index in range(32):
        payload[4 + 2 * index : 6 + 2 * index] = (index + 1).to_bytes(2, "little")
    # qh[0:8] bit 0 signs the first low group; bit 4 signs first high group.
    payload[68:76] = bytes([0x11] * 8)
    return bytes(payload)


def _iq4_payload() -> bytes:
    payload = bytearray(IQ4_KT_PAYLOAD_BYTES)
    headers = [0x05, 0x7A, 0x83, 0xFC, 0x01, 0x42, 0xBD, 0x80]
    for index, header in enumerate(headers):
        payload[4 * index : 4 * index + 4] = (
            header | (index << 8) | ((7 - index) << 20)
        ).to_bytes(4, "little")
    payload[32:96] = bytes(range(1, 65))
    payload[96:128] = bytes([0x01, 0x32, 0x54, 0x76] * 8)
    return bytes(payload)


def test_iq1_kt_qh_and_sh_complete_the_seed_and_use_codebook_scale():
    output = dequantize_iq1_kt_reference(
        _row(2.0, payloads=[_iq1_payload()]), 1, QK, torch.float32
    )

    # ql=1, qh=0xA supplies bits 8..11, sh[0] bit 4 supplies bit 12.
    seed = 0x1A01
    expected = (
        torch.tensor(_trellis_fp16(seed), dtype=torch.float32)
        * 2.0
        * 31.75
        * -104
    )
    torch.testing.assert_close(output[0, :8], expected)


def test_iq2_kt_uses_low_and_high_scale_nibbles_at_scalar_output_offsets():
    output = dequantize_iq2_kt_reference(
        _row(1.0, payloads=[_iq2_payload()]), 1, QK, torch.float32
    )

    first = (
        torch.tensor(_trellis_fp16(1), dtype=torch.float32)
        * 31.75
        * 1.0
        * -104
    )
    high_half = (
        torch.tensor(_trellis_fp16(17), dtype=torch.float32)
        * 31.75
        * -83
    )
    torch.testing.assert_close(output[0, :8], first)
    torch.testing.assert_close(output[0, 128:136], high_half)


def test_iq3_kt_applies_per_value_qh_signs_to_integer_trellis_magnitudes():
    output = dequantize_iq3_kt_reference(
        _row(1.0, payloads=[_iq3_payload()]), 1, QK, torch.float32
    )

    low = -torch.tensor(_trellis_int(1), dtype=torch.float32)
    high = -torch.tensor(_trellis_int(17), dtype=torch.float32) * 2
    torch.testing.assert_close(output[0, :8], low)
    torch.testing.assert_close(output[0, 128:136], high)


def test_iq4_kt_reads_header_then_assembles_interleaved_seed_pairs():
    output = dequantize_iq4_kt_reference(
        _row(0.5, payloads=[_iq4_payload()]), 1, QK, torch.float32
    )

    # Header 0 has scale ((0x05 >> 1) - 64) == -62 and offset bit set.
    # ql[0]=1, qh[0] bit 0 supplies seed bits 8..11.
    seed0 = 0x101
    seed1 = 0x202
    expected = torch.tensor(
        _trellis_fp16_with_offset(seed0, 32768 + 4096, 4)
        + _trellis_fp16_with_offset(seed1, 32768 + 4096, 4)
    )
    expected = expected * (0.5 * 31.75 * -62)
    torch.testing.assert_close(output[0, :8], expected)


@given(
    m=st.integers(min_value=1, max_value=3),
    blocks=st.integers(min_value=1, max_value=3),
    format_data=st.sampled_from(FORMATS),
)
def test_kt_row_prefix_is_once_per_row_for_multi_block_rows(m, blocks, format_data):
    _, prefix_bytes, payload_bytes, decoder = format_data
    prefix = (1.0,) if prefix_bytes == 4 else (1.0, 0.0)
    raw = _row(*prefix, payloads=[bytes(payload_bytes)] * blocks).repeat(m)

    output = decoder(raw, m, blocks * QK, dtype=torch.float32)
    assert output.shape == (m, blocks * QK)


@pytest.mark.parametrize("_, prefix_bytes, payload_bytes, decoder", FORMATS)
def test_kt_rejects_malformed_dtype_shape_and_row_byte_count(
    _, prefix_bytes, payload_bytes, decoder
):
    valid = torch.zeros(prefix_bytes + payload_bytes, dtype=torch.uint8)
    with pytest.raises(TypeError, match="uint8"):
        decoder(valid.to(torch.int16), 1, QK, dtype=torch.float32)
    with pytest.raises(ValueError, match="Invalid|positive|shape"):
        decoder(valid, 0, QK, dtype=torch.float32)
    with pytest.raises(ValueError, match="Invalid|divisible|shape"):
        decoder(valid, 1, QK - 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly|requires"):
        decoder(valid[:-1], 1, QK, dtype=torch.float32)


def test_kt_public_dispatch_registers_all_scalar_decoders():
    expected = {
        GGML_TYPE_IQ1_KT: dequantize_iq1_kt_reference,
        GGML_TYPE_IQ2_KT: dequantize_iq2_kt_reference,
        GGML_TYPE_IQ3_KT: dequantize_iq3_kt_reference,
        GGML_TYPE_IQ4_KT: dequantize_iq4_kt_reference,
    }
    assert expected == REFERENCE_DECODERS
    assert {key: interface.REFERENCE_DECODERS[key] for key in expected} == expected

    for quant_type, prefix_bytes, payload_bytes, decoder in FORMATS:
        prefix = (0.0,) if prefix_bytes == 4 else (0.0, 0.0)
        raw = _row(*prefix, payloads=[bytes(payload_bytes)])
        expected_output = decoder(raw, 1, QK, dtype=torch.float32)
        actual = interface.ggml_dequantize_triton(raw, quant_type, 1, QK, torch.float32)
        torch.testing.assert_close(actual, expected_output)
