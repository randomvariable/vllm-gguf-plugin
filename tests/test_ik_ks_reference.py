import struct

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from vllm_gguf_plugin.ik_types import (
    GGML_TYPE_IQ2_KS,
    GGML_TYPE_IQ3_KS,
    GGML_TYPE_IQ4_KS,
    GGML_TYPE_IQ5_KS,
    QK_IQ2_KS,
    QK_IQ3_KS,
    QK_IQ4_KS,
    QK_IQ5_KS,
)
from vllm_gguf_plugin.triton.dequantize.ik_ks_reference import (
    REFERENCE_DECODERS,
    dequantize_iq2_ks_reference,
    dequantize_iq3_ks_reference,
    dequantize_iq4_ks_reference,
    dequantize_iq5_ks_reference,
)

FORMATS = (
    (GGML_TYPE_IQ2_KS, QK_IQ2_KS, 2, 70, dequantize_iq2_ks_reference),
    (GGML_TYPE_IQ3_KS, QK_IQ3_KS, 2, 102, dequantize_iq3_ks_reference),
    (GGML_TYPE_IQ4_KS, QK_IQ4_KS, 4, 136, dequantize_iq4_ks_reference),
    (GGML_TYPE_IQ5_KS, QK_IQ5_KS, 4, 168, dequantize_iq5_ks_reference),
)


def test_reference_decoder_registry_contains_authoritative_types():
    assert set(REFERENCE_DECODERS) == {
        GGML_TYPE_IQ2_KS,
        GGML_TYPE_IQ3_KS,
        GGML_TYPE_IQ4_KS,
        GGML_TYPE_IQ5_KS,
    }


@pytest.mark.parametrize("_, qk, prefix, block, decoder", FORMATS)
def test_row_prefix_and_storage_geometry(_, qk, prefix, block, decoder):
    raw = torch.zeros(prefix + block, dtype=torch.uint8)
    output = decoder(raw, 1, qk, dtype=torch.float32)
    assert output.shape == (1, qk)
    with pytest.raises(ValueError, match="exactly"):
        decoder(torch.zeros(raw.numel() - 1, dtype=torch.uint8), 1, qk)


def test_iq2_ks_decodes_little_endian_prefix_scales_extra_and_codes():
    raw = struct.pack("<e", 2.0) + struct.pack("<H", 0x0001)
    raw += bytes([0x21, 0x43, 0x65, 0x87]) + bytes([0xE4] * 64)
    output = dequantize_iq2_ks_reference(
        torch.tensor(list(raw), dtype=torch.uint8), 1, 256, dtype=torch.float32
    )
    assert output[0, :32].unique().tolist() == [780.0]
    assert output[0, 32:64].unique().tolist() == [364.0]


def test_iq4_ks_decodes_codebook_and_row_scale():
    raw = struct.pack("<f", 2.0) + bytes([0x00] * 8) + bytes([0x21] * 128)
    output = dequantize_iq4_ks_reference(
        torch.tensor(list(raw), dtype=torch.uint8), 1, 256, dtype=torch.float32
    )
    torch.testing.assert_close(output[0, :32], torch.full((32,), 26416.0))


@pytest.mark.parametrize("_, qk, prefix, block, decoder", FORMATS)
def test_rejects_wrong_dtype_shape_and_noncontiguous_storage(
    _, qk, prefix, block, decoder
):
    row_bytes = prefix + block
    with pytest.raises(TypeError, match="uint8"):
        decoder(torch.zeros(row_bytes, dtype=torch.float32), 1, qk)
    with pytest.raises(ValueError, match="positive"):
        decoder(torch.zeros(row_bytes, dtype=torch.uint8), 0, qk)
    with pytest.raises(ValueError, match="divisible"):
        decoder(torch.zeros(row_bytes, dtype=torch.uint8), 1, qk - 1)
    with pytest.raises(ValueError, match="contiguous"):
        decoder(torch.zeros(row_bytes * 2, dtype=torch.uint8)[::2], 1, qk)


@given(
    m=st.integers(min_value=1, max_value=3),
    blocks_per_row=st.integers(min_value=1, max_value=3),
    format_data=st.sampled_from(FORMATS),
)
def test_reference_shapes_are_finite_for_valid_rows(m, blocks_per_row, format_data):
    _, qk, prefix, block, decoder = format_data
    raw = torch.zeros(m * (prefix + block * blocks_per_row), dtype=torch.uint8)
    output = decoder(raw, m, qk * blocks_per_row, dtype=torch.float32)
    assert output.shape == (m, qk * blocks_per_row)
    assert torch.isfinite(output).all()
