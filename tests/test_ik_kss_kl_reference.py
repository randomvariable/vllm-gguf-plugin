import struct

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from vllm_gguf_plugin.triton.dequantize.ik_kss_kl_reference import (
    REFERENCE_DECODERS,
    dequantize_iq2_kl_reference,
    dequantize_iq4_kss_reference,
)


def _row(prefix: bytes, block: bytes) -> torch.Tensor:
    return torch.tensor(list(prefix + block), dtype=torch.uint8)


def test_iq4_kss_reconstructs_xor_grey_and_codebook_selection():
    # Zero words exercise code zero, zero Grey reconstruction, and ls zero.
    block = bytes(128)
    raw = _row(struct.pack("<f", 1.0), block)

    out = dequantize_iq4_kss_reference(raw, 1, 256, torch.float32)

    assert out.shape == (1, 256)
    assert torch.equal(out[0, :16], torch.full((16,), 16129.0))
    assert torch.equal(out[0, 16:32], torch.full((16,), 16129.0))


def test_iq4_kss_packs_low_bits_by_position():
    # Bits at positions 0 and 2 produce ls=5, not the popcount value 2.
    block = bytearray(128)
    for word_index in (0, 2):
        struct.pack_into("<H", block, word_index * 2, 1)
    raw = _row(struct.pack("<f", 1.0), bytes(block))

    out = dequantize_iq4_kss_reference(raw, 1, 256, torch.float32)

    # ls=5 selects the second codebook half and scale -123.
    assert torch.equal(out[0, :16], torch.full((16,), 15621.0))


def test_iq2_kl_reconstructs_scale_packing_and_pair_codebook():
    # Scale bytes encode 1 and 2; q nibble zero with high bits zero selects 0xe9c1.
    block = struct.pack("<H", 0) + bytes([1, 2, 3, 4]) + bytes(64) + bytes(16)
    raw = _row(struct.pack("<e", 1.0), block)

    out = dequantize_iq2_kl_reference(raw, 1, 256, torch.float32)

    assert torch.equal(out[0, :32], torch.tensor([1953.0, 713.0] * 16))
    assert torch.equal(out[0, 32:64], torch.tensor([1890.0, 690.0] * 16))


@pytest.mark.parametrize(
    "decoder", [dequantize_iq4_kss_reference, dequantize_iq2_kl_reference]
)
def test_reference_rejects_bad_dtype_shape_storage(decoder):
    with pytest.raises(TypeError):
        decoder(torch.zeros(132, dtype=torch.int8), 1, 256)
    with pytest.raises(ValueError):
        decoder(torch.zeros(132, dtype=torch.uint8), 1, 255)
    with pytest.raises(ValueError):
        decoder(torch.zeros(1, dtype=torch.uint8), 1, 256)


def test_reference_requires_contiguous_storage_and_registers_types():
    raw = torch.zeros(264, dtype=torch.uint8)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        dequantize_iq4_kss_reference(raw, 1, 256)
    assert set(REFERENCE_DECODERS) == {146, 157}


@given(st.integers(min_value=1, max_value=3))
def test_iq4_kss_finite_for_multiple_rows(rows):
    raw = torch.zeros(rows * 132, dtype=torch.uint8)
    output = dequantize_iq4_kss_reference(raw, rows, 256, torch.float32)
    assert output.shape == (rows, 256)
    assert torch.isfinite(output).all()
