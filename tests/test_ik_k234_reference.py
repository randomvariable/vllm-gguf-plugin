from __future__ import annotations

import struct

import pytest
import torch

from tests import ik_abi_oracle
from vllm_gguf_plugin.triton.dequantize.ik_k234_reference import (
    IQ2_K_BLOCK_BYTES,
    IQ3_K_BLOCK_BYTES,
    IQ4_K_BLOCK_BYTES,
    REFERENCE_DECODERS,
    dequantize_iq2_k_reference,
    dequantize_iq3_k_reference,
    dequantize_iq4_k_reference,
)


@pytest.mark.parametrize(
    ("decoder", "block_bytes"),
    [
        (dequantize_iq2_k_reference, IQ2_K_BLOCK_BYTES),
        (dequantize_iq3_k_reference, IQ3_K_BLOCK_BYTES),
        (dequantize_iq4_k_reference, IQ4_K_BLOCK_BYTES),
    ],
)
def test_zero_blocks_decode_to_zero(decoder, block_bytes):
    raw = torch.zeros(block_bytes, dtype=torch.uint8)
    torch.testing.assert_close(
        decoder(raw, 1, 256, dtype=torch.float32), torch.zeros(1, 256)
    )


def test_iq2_k_decodes_dual_codebooks_and_group_order():
    block = bytearray(IQ2_K_BLOCK_BYTES)
    block[:2] = struct.pack("<e", 1.0)
    block[2:4] = (0b01 | (0b10 << 2)).to_bytes(2, "little")
    block[4:12] = bytes([0x18] * 8)  # low scale 0, high scale 1
    block[12:76] = bytes([0xE4] * 64)  # codes 0, 1, 2, 3 by plane

    output = dequantize_iq2_k_reference(
        torch.tensor(list(block), dtype=torch.uint8), 1, 256, torch.float32
    )
    # Scales are `nibble - 8` (not -32), and qs advances only every 4th group,
    # so groups 0..3 share one 32-byte window at shifts 0,2,4,6.
    expected = torch.tensor(ik_abi_oracle.iq2_k(block), dtype=torch.float32)
    torch.testing.assert_close(output.reshape(-1), expected)


def test_iq2_k_decodes_per_block_fp16_scale_and_extra_codebooks():
    blocks = []
    for scale, extra in ((1.0, 0b01), (2.0, 0b10)):
        block = bytearray(IQ2_K_BLOCK_BYTES)
        block[:2] = struct.pack("<e", scale)
        block[2:4] = extra.to_bytes(2, "little")
        block[4:12] = bytes([0x18] * 8)
        block[12:76] = bytes([0xE4] * 64)
        blocks.append(block)

    raw = torch.tensor([byte for block in blocks for byte in block], dtype=torch.uint8)
    output = dequantize_iq2_k_reference(raw, 2, 256, torch.float32)
    first = torch.tensor(ik_abi_oracle.iq2_k(blocks[0]), dtype=torch.float32)
    second = torch.tensor(ik_abi_oracle.iq2_k(blocks[1]), dtype=torch.float32)
    torch.testing.assert_close(output[0], first)
    torch.testing.assert_close(output[1], second)


def test_iq3_k_decodes_high_bits_signed_scales_and_codebooks():
    block = bytearray(IQ3_K_BLOCK_BYTES)
    block[:2] = struct.pack("<e", 1.0)
    block[2:4] = (0b01 | (0b10 << 2)).to_bytes(2, "little")
    block[4:6] = (0b01).to_bytes(2, "little")  # negative first scale in group 0
    block[6:14] = bytes([0x10] * 8)  # scales 0 and 1
    block[14:78] = bytes([0xE4] * 64)
    block[78:110] = bytes([0x01] * 32)  # high bit set for every first-half code

    output = dequantize_iq3_k_reference(
        torch.tensor(list(block), dtype=torch.uint8), 1, 256, torch.float32
    )
    # Scale magnitude is `2*nibble + 1` with the sign taken from scales_h, and
    # the third code bit comes from qh at shift ib32 % 8.
    expected = torch.tensor(ik_abi_oracle.iq3_k(block), dtype=torch.float32)
    torch.testing.assert_close(output.reshape(-1), expected)


def test_iq3_k_decodes_per_block_fp16_scale_and_extra_codebooks():
    blocks = []
    for scale, extra in ((1.0, 0b01), (2.0, 0b10)):
        block = bytearray(IQ3_K_BLOCK_BYTES)
        block[:2] = struct.pack("<e", scale)
        block[2:4] = extra.to_bytes(2, "little")
        block[6:14] = bytes([0x10] * 8)
        block[14:78] = bytes([0xE4] * 64)
        blocks.append(block)

    raw = torch.tensor([byte for block in blocks for byte in block], dtype=torch.uint8)
    output = dequantize_iq3_k_reference(raw, 2, 256, torch.float32)
    assert output[0, 0].item() == pytest.approx(-59.0)
    assert output[1, 0].item() == pytest.approx(-126.0)
    assert output[1, 128].item() == pytest.approx(-126.0)


def test_iq4_k_decodes_scale_split_and_nibble_order():
    block = bytearray(IQ4_K_BLOCK_BYTES)
    block[:2] = struct.pack("<e", 1.0)
    block[2:4] = (0b01).to_bytes(2, "little")
    block[4:8] = bytes([0x00] * 4)
    block[8:16] = bytes([0x21] * 8)  # signed scales -32 and -31
    block[16:144] = bytes([0x21] * 128)

    output = dequantize_iq4_k_reference(
        torch.tensor(list(block), dtype=torch.uint8), 1, 256, torch.float32
    )
    # The 6-bit scale is 4 low bits from scales_l plus 2 high bits from
    # scales_h, biased by -32; both halves read the same qs byte.
    expected = torch.tensor(ik_abi_oracle.iq4_k(block), dtype=torch.float32)
    torch.testing.assert_close(output.reshape(-1), expected)


def test_iq4_k_decodes_per_block_fp16_scale_and_extra_codebooks():
    blocks = []
    for scale, extra in ((1.0, 0b01), (2.0, 0b10)):
        block = bytearray(IQ4_K_BLOCK_BYTES)
        block[:2] = struct.pack("<e", scale)
        block[2:4] = extra.to_bytes(2, "little")
        block[8:16] = bytes([0x00] * 8)
        block[16:144] = bytes([0x00] * 128)
        blocks.append(block)

    raw = torch.tensor([byte for block in blocks for byte in block], dtype=torch.uint8)
    output = dequantize_iq4_k_reference(raw, 2, 256, torch.float32)
    torch.testing.assert_close(output[0, :16], torch.full((16,), 3936.0))
    torch.testing.assert_close(output[1, :16], torch.full((16,), 8128.0))
    torch.testing.assert_close(output[1, 128:144], torch.full((16,), 8128.0))


@pytest.mark.parametrize(
    "decoder",
    [
        dequantize_iq2_k_reference,
        dequantize_iq3_k_reference,
        dequantize_iq4_k_reference,
    ],
)
def test_rejects_malformed_input(decoder):
    with pytest.raises(TypeError):
        decoder(torch.zeros(76, dtype=torch.int16), 1, 256)
    with pytest.raises(ValueError):
        decoder(torch.zeros(76, dtype=torch.uint8), 1, 255)
    with pytest.raises(ValueError):
        decoder(torch.zeros(1, dtype=torch.uint8), 1, 256)
    with pytest.raises(ValueError):
        decoder(torch.zeros(2, 38, dtype=torch.uint8).t(), 1, 256)


def test_reference_mapping_uses_authoritative_type_ids():
    assert set(REFERENCE_DECODERS) == {137, 138, 139}
