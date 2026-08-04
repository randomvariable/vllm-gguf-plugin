import pytest
import torch

from tests import ik_abi_oracle
from vllm_gguf_plugin.triton.dequantize.ik_k56_reference import (
    REFERENCE_DECODERS,
    dequantize_iq5_k_reference,
    dequantize_iq6_k_reference,
)


def _half(value: float) -> bytes:
    return torch.tensor([value], dtype=torch.float16).numpy().tobytes()


def test_iq5_k_decodes_codebook_and_high_bits():
    block = bytearray(176)
    block[:2] = _half(1.0)
    block[2:4] = (1 | 2 << 1 | 1 << 4 | 2 << 5).to_bytes(2, "little")
    block[4:8] = bytes([0x00, 0x00, 0x00, 0x00])
    block[8:16] = bytes([0x10] * 8)
    block[16] = 0x0F
    block[16 + 16] = 0xF0
    block[144] = 0x01
    block[144 + 16] = 0x02

    decoded = dequantize_iq5_k_reference(
        torch.tensor(block, dtype=torch.uint8), 1, 256, torch.float32
    )

    # Each 64-value group uses four sub-scales derived from one scales_h byte,
    # all biased by -32; y[j+32]/y[j+48] reuse the high nibbles of qs[j]/qs[j+16].
    assert decoded.shape == (1, 256)
    expected = torch.tensor(ik_abi_oracle.iq5_k(block), dtype=torch.float32)
    torch.testing.assert_close(decoded.reshape(-1), expected)


def test_iq6_k_decodes_extra_codebook_branch_and_little_endian_scale():
    block = bytearray(212)
    block[:2] = _half(0.5)
    block[2:4] = (1 | 2 << 1).to_bytes(2, "little")
    block[4:20] = bytes([2, 3] * 8)
    block[20] = 0x0F
    block[20 + 16] = 0xF0
    block[148] = 0x01
    block[148 + 16] = 0x02

    decoded = dequantize_iq6_k_reference(
        torch.tensor(block, dtype=torch.uint8), 1, 256, torch.float32
    )

    # block_iq6_k.scales is int8_t and is applied directly (no bias), with the
    # extra bits selecting the upper half of the 128-entry codebook.
    assert decoded.shape == (1, 256)
    expected = torch.tensor(ik_abi_oracle.iq6_k(block), dtype=torch.float32)
    torch.testing.assert_close(decoded.reshape(-1), expected)


@pytest.mark.parametrize(
    "decoder, block_bytes, scale",
    [
        (dequantize_iq5_k_reference, 176, 1.25),
        (dequantize_iq6_k_reference, 212, 0.75),
    ],
)
def test_iq_k56_reads_one_fp16_scale_per_block(decoder, block_bytes, scale):
    block = bytearray(block_bytes)
    block[:2] = _half(scale)
    if block_bytes == 176:
        block[16] = 0x01
    else:
        # IQ6_K multiplies by signed per-group scales; leaving them zero would
        # make the whole block decode to zero regardless of the packed codes.
        block[4:20] = bytes([1] * 16)
        block[20] = 0x01

    decoded = decoder(torch.tensor(block, dtype=torch.uint8), 1, 256, torch.float32)

    assert decoded.shape == (1, 256)
    assert torch.isfinite(decoded).all()
    assert decoded.abs().max().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "decoder, block_bytes",
    [(dequantize_iq5_k_reference, 176), (dequantize_iq6_k_reference, 212)],
)
def test_iq_k56_codebooks_follow_cuda_input(decoder, block_bytes):
    block = bytearray(block_bytes)
    block[:2] = _half(1.0)
    block[16 if block_bytes == 176 else 20] = 0x0F

    decoded = decoder(
        torch.tensor(block, dtype=torch.uint8, device="cuda"), 1, 256, torch.float32
    )

    assert decoded.device.type == "cuda"
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("quant_type", [140, 141])
def test_iq_k56_decoders_are_registered(quant_type):
    assert quant_type in REFERENCE_DECODERS


@pytest.mark.parametrize(
    "decoder, block_bytes",
    [(dequantize_iq5_k_reference, 176), (dequantize_iq6_k_reference, 212)],
)
def test_iq_k56_rejects_malformed_inputs(decoder, block_bytes):
    with pytest.raises(TypeError, match="torch.uint8"):
        decoder(torch.zeros(block_bytes, dtype=torch.int16), 1, 256)
    with pytest.raises(ValueError, match="divisible by 256"):
        decoder(torch.zeros(block_bytes, dtype=torch.uint8), 1, 255)
    with pytest.raises(ValueError, match="exactly"):
        decoder(torch.zeros(block_bytes - 1, dtype=torch.uint8), 1, 256)
    with pytest.raises(ValueError, match="positive"):
        decoder(torch.zeros(0, dtype=torch.uint8), 0, 256)


def test_iq_k56_accepts_multiple_blocks_and_dtype():
    raw = torch.zeros(2 * 176, dtype=torch.uint8)
    decoded = dequantize_iq5_k_reference(raw, 1, 512, torch.float64)
    assert decoded.dtype is torch.float64
    assert decoded.shape == (1, 512)


try:
    from hypothesis import given
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - optional local development dependency
    given = None
else:

    @given(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3))
    def test_iq_k56_finite_shapes(rows, blocks):
        n = blocks * 256
        for decoder, block_bytes in (
            (dequantize_iq5_k_reference, 176),
            (dequantize_iq6_k_reference, 212),
        ):
            output = decoder(
                torch.zeros(rows * blocks * block_bytes, dtype=torch.uint8), rows, n
            )
            assert output.shape == (rows, n)
            assert torch.isfinite(output).all()
