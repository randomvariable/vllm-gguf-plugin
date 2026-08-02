import struct

import pytest
import torch

from vllm_gguf_plugin.ik_types import GGML_TYPE_I2_S, GGML_TYPE_Q6_0
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton


def test_q6_0_reference_decodes_packed_high_bits_on_cpu():
    block = struct.pack("<e", 0.5) + bytes([0x01, 0x02, 0x04, 0x08] * 2)
    block += bytes(range(16))

    output = ggml_dequantize_triton(
        torch.tensor(list(block), dtype=torch.uint8),
        GGML_TYPE_Q6_0,
        1,
        32,
        dtype=torch.float32,
    )

    expected = []
    qs = block[10:]
    qh = block[2:10]
    for i, q in enumerate(qs):
        high = qh[i % 8] >> (4 * (i // 8))
        expected.append(0.5 * (((q & 0x0F) | ((high << 4) & 0x30)) - 32))
        expected.append(0.5 * (((q >> 4) | ((high << 2) & 0x30)) - 32))

    torch.testing.assert_close(output.reshape(-1), torch.tensor(expected))


def test_i2_s_reference_decodes_row_scale_and_four_planes():
    # One row: 32 packed bytes for 128 values, followed by float32 scale.
    packed = bytes([0b11100100] * 32)
    raw = packed + struct.pack("<f", 2.0)

    output = ggml_dequantize_triton(
        torch.tensor(list(raw), dtype=torch.uint8),
        GGML_TYPE_I2_S,
        1,
        128,
        dtype=torch.float32,
    )

    expected = torch.tensor([4.0] * 32 + [2.0] * 32 + [-2.0] * 64)
    torch.testing.assert_close(output.reshape(-1), expected)


@pytest.mark.parametrize("quant_type", [GGML_TYPE_I2_S, GGML_TYPE_Q6_0])
def test_ik_reference_rejects_wrong_shape_or_byte_count(quant_type):
    with pytest.raises(ValueError):
        ggml_dequantize_triton(torch.zeros(1, dtype=torch.uint8), quant_type, 1, 31)
