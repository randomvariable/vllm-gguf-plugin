import numpy as np
import pytest
import torch

from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton


def _block(payload: bytes, *scales: int) -> bytes:
    return payload + bytes(scales)


@pytest.mark.parametrize(
    ("quant_type", "raw", "expected"),
    [
        (
            100,
            _block(bytes([0x12] * 16), 0x40, 0x48),
            np.array(
                [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                dtype=np.float32,
            ),
        ),
        (
            107,
            _block(bytes([0xE4] * 8), 0x40, 0x48),
            np.array([-4, -1, 1, 4] * 4 + [-8, -2, 2, 8] * 4, dtype=np.float32),
        ),
        (
            103,
            _block(bytes([1, 255] * 16), 0x40),
            np.array([1, -1] * 16, dtype=np.float32),
        ),
    ],
)
def test_rocmfpx_reference_decodes_registered_layouts(quant_type, raw, expected):
    decoded = ggml_dequantize_triton(
        torch.frombuffer(raw, dtype=torch.uint8),
        quant_type,
        1,
        32,
        dtype=torch.float32,
    )
    np.testing.assert_allclose(decoded.numpy().reshape(-1), expected)


def test_rocmfpx_reference_decodes_q3_packed_codes():
    # Four groups of eight 3-bit values, packed exactly as dequantize.cuh.
    raw = _block(bytes([0x88, 0x89, 0xFA] * 4), 0x40, 0x40)
    decoded = ggml_dequantize_triton(
        torch.frombuffer(raw, dtype=torch.uint8),
        104,
        1,
        32,
        dtype=torch.float32,
    )
    np.testing.assert_allclose(decoded, torch.tensor([[0, 1, 2, 4, 0, -1, -2, -4] * 4]))


def test_rocmfpx_reference_decodes_q6_packed_codes():
    # Every packed code is 0x20: the signed-magnitude -32 edge case.
    raw = _block(bytes([0x20, 0x08, 0x02] * 8), 0x40, 0x40)
    decoded = ggml_dequantize_triton(
        torch.frombuffer(raw, dtype=torch.uint8),
        102,
        1,
        32,
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, torch.full((1, 32), -32.0))


@pytest.mark.parametrize("quant_type", [100, 102, 103, 104, 107])
def test_rocmfpx_reference_rejects_malformed_geometry_and_storage(quant_type):
    with pytest.raises(ValueError, match="divisible by 32"):
        ggml_dequantize_triton(torch.zeros(1, dtype=torch.uint8), quant_type, 1, 31)
    with pytest.raises(ValueError, match="requires"):
        ggml_dequantize_triton(torch.zeros(1, dtype=torch.uint8), quant_type, 1, 32)
