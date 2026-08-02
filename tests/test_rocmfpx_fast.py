import gguf
import numpy as np
import pytest
import torch

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton

ROCMFP4_FAST = 101
ROCMFP4_FAST_BLOCK = 17
ROCMFP4_FAST_QK = 32
ROCMFP4_CODEBOOK = np.array(
    [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10],
    dtype=np.float32,
)


def _fast_block(scale: int, qs: bytes) -> bytes:
    assert len(qs) == 16
    return qs + bytes([scale])


def test_rocmfp4_fast_enum_and_geometry_registered():
    assert gguf.GGMLQuantizationType.Q4_0_ROCMFP4_FAST.value == ROCMFP4_FAST
    assert gguf.GGML_QUANT_SIZES[ROCMFP4_FAST] == (
        ROCMFP4_FAST_QK,
        ROCMFP4_FAST_BLOCK,
    )


def test_rocmfp4_fast_is_dequant_only():
    assert ROCMFP4_FAST in ops._CUDA_DEQUANT_ONLY_TYPES
    assert ROCMFP4_FAST not in ops._CUDA_GEMV_QUANT_TYPES
    assert ROCMFP4_FAST not in ops._CUDA_GEMM_QUANT_TYPES


def test_rocmfp4_fast_reference_decoder_uses_low_then_high_nibbles():
    qs = bytes((j | ((15 - j) << 4)) for j in range(16))
    raw = _fast_block(0x40, qs)

    decoded = (
        ggml_dequantize_triton(
            torch.frombuffer(raw, dtype=torch.uint8), ROCMFP4_FAST, 1, 32,
            dtype=torch.float32,
        )
        .numpy()
    )
    expected = np.empty(ROCMFP4_FAST_QK, dtype=np.float32)
    expected[:16] = ROCMFP4_CODEBOOK
    expected[16:] = ROCMFP4_CODEBOOK[::-1]

    np.testing.assert_allclose(decoded.reshape(-1), expected)


def test_rocmfp4_fast_dequantizes_multiple_raw_blocks():
    first = _fast_block(0x40, bytes([0x11] * 16))
    second = _fast_block(0x48, bytes([0x2E] * 16))

    decoded = (
        ggml_dequantize_triton(
            torch.frombuffer(first + second, dtype=torch.uint8),
            ROCMFP4_FAST,
            1,
            64,
            dtype=torch.float32,
        )
        .numpy()
    )

    expected = np.concatenate(
        (
            np.full(16, 1.0, dtype=np.float32),
            np.full(16, 1.0, dtype=np.float32),
            np.full(16, -16.0, dtype=np.float32),
            np.full(16, 4.0, dtype=np.float32),
        )
    )
    np.testing.assert_allclose(decoded.reshape(-1), expected)


def test_rocmfp4_fast_cpu_dequantize_bypasses_native_extension(monkeypatch):
    expected = torch.full((1, 32), 3.0)

    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *args: True)
    monkeypatch.setattr(ops, "ggml_dequantize_triton", lambda *args: expected)
    output = ops.ggml_dequantize(
        torch.zeros(17, dtype=torch.uint8), ROCMFP4_FAST, 1, 32, torch.float32
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("shape", [(1, 31), (1, 33), (2, 16)])
def test_rocmfp4_fast_rejects_non_block_aligned_shape(shape):
    with pytest.raises(ValueError, match="divisible by 32"):
        ggml_dequantize_triton(
            torch.zeros(17, dtype=torch.uint8), ROCMFP4_FAST, *shape
        )


@pytest.mark.parametrize("shape", [(0, 32), (-1, 32), (1, 0), (1, -32)])
def test_rocmfp4_fast_rejects_non_positive_dimensions(shape):
    with pytest.raises(ValueError, match="positive dimensions"):
        ggml_dequantize_triton(
            torch.zeros(17, dtype=torch.uint8), ROCMFP4_FAST, *shape
        )


@pytest.mark.parametrize("byte_count", [16, 18])
def test_rocmfp4_fast_requires_exact_block_storage(byte_count):
    with pytest.raises(ValueError, match="requires 17 bytes"):
        ggml_dequantize_triton(
            torch.zeros(byte_count, dtype=torch.uint8), ROCMFP4_FAST, 1, 32
        )


def test_rocmfp4_fast_rejects_non_uint8_input():
    with pytest.raises(TypeError, match="torch.uint8"):
        ggml_dequantize_triton(
            torch.zeros(17, dtype=torch.int8), ROCMFP4_FAST, 1, 32
        )


@pytest.mark.parametrize("scale", [0x7F, 0x80, 0xFF])
def test_rocmfp4_fast_invalid_scale_bytes_decode_as_zero(scale):
    decoded = ggml_dequantize_triton(
        torch.frombuffer(_fast_block(scale, bytes([0x11] * 16)), dtype=torch.uint8),
        ROCMFP4_FAST,
        1,
        32,
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, torch.zeros((1, 32)))
