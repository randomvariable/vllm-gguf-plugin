import numpy as np
import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton


def _block(payload: bytes, *scales: int) -> bytes:
    return payload + bytes(scales)


_TYPE_100_CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)


def _type_100_scale(scale_byte: int) -> float:
    """Independent UE4M3 bias-8 oracle for one type-100 scale byte."""
    if scale_byte > 0x7E:
        return 0.0
    exponent, mantissa = divmod(scale_byte, 8)
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _type_100_oracle(
    packed_codes: bytes, low_scale: int, high_scale: int
) -> torch.Tensor:
    """Decode type-100 ABI directly, without using plugin decode helpers."""
    assert len(packed_codes) == 16
    decoded = [
        _TYPE_100_CODEBOOK[byte & 0x0F] * _type_100_scale(low_scale)
        for byte in packed_codes
    ]
    decoded.extend(
        _TYPE_100_CODEBOOK[byte >> 4] * _type_100_scale(high_scale)
        for byte in packed_codes
    )
    return torch.tensor([decoded], dtype=torch.float32)


_NATIVE_TYPE_100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not ops._cuda_kernel_available("ggml_dequantize", 100),
    reason="requires native type-100 CUDA/HIP extension",
)


@_NATIVE_TYPE_100
@pytest.mark.parametrize("low_scale, high_scale", [(0x40, 0x48), (0x7F, 0xFF)])
def test_type100_native_decode_matches_independent_oracle(
    low_scale: int, high_scale: int
) -> None:
    packed_codes = bytes(
        (low | (high << 4)) for low, high in zip(range(16), range(15, -1, -1))
    )
    weight = torch.tensor(
        list(_block(packed_codes, low_scale, high_scale)),
        dtype=torch.uint8,
        device="cuda",
    )

    decoded = torch.ops._C_gguf.ggml_dequantize(weight, 100, 1, 32, torch.float32)

    torch.testing.assert_close(
        decoded, _type_100_oracle(packed_codes, low_scale, high_scale).cuda()
    )


@_NATIVE_TYPE_100
def test_type100_native_decode_rejects_unsupported_output_dtype() -> None:
    weight = torch.zeros(18, dtype=torch.uint8, device="cuda")

    with pytest.raises(RuntimeError, match="float32, float16, or bfloat16"):
        torch.ops._C_gguf.ggml_dequantize(weight, 100, 1, 32, torch.int32)


def test_type100_public_decode_rejects_unsupported_output_dtype() -> None:
    weight = torch.zeros(18, dtype=torch.uint8)

    with pytest.raises(TypeError, match="float32, float16, or bfloat16"):
        ops.ggml_dequantize(weight, 100, 1, 32, torch.int32)


def test_type100_public_decode_accepts_default_dtype_without_native_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = torch.zeros(18, dtype=torch.uint8)
    sentinel = object()

    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *_args: False)
    monkeypatch.setattr(
        ops,
        "ggml_dequantize_triton",
        lambda *args: sentinel,
    )

    assert ops.ggml_dequantize(weight, 100, 1, 32, None) is sentinel


@pytest.mark.parametrize(
    ("quant_type", "raw", "expected"),
    [
        (
            100,
            _block(bytes([0x21] * 16), 0x40, 0x48),
            np.array([1] * 16 + [4] * 16, dtype=np.float32),
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
    np.testing.assert_allclose(
        decoded, torch.tensor([[0, 1, -2, 0, 0, -1, -2, -4] * 4])
    )


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
    torch.testing.assert_close(decoded, torch.tensor([[-32.0, -32.0, -32.0, 0.0] * 8]))


def test_rocmfp4_reference_decodes_contiguous_nibble_halves():
    # Type-100 assigns one scale to each contiguous 16-value nibble half.
    raw = _block(bytes([0x21] * 16), 0x40, 0x48)
    decoded = ggml_dequantize_triton(
        torch.frombuffer(raw, dtype=torch.uint8),
        100,
        1,
        32,
        dtype=torch.float32,
    )

    expected = torch.tensor([[1.0] * 16 + [4.0] * 16])
    torch.testing.assert_close(decoded, expected)


@pytest.mark.parametrize(
    ("low_scale", "high_scale"),
    [(0x7F, 0x48), (0x40, 0x7F), (0x80, 0x48), (0x40, 0xFE)],
)
def test_type100_reference_reserved_scales_are_half_local(
    low_scale: int, high_scale: int
):
    raw = _block(bytes([0x21] * 16), low_scale, high_scale)
    decoded = ggml_dequantize_triton(
        torch.frombuffer(raw, dtype=torch.uint8),
        100,
        1,
        32,
        dtype=torch.float32,
    )
    expected = _type_100_oracle(bytes([0x21] * 16), low_scale, high_scale)
    np.testing.assert_allclose(decoded.numpy(), expected.numpy())


@pytest.mark.parametrize("quant_type", [100, 102, 103, 104, 107])
def test_rocmfpx_reference_rejects_malformed_geometry_and_storage(quant_type):
    with pytest.raises(ValueError, match="divisible by 32"):
        ggml_dequantize_triton(torch.zeros(1, dtype=torch.uint8), quant_type, 1, 31)
    with pytest.raises(ValueError, match="requires"):
        ggml_dequantize_triton(torch.zeros(1, dtype=torch.uint8), quant_type, 1, 32)
