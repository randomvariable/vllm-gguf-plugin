"""RED contracts for public Q8_0_ROCMFPX dense fallback."""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin import ops, rocmfpx_types
from vllm_gguf_plugin.quantization import linear
from vllm_gguf_plugin.rocmfpx_types import (
    GGML_TYPE_Q8_0_ROCMFPX,
    Q8_0_ROCMFPX_BLOCK_BYTES,
    Q8_0_ROCMFPX_QK,
)


def _ue4m3_scale(byte: int) -> float:
    """Independent UE4M3 bias-8 reference for one ROCmFPX scale byte."""
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if byte > 0x7E:
        return 0.0
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 8))


def _signed_int8(value: int) -> int:
    return value if value < 128 else value - 256


def _q8_block(values: list[int], scale: int) -> torch.Tensor:
    assert len(values) == Q8_0_ROCMFPX_QK
    return torch.tensor(values + [scale], dtype=torch.uint8)


def _reference_weight(qweight: torch.Tensor) -> torch.Tensor:
    rows = []
    for packed_row in qweight.tolist():
        decoded = []
        for offset in range(0, len(packed_row), Q8_0_ROCMFPX_BLOCK_BYTES):
            block = packed_row[offset : offset + Q8_0_ROCMFPX_BLOCK_BYTES]
            scale = _ue4m3_scale(block[-1])
            decoded.extend(_signed_int8(value) * scale for value in block[:-1])
        rows.append(decoded)
    return torch.tensor(rows, dtype=torch.float32)


def test_type103_registers_authoritative_packed_geometry() -> None:
    assert GGML_TYPE_Q8_0_ROCMFPX == 103
    assert Q8_0_ROCMFPX_QK == 32
    assert Q8_0_ROCMFPX_BLOCK_BYTES == 33
    assert rocmfpx_types.gguf.GGML_QUANT_SIZES[GGML_TYPE_Q8_0_ROCMFPX] == (32, 33)


def test_type103_public_dense_fallback_matches_independent_int8_ue4m3_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU type 103 must materialize dense weights, never choose fused ops."""
    qweight = torch.stack(
        [
            _q8_block([0, 1, 127, 128, 255] + [2] * 27, 0x43),
            _q8_block([255, 128, 127, 1, 0] + [254] * 27, 0x48),
        ]
    )
    x = torch.tensor([[1.0] * 32, [0.5] * 32], dtype=torch.float32)

    def unexpected_native_op(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("type-103 CPU fallback selected a native matrix op")

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected_native_op)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected_native_op)

    output = linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q8_0_ROCMFPX)

    torch.testing.assert_close(output, x @ _reference_weight(qweight).T)


@pytest.mark.parametrize("activation_shape", [(2, 32), (2, 3, 32)])
def test_type103_public_dense_fallback_preserves_rank_and_values(
    monkeypatch: pytest.MonkeyPatch,
    activation_shape: tuple[int, ...],
) -> None:
    qweight = torch.stack(
        [
            _q8_block([1, 255] * 16, 0x40),
            _q8_block([2, 254] * 16, 0x48),
        ]
    )
    x = torch.arange(
        int(torch.tensor(activation_shape).prod()), dtype=torch.float32
    ).reshape(activation_shape)

    def unexpected_native_op(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("type-103 CPU fallback selected a native matrix op")

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected_native_op)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected_native_op)

    output = linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q8_0_ROCMFPX)

    assert output.shape == (*activation_shape[:-1], qweight.shape[0])
    torch.testing.assert_close(output, x @ _reference_weight(qweight).T)


@pytest.mark.parametrize(
    ("qweight", "x", "error", "message"),
    [
        (
            torch.zeros((2, Q8_0_ROCMFPX_BLOCK_BYTES - 1), dtype=torch.uint8),
            torch.zeros((1, 32), dtype=torch.float32),
            ValueError,
            "packed storage",
        ),
        (
            torch.zeros((2, Q8_0_ROCMFPX_BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 31), dtype=torch.float32),
            ValueError,
            "hidden size",
        ),
    ],
)
def test_type103_public_dense_fallback_rejects_invalid_storage_or_hidden_size(
    qweight: torch.Tensor,
    x: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q8_0_ROCMFPX)


@pytest.mark.parametrize("m", [0, -1, 1, 3])
def test_type103_public_dequantize_rejects_invalid_row_count_before_fallback_dispatch(
    monkeypatch: pytest.MonkeyPatch, m: int
) -> None:
    qweight = torch.zeros((2, Q8_0_ROCMFPX_BLOCK_BYTES), dtype=torch.uint8)

    def unexpected_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("type-103 invalid m reached fallback dispatch")

    monkeypatch.setattr(ops, "ggml_dequantize_triton", unexpected_fallback)

    with pytest.raises(ValueError, match="row count"):
        ops.ggml_dequantize(qweight, GGML_TYPE_Q8_0_ROCMFPX, m, 32, torch.float32)


@pytest.mark.parametrize(
    "qweight", [torch.tensor(0, dtype=torch.uint8), torch.zeros(33, dtype=torch.uint8)]
)
def test_type103_public_dequantize_rejects_rank_before_row_count(
    qweight: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="2D uint8 packed tensor"):
        ops.ggml_dequantize(qweight, GGML_TYPE_Q8_0_ROCMFPX, 1, 32, torch.float32)


@pytest.mark.parametrize("n", [0, -32])
def test_type103_public_dequantize_rejects_non_positive_columns_before_fallback(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    qweight = torch.zeros((2, Q8_0_ROCMFPX_BLOCK_BYTES), dtype=torch.uint8)

    def unexpected_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("type-103 invalid n reached fallback dispatch")

    monkeypatch.setattr(ops, "ggml_dequantize_triton", unexpected_fallback)

    with pytest.raises(ValueError, match="positive dimensions"):
        ops.ggml_dequantize(qweight, GGML_TYPE_Q8_0_ROCMFPX, 2, n, torch.float32)
