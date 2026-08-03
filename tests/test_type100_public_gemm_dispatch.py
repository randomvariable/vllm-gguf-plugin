"""Public dispatch contracts for type-100 (Q4_0_ROCMFP4) dense GEMM."""

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import linear
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4

BLOCK_BYTES = 18
BLOCK_SIZE = 32
HIDDEN_SIZE = BLOCK_SIZE  # one packed block per row
WEIGHT_ROWS = 4


def _weights(rows: int = WEIGHT_ROWS) -> torch.Tensor:
    return torch.zeros((rows, BLOCK_BYTES), dtype=torch.uint8)


@pytest.mark.parametrize("m", [1, 2, 6, 7, 32])
def test_type100_fused_routes_every_batch_to_gemm_wrapper(
    monkeypatch: pytest.MonkeyPatch, m: int
) -> None:
    """On eligible devices type-100 uses ops GEMM for every batch size.

    Type-100 has no native GEMV, so there is no small-M split.
    """
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: True)
    calls: list[tuple[int, int]] = []

    def fake_gemm(
        qweight: torch.Tensor, x: torch.Tensor, qweight_type: int, rows: int
    ) -> torch.Tensor:
        calls.append((qweight_type, rows))
        return torch.zeros((*x.shape[:-1], rows), dtype=x.dtype)

    def unexpected(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("type-100 reached dequant/GEMV fallback")

    monkeypatch.setattr(ops, "ggml_mul_mat_a8", fake_gemm)
    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected)
    monkeypatch.setattr(ops, "ggml_dequantize", unexpected)

    x = torch.zeros((m, HIDDEN_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(), GGML_TYPE_Q4_0_ROCMFP4)

    assert calls == [(GGML_TYPE_Q4_0_ROCMFP4, WEIGHT_ROWS)]
    assert output.shape == (m, WEIGHT_ROWS)


def test_type100_fused_preserves_rank3_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-3 activations keep leading dimensions in GEMM outputs."""
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: True)
    calls: list[str] = []

    def fake_gemm(
        qweight: torch.Tensor, x: torch.Tensor, qweight_type: int, rows: int
    ) -> torch.Tensor:
        calls.append("gemm")
        return torch.zeros((*x.shape[:-1], rows), dtype=x.dtype)

    def unexpected(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("type-100 reached dequant/GEMV fallback")

    monkeypatch.setattr(ops, "ggml_mul_mat_a8", fake_gemm)
    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected)
    monkeypatch.setattr(ops, "ggml_dequantize", unexpected)

    x = torch.zeros((2, 3, HIDDEN_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(), GGML_TYPE_Q4_0_ROCMFP4)

    assert calls == ["gemm"]
    assert output.shape == (2, 3, WEIGHT_ROWS)


@pytest.mark.parametrize(
    "qweight,x,error,message",
    [
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.float32),
            torch.zeros((0, HIDDEN_SIZE), dtype=torch.float32),
            TypeError,
            "weights must use uint8",
        ),
        (
            torch.zeros((1, 1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 0, HIDDEN_SIZE), dtype=torch.float32),
            ValueError,
            "weights must be 2D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((HIDDEN_SIZE,), dtype=torch.float32),
            ValueError,
            "activations must be 2D or 3D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((0, HIDDEN_SIZE), dtype=torch.int32),
            TypeError,
            "float16, bfloat16, or float32",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 0, HIDDEN_SIZE - 1), dtype=torch.float32),
            ValueError,
            "hidden size does not match packed weights",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((0, HIDDEN_SIZE), dtype=torch.float32),
            ValueError,
            "positive dimensions",
        ),
    ],
)
def test_type100_fused_rejects_malformed_input_before_empty_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
    qweight: torch.Tensor,
    x: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed type-100 inputs fail validation before any routing."""

    def unexpected_route(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("malformed type-100 input reached a dispatch route")

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected_route)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected_route)
    monkeypatch.setattr(ops, "ggml_dequantize", unexpected_route)

    with pytest.raises(error, match=message):
        linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q4_0_ROCMFP4)


def test_type100_fused_uses_dequant_fallback_when_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ineligible devices stay fail-closed on dequantize-plus-dense."""
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: False)
    calls: list[str] = []

    def fake_dequant(
        qweight: torch.Tensor,
        quant_type: int,
        rows: int,
        cols: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        calls.append("dequant")
        return torch.zeros((rows, cols), dtype=dtype)

    def unexpected(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("ineligible type-100 reached a fused route")

    monkeypatch.setattr(ops, "ggml_dequantize", fake_dequant)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected)
    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected)

    x = torch.zeros((8, HIDDEN_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(), GGML_TYPE_Q4_0_ROCMFP4)

    assert calls == ["dequant"]
    assert output.shape == (8, WEIGHT_ROWS)


def test_type100_ops_prefers_triton_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible devices take the dedicated Triton GEMM path."""
    W = _weights(rows=1)
    X = torch.zeros((1, HIDDEN_SIZE), dtype=torch.float32)
    calls: list[int] = []

    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: True)
    monkeypatch.setattr(
        ops,
        "ggml_mul_mat_a8_triton",
        lambda W_, X_, qt, row: (calls.append(qt), torch.zeros((row, 1)))[1],
    )

    def fail_dequant(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("dequant fallback used despite Triton availability")

    monkeypatch.setattr(ops, "ggml_dequantize", fail_dequant)

    ops.ggml_mul_mat_a8(W, X, GGML_TYPE_Q4_0_ROCMFP4, row=1)
    assert calls == [GGML_TYPE_Q4_0_ROCMFP4]


def test_type100_ops_falls_back_when_triton_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ineligible devices dequantize + dense matmul instead of the kernel."""
    W = _weights(rows=2)
    X = torch.ones((3, HIDDEN_SIZE), dtype=torch.float32)
    decoded_rows: list[int] = []

    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: False)

    def fail_triton(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("Triton GEMM used despite unavailable eligibility")

    monkeypatch.setattr(ops, "ggml_mul_mat_a8_triton", fail_triton)

    def fake_dequant(
        qweight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype
    ) -> torch.Tensor:
        decoded_rows.append(m)
        return torch.zeros((m, n), dtype=dtype)

    monkeypatch.setattr(ops, "ggml_dequantize", fake_dequant)

    output = ops.ggml_mul_mat_a8(W, X, GGML_TYPE_Q4_0_ROCMFP4, row=2)

    assert decoded_rows == [2]
    assert output.shape == (3, 2)
