"""RED contracts for public type-101 dense GEMM dispatch."""

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import linear
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4_FAST

BLOCK_BYTES = 17
HIDDEN_SIZE = 32
# Type-101 uses the non-i-matrix threshold when output rows are <= 5120.
GEMM_THRESHOLD = 6


@pytest.mark.parametrize(
    ("activation_shape", "expected_route"),
    [
        ((1, GEMM_THRESHOLD - 1, HIDDEN_SIZE), "gemv"),
        ((1, GEMM_THRESHOLD, HIDDEN_SIZE), "gemv"),
        ((1, GEMM_THRESHOLD + 1, HIDDEN_SIZE), "gemm"),
    ],
)
def test_type101_fused_dispatch_uses_flattened_activation_rows(
    monkeypatch: pytest.MonkeyPatch,
    activation_shape: tuple[int, int, int],
    expected_route: str,
) -> None:
    """3D inputs select GEMV/GEMM from flattened M, not batch dimension."""
    calls: list[str] = []
    x = torch.zeros(activation_shape, dtype=torch.float32)
    qweight = torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8)

    def fake_gemv(*args: object) -> torch.Tensor:
        calls.append("gemv")
        return torch.empty((*x.shape[:-1], qweight.shape[0]), dtype=x.dtype)

    def fake_gemm(*args: object) -> torch.Tensor:
        calls.append("gemm")
        return torch.empty((*x.shape[:-1], qweight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", fake_gemv)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", fake_gemm)

    linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q4_0_ROCMFP4_FAST)

    assert calls == [expected_route]


def test_type101_fused_gemv_flattens_rank3_activation_and_restores_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-3 type-101 GEMV receives flattened activations and restores rank."""
    x = torch.zeros((1, GEMM_THRESHOLD, HIDDEN_SIZE), dtype=torch.float32)
    qweight = torch.zeros((3, BLOCK_BYTES), dtype=torch.uint8)
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_gemv(
        _qweight: torch.Tensor,
        activation: torch.Tensor,
        _qweight_type: int,
        rows: int,
    ) -> torch.Tensor:
        calls.append((tuple(activation.shape), rows))
        return torch.empty((activation.shape[0], rows), dtype=activation.dtype)

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", fake_gemv)

    output = linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q4_0_ROCMFP4_FAST)

    assert calls == [((GEMM_THRESHOLD, HIDDEN_SIZE), qweight.shape[0])]
    assert output.shape == (*x.shape[:-1], qweight.shape[0])


@pytest.mark.parametrize(
    "activation_shape,dtype",
    [
        ((GEMM_THRESHOLD, HIDDEN_SIZE), torch.float16),
        ((GEMM_THRESHOLD, HIDDEN_SIZE), torch.bfloat16),
        ((GEMM_THRESHOLD, HIDDEN_SIZE), torch.float32),
        ((1, GEMM_THRESHOLD, HIDDEN_SIZE), torch.float16),
        ((1, GEMM_THRESHOLD, HIDDEN_SIZE), torch.bfloat16),
        ((1, GEMM_THRESHOLD, HIDDEN_SIZE), torch.float32),
    ],
)
def test_type101_fused_small_m_gemv_accepts_public_activation_contract(
    monkeypatch: pytest.MonkeyPatch,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Small-M type-101 inputs validate before routing to GEMV."""
    x = torch.zeros(activation_shape, dtype=dtype)
    qweight = torch.zeros((2, BLOCK_BYTES), dtype=torch.uint8)
    calls: list[tuple[tuple[int, ...], torch.dtype]] = []

    def fake_gemv(
        _qweight: torch.Tensor,
        activation: torch.Tensor,
        _qweight_type: int,
        rows: int,
    ) -> torch.Tensor:
        calls.append((tuple(activation.shape), activation.dtype))
        return torch.empty((activation.shape[0], rows), dtype=activation.dtype)

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", fake_gemv)
    monkeypatch.setattr(
        ops,
        "ggml_mul_mat_a8",
        lambda *_args: pytest.fail("valid small-M input reached GEMM"),
    )

    output = linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q4_0_ROCMFP4_FAST)

    assert calls == [((GEMM_THRESHOLD, HIDDEN_SIZE), dtype)]
    assert output.shape == (*activation_shape[:-1], qweight.shape[0])
    assert output.dtype == dtype


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
            torch.zeros(HIDDEN_SIZE, dtype=torch.float32),
            ValueError,
            "activations must be 2D or 3D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((0, HIDDEN_SIZE), dtype=torch.int32),
            TypeError,
            "activations must use float16, bfloat16, or float32",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 0, HIDDEN_SIZE - 1), dtype=torch.float32),
            ValueError,
            "hidden size does not match packed weights",
        ),
    ],
)
def test_type101_fused_rejects_malformed_input_before_empty_short_circuit_or_route(
    monkeypatch: pytest.MonkeyPatch,
    qweight: torch.Tensor,
    x: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Empty type-101 activations cannot bypass public input validation."""

    def unexpected_route(*_args: object) -> torch.Tensor:
        pytest.fail("malformed type-101 input reached a dispatch route")

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected_route)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected_route)

    with pytest.raises(error, match=message):
        linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q4_0_ROCMFP4_FAST)


def test_type101_fused_fake_restores_rank3_metadata_shape() -> None:
    """Fake registration exposes output metadata for every activation dimension."""
    x = torch.empty((2, 3, HIDDEN_SIZE), device="meta")
    qweight = torch.empty((5, BLOCK_BYTES), dtype=torch.uint8, device="meta")

    output = linear._fused_mul_mat_gguf_fake(x, qweight, GGML_TYPE_Q4_0_ROCMFP4_FAST)

    assert output.shape == (*x.shape[:-1], qweight.shape[0])


@pytest.mark.parametrize(
    ("weights", "activations", "row", "error", "message"),
    [
        (
            torch.zeros((1, 1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.float32),
            1,
            ValueError,
            "weights must be 2D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros(HIDDEN_SIZE, dtype=torch.float32),
            1,
            ValueError,
            "2D or 3D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.float32),
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.float32),
            1,
            TypeError,
            "uint8",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.int32),
            1,
            TypeError,
            "float16",
        ),
        (
            torch.empty((1, BLOCK_BYTES), dtype=torch.uint8, device="meta"),
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.float32),
            1,
            ValueError,
            "same device",
        ),
        (
            torch.zeros((1, BLOCK_BYTES - 1), dtype=torch.uint8),
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.float32),
            1,
            ValueError,
            "divisible by 17",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, HIDDEN_SIZE - 1), dtype=torch.float32),
            1,
            ValueError,
            "hidden size",
        ),
    ],
)
def test_type101_public_gemm_validates_before_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    weights: torch.Tensor,
    activations: torch.Tensor,
    row: int,
    error: type[Exception],
    message: str,
) -> None:
    """Public API owns type-101 input validation, before route capability checks."""

    def backend_probe(*args: object) -> bool:
        pytest.fail("malformed type-101 inputs reached backend selection")

    monkeypatch.setattr(ops, "_type101_triton_gemm_available", backend_probe)

    with pytest.raises(error, match=message):
        ops.ggml_mul_mat_a8(
            weights,
            activations,
            GGML_TYPE_Q4_0_ROCMFP4_FAST,
            row=row,
        )
