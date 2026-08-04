"""Public dispatch contracts for ROCmFPX dense GEMM (types 102, 104, 107).

Types 100 and 103 have dedicated suites; 101 has its own GEMV/GEMM split.
These three share the table-driven route in ``linear._fused_mul_mat_gguf``
via ``ops.ROCMFPX_GEMM_BLOCK_BYTES``, so they are covered together.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import linear
from vllm_gguf_plugin.rocmfpx_types import (
    GGML_TYPE_Q2_0_ROCMFPX,
    GGML_TYPE_Q3_0_ROCMFPX,
    GGML_TYPE_Q6_0_ROCMFPX,
)

BLOCK_SIZE = 32
WEIGHT_ROWS = 4

# (quant type, packed bytes per 32-weight block)
TABLE_DRIVEN_TYPES = [
    pytest.param(GGML_TYPE_Q6_0_ROCMFPX, 26, id="q6_0_rocmfpx"),
    pytest.param(GGML_TYPE_Q3_0_ROCMFPX, 14, id="q3_0_rocmfpx"),
    pytest.param(GGML_TYPE_Q2_0_ROCMFPX, 10, id="q2_0_rocmfpx"),
]


def _weights(block_bytes: int, blocks: int = 1) -> torch.Tensor:
    return torch.zeros((WEIGHT_ROWS, blocks * block_bytes), dtype=torch.uint8)


def _unexpected(*_: object, **__: object) -> torch.Tensor:
    pytest.fail("ROCmFPX type reached the dequant/GEMV fallback")


@pytest.mark.parametrize(("quant_type", "block_bytes"), TABLE_DRIVEN_TYPES)
@pytest.mark.parametrize("m", [1, 2, 6, 7, 32])
def test_rocmfpx_fused_routes_every_batch_size_to_gemm(
    monkeypatch: pytest.MonkeyPatch, quant_type: int, block_bytes: int, m: int
) -> None:
    """Eligible devices use the dedicated GEMM for every batch size.

    None of these formats has a native GEMV, so unlike type-101 there is no
    small-M split to fall back on.
    """
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: True)
    calls: list[tuple[int, int]] = []

    def fake_gemm(
        qweight: torch.Tensor, x: torch.Tensor, qweight_type: int, rows: int
    ) -> torch.Tensor:
        calls.append((qweight_type, rows))
        return torch.zeros((*x.shape[:-1], rows), dtype=x.dtype)

    monkeypatch.setattr(ops, "ggml_mul_mat_a8", fake_gemm)
    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", _unexpected)
    monkeypatch.setattr(ops, "ggml_dequantize", _unexpected)

    x = torch.zeros((m, BLOCK_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(block_bytes), quant_type)

    assert calls == [(quant_type, WEIGHT_ROWS)]
    assert output.shape == (m, WEIGHT_ROWS)


@pytest.mark.parametrize(("quant_type", "block_bytes"), TABLE_DRIVEN_TYPES)
def test_rocmfpx_fused_preserves_rank3_output_shape(
    monkeypatch: pytest.MonkeyPatch, quant_type: int, block_bytes: int
) -> None:
    """Rank-3 activations keep their leading dimensions through the GEMM."""
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: True)
    calls: list[str] = []

    def fake_gemm(
        qweight: torch.Tensor, x: torch.Tensor, qweight_type: int, rows: int
    ) -> torch.Tensor:
        calls.append("gemm")
        return torch.zeros((*x.shape[:-1], rows), dtype=x.dtype)

    monkeypatch.setattr(ops, "ggml_mul_mat_a8", fake_gemm)
    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", _unexpected)
    monkeypatch.setattr(ops, "ggml_dequantize", _unexpected)

    x = torch.zeros((2, 3, BLOCK_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(block_bytes), quant_type)

    assert calls == ["gemm"]
    assert output.shape == (2, 3, WEIGHT_ROWS)


@pytest.mark.parametrize(("quant_type", "block_bytes"), TABLE_DRIVEN_TYPES)
def test_rocmfpx_falls_back_to_dequant_when_ineligible(
    monkeypatch: pytest.MonkeyPatch, quant_type: int, block_bytes: int
) -> None:
    """Outside the gfx115x envelope the route must degrade, not fail closed.

    This is the contract that keeps CPU and non-gfx115x accelerators working:
    the dedicated Triton GEMM is skipped and dense dequant-plus-matmul runs.
    """
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: False)
    dequantized: list[int] = []

    def fake_dequant(
        qweight: torch.Tensor, qweight_type: int, m: int, n: int, dtype: torch.dtype
    ) -> torch.Tensor:
        dequantized.append(qweight_type)
        return torch.zeros((m, n), dtype=dtype)

    def unexpected_gemm(*_: object, **__: object) -> torch.Tensor:
        pytest.fail("ineligible device reached the dedicated GEMM")

    monkeypatch.setattr(ops, "ggml_dequantize", fake_dequant)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected_gemm)

    x = torch.zeros((4, BLOCK_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(block_bytes), quant_type)

    assert dequantized == [quant_type]
    assert output.shape == (4, WEIGHT_ROWS)


@pytest.mark.parametrize(("quant_type", "block_bytes"), TABLE_DRIVEN_TYPES)
def test_rocmfpx_multi_block_rows_reach_gemm(
    monkeypatch: pytest.MonkeyPatch, quant_type: int, block_bytes: int
) -> None:
    """Hidden sizes spanning several packed blocks still select the GEMM."""
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda W_, X_: True)
    calls: list[int] = []

    def fake_gemm(
        qweight: torch.Tensor, x: torch.Tensor, qweight_type: int, rows: int
    ) -> torch.Tensor:
        calls.append(qweight_type)
        return torch.zeros((*x.shape[:-1], rows), dtype=x.dtype)

    monkeypatch.setattr(ops, "ggml_mul_mat_a8", fake_gemm)
    monkeypatch.setattr(ops, "ggml_dequantize", _unexpected)

    blocks = 3
    x = torch.zeros((5, blocks * BLOCK_SIZE), dtype=torch.float32)
    output = linear._fused_mul_mat_gguf(x, _weights(block_bytes, blocks), quant_type)

    assert calls == [quant_type]
    assert output.shape == (5, WEIGHT_ROWS)


def test_every_table_driven_type_is_covered() -> None:
    """Guard against a new ROCmFPX format landing without dispatch coverage."""
    covered = {param.values[0] for param in TABLE_DRIVEN_TYPES}
    # Types 100 and 103 have dedicated suites of their own.
    dedicated = {100, 103}
    assert covered | dedicated == set(ops.ROCMFPX_GEMM_BLOCK_BYTES)
