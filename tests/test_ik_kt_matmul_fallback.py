# SPDX-License-Identifier: Apache-2.0
"""Public matmul fallback coverage for ik_llama KT row-prefix formats."""

from __future__ import annotations

import importlib

import pytest
import torch

_KT_TYPE_NAMES = ("IQ1_KT", "IQ2_KT", "IQ3_KT", "IQ4_KT")


class _RecordingDequantize:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.calls: list[tuple[int, int, int, torch.dtype | None]] = []

    def __call__(
        self,
        qweight: torch.Tensor,
        quant_type: int,
        m: int,
        n: int,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        self.calls.append((int(quant_type), int(m), int(n), dtype))
        assert qweight.device == self.weight.device
        assert (m, n) == self.weight.shape
        return self.weight.to(dtype=dtype, device=qweight.device)


@pytest.mark.parametrize(
    ("wrapper", "tokens"),
    [
        ("ggml_mul_mat_vec_a8", 1),
        ("ggml_mul_mat_a8", 3),
    ],
)
@pytest.mark.parametrize("quant_type_name", _KT_TYPE_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ik_kt_matmul_dequantizes_row_prefix_weights_before_matmul(
    monkeypatch, quant_type_name, wrapper, tokens, dtype
):
    """KT types use dense reference fallback, not direct Triton GEMM kernels."""
    # Keep collection usable when GPU-only plugin dependencies are absent.
    try:
        ops = importlib.import_module("vllm_gguf_plugin.ops")
        ik_types = importlib.import_module("vllm_gguf_plugin.ik_types")
    except ModuleNotFoundError as error:
        pytest.skip(f"plugin dependencies unavailable: {error.name}")

    quant_type = getattr(ik_types, f"GGML_TYPE_{quant_type_name}")
    wrapper_fn = getattr(ops, wrapper)
    rows = 3
    blocks_per_row = 2
    row_meta, payload_bytes, block_size = ik_types.ROW_PREFIX_GGUF_TYPES[quant_type]
    row_bytes = row_meta + blocks_per_row * payload_bytes
    qweight = torch.zeros((rows, row_bytes), dtype=torch.uint8)
    shape = ik_types.gguf_qweight_dequant_shape(rows, row_bytes, quant_type)
    assert shape == (rows, blocks_per_row * block_size)

    dense_weight = torch.arange(
        rows * shape[1], dtype=torch.float32
    ).reshape(shape) / 128
    x = torch.arange(tokens * shape[1], dtype=torch.float32).reshape(tokens, shape[1])
    x = (x / 64).to(dtype)
    dequantize = _RecordingDequantize(dense_weight)

    # KT formats have no direct CUDA GEMV/GEMM kernels, even when an extension exists.
    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *args: False)
    monkeypatch.setattr(ops, "_cuda_gemm_kernel_available", lambda *args: False)
    monkeypatch.setattr(ops, "ggml_dequantize", dequantize)

    actual = wrapper_fn(qweight, x, quant_type, rows)

    expected = x @ dense_weight.to(dtype).T
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert actual.shape == (tokens, rows)
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert dequantize.calls == [(quant_type, rows, shape[1], x.dtype)]
