import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization.utils import MMVQ_QUANT_TYPES
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4_FAST
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton


def _type_101_block() -> torch.Tensor:
    # One deterministic Q4_0_ROCMFP4_FAST block: low nibbles then high nibbles.
    return torch.tensor([0x21] * 16 + [0x40], dtype=torch.uint8)


def _enable_fused_type_101_hook(monkeypatch, hook):
    monkeypatch.setattr(
        ops,
        "_cuda_kernel_available",
        lambda op_name, quant_type: (
            op_name == "ggml_mul_mat_vec_rocmfp4_fast"
            and quant_type == GGML_TYPE_Q4_0_ROCMFP4_FAST
        ),
    )
    monkeypatch.setattr(
        ops,
        "ggml_mul_mat_vec_rocmfp4_fast",
        hook,
        raising=False,
    )
    monkeypatch.setattr(
        ops,
        "ggml_mul_mat_a8_triton",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("type 101 must not fall back to generic GEMV")
        ),
    )


def test_type_101_uses_dedicated_fused_gemv_capability():
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST in MMVQ_QUANT_TYPES
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST not in ops._CUDA_GEMV_QUANT_TYPES
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST in ops._CUDA_DEQUANT_ONLY_TYPES


def test_type_101_cpu_inputs_use_dequantize_fallback_when_native_available(
    monkeypatch,
):
    qweight = _type_101_block().reshape(1, -1)
    activation = torch.arange(1, 33, dtype=torch.float32).reshape(1, -1)
    expected = activation @ ggml_dequantize_triton(
        qweight.reshape(-1),
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        1,
        32,
        dtype=activation.dtype,
    ).T
    calls = []

    def fused_hook(weight, x, quant_type, row):
        calls.append((weight, x, quant_type, row))
        raise AssertionError("native hook must not receive CPU tensors")

    _enable_fused_type_101_hook(monkeypatch, fused_hook)

    output = ops.ggml_mul_mat_vec_a8(
        qweight,
        activation,
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        1,
    )

    assert calls == []
    torch.testing.assert_close(output, expected)


def test_fused_type_101_gemv_matches_dequantize_then_matmul_on_cpu(monkeypatch):
    qweight = _type_101_block().reshape(1, -1)
    activation = torch.arange(1, 33, dtype=torch.float32).reshape(1, -1)
    expected = activation @ ggml_dequantize_triton(
        qweight.reshape(-1),
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        1,
        32,
        dtype=activation.dtype,
    ).T

    def fused_hook(weight, x, quant_type, row):
        decoded = ggml_dequantize_triton(
            weight.reshape(-1), quant_type, row, x.shape[1], dtype=x.dtype
        )
        return x @ decoded.T

    _enable_fused_type_101_hook(monkeypatch, fused_hook)

    output = ops.ggml_mul_mat_vec_a8(
        qweight,
        activation,
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        1,
    )

    torch.testing.assert_close(output, expected)


def test_type_101_gemv_dequantizes_when_native_op_unavailable(monkeypatch):
    qweight = _type_101_block().reshape(1, -1)
    activation = torch.arange(1, 33, dtype=torch.float32).reshape(1, -1)
    expected = activation @ ggml_dequantize_triton(
        qweight.reshape(-1),
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        1,
        32,
        dtype=activation.dtype,
    ).T

    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *_args: False)
    monkeypatch.setattr(
        ops,
        "ggml_mul_mat_a8_triton",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("type 101 must not fall back to generic GEMV")
        ),
    )

    output = ops.ggml_mul_mat_vec_a8(
        qweight,
        activation,
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        1,
    )

    torch.testing.assert_close(output, expected)


def test_type_101_gemv_rejects_malformed_row_count(monkeypatch):
    qweight = _type_101_block().reshape(1, -1)
    activation = torch.arange(1, 33, dtype=torch.float32).reshape(1, -1)

    with pytest.raises(
        ValueError,
        match=r"GEMV row count 2 does not match weight rows 1",
    ):
        ops.ggml_mul_mat_vec_a8(
            qweight,
            activation,
            GGML_TYPE_Q4_0_ROCMFP4_FAST,
            2,
        )
