import struct

import gguf
import numpy as np
import pytest
import torch

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.ik_types import (
    GGML_TYPE_Q1_0_G128,
    Q1_0_G128_BLOCK_BYTES,
    QK_Q1_0_G128,
)
from vllm_gguf_plugin.llama_types import (
    GGML_TYPE_Q1_0,
    GGML_TYPE_Q2_0,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
)
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4_FAST
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton

ROCMFP4_FAST_CODEBOOK = (
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    10,
    0,
    -1,
    -2,
    -3,
    -4,
    -6,
    -8,
    -10,
)


def test_llama_quant_types_registered():
    assert GGML_TYPE_Q1_0 == 41
    assert GGML_TYPE_Q1_0_G128 == GGML_TYPE_Q1_0
    assert gguf.GGMLQuantizationType.Q1_0.value == 41
    assert gguf.GGMLQuantizationType.Q1_0_G128.value == 41
    assert gguf.GGMLQuantizationType.Q1_0_G128 is gguf.GGMLQuantizationType.Q1_0
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_Q1_0] == (128, 18)
    assert gguf.GGMLQuantizationType.TQ1_0 == GGML_TYPE_TQ1_0
    assert gguf.GGMLQuantizationType.TQ2_0 == GGML_TYPE_TQ2_0
    assert gguf.GGMLQuantizationType.Q2_0 == GGML_TYPE_Q2_0
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_TQ1_0] == (256, 54)
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_TQ2_0] == (256, 66)
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_Q2_0] == (64, 18)


def test_rocmfpx_cuda_capability_is_dequant_only():
    rocmfpx_types = {102, 103, 104, 107}
    assert rocmfpx_types.isdisjoint(ops._CUDA_GEMV_QUANT_TYPES)
    assert rocmfpx_types <= ops._CUDA_DEQUANT_ONLY_TYPES
    assert 100 in ops._CUDA_DEQUANT_ONLY_TYPES
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST in ops._CUDA_DEQUANT_ONLY_TYPES


def test_rocmfp4_fast_registered_with_expected_geometry():
    assert (
        gguf.GGMLQuantizationType.Q4_0_ROCMFP4_FAST.value
        == GGML_TYPE_Q4_0_ROCMFP4_FAST
        == 101
    )
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_Q4_0_ROCMFP4_FAST] == (32, 17)


def test_rocmfp4_fast_is_dequant_only():
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST in ops._CUDA_DEQUANT_ONLY_TYPES
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST not in ops._CUDA_GEMV_QUANT_TYPES


def test_rocmfp4_fast_reference_decode_uses_low_then_high_nibbles():
    quantized = bytes(
        low | (high << 4) for low, high in zip(range(16), range(15, -1, -1))
    )
    block = quantized + bytes([0x40])

    decoded = (
        ggml_dequantize_triton(
            torch.frombuffer(block, dtype=torch.uint8),
            GGML_TYPE_Q4_0_ROCMFP4_FAST,
            1,
            32,
            dtype=torch.float32,
        )
        .numpy()
        .reshape(-1)
    )
    expected = np.array(
        ROCMFP4_FAST_CODEBOOK + ROCMFP4_FAST_CODEBOOK[::-1],
        dtype=np.float32,
    )

    np.testing.assert_allclose(decoded, expected)


_NATIVE_TYPE_101 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not ops._cuda_kernel_available("ggml_dequantize", 101),
    reason="requires native type-101 CUDA extension",
)


@_NATIVE_TYPE_101
def test_rocmfp4_fast_native_decode_matches_reference():
    quantized = bytes(
        low | (high << 4) for low, high in zip(range(16), range(15, -1, -1))
    )
    output = _dequantize(quantized + bytes([0x40]), GGML_TYPE_Q4_0_ROCMFP4_FAST, 32)
    expected = torch.tensor(
        [*ROCMFP4_FAST_CODEBOOK, *ROCMFP4_FAST_CODEBOOK[::-1]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(output, expected.unsqueeze(0))


def _native_type_101_input(byte_count: int = 17) -> torch.Tensor:
    return torch.zeros(byte_count, dtype=torch.uint8, device="cuda")


@_NATIVE_TYPE_101
def test_rocmfp4_fast_native_decode_rejects_malformed_dtype():
    weight = torch.zeros(17, dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="uint8 scalar type"):
        torch.ops._C_gguf.ggml_dequantize(
            weight, GGML_TYPE_Q4_0_ROCMFP4_FAST, 1, 32, torch.float32
        )


@_NATIVE_TYPE_101
def test_rocmfp4_fast_native_decode_rejects_wrong_byte_count():
    with pytest.raises(RuntimeError, match="exactly 17 bytes"):
        torch.ops._C_gguf.ggml_dequantize(
            _native_type_101_input(16),
            GGML_TYPE_Q4_0_ROCMFP4_FAST,
            1,
            32,
            torch.float32,
        )


@_NATIVE_TYPE_101
def test_rocmfp4_fast_native_decode_rejects_non_block_aligned_columns():
    with pytest.raises(RuntimeError, match="divisible by 32"):
        torch.ops._C_gguf.ggml_dequantize(
            _native_type_101_input(17),
            GGML_TYPE_Q4_0_ROCMFP4_FAST,
            1,
            31,
            torch.float32,
        )


@_NATIVE_TYPE_101
def test_rocmfp4_fast_native_decode_rejects_non_contiguous_input():
    weight = torch.zeros(34, dtype=torch.uint8, device="cuda")[::2]
    assert not weight.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops._C_gguf.ggml_dequantize(
            weight, GGML_TYPE_Q4_0_ROCMFP4_FAST, 1, 32, torch.float32
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dequantize_q1_0_matches_raw_bit_reference():
    raw = _f16(2.0) + bytes([0b10000001, 0b01000010] + [0] * 14)
    output = _dequantize(raw, GGML_TYPE_Q1_0, QK_Q1_0_G128)
    expected = torch.tensor(
        [
            value
            for byte in raw[2:]
            for bit in range(8)
            for value in ([2.0] if byte & (1 << bit) else [-2.0])
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output, expected.unsqueeze(0))


def test_q1_0_geometry_constants_are_canonical():
    assert (QK_Q1_0_G128, Q1_0_G128_BLOCK_BYTES) == (128, 18)


def _f16(value: float) -> bytes:
    return struct.pack("<e", value)


@pytest.mark.parametrize(
    ("quant_type", "qk", "block", "expected"),
    [
        (
            GGML_TYPE_Q1_0,
            128,
            _f16(2.0) + bytes([0b10000001, 0b01000010] + [0] * 14),
            [
                value
                for byte in [0b10000001, 0b01000010] + [0] * 14
                for bit in range(8)
                for value in ([2.0] if byte & (1 << bit) else [-2.0])
            ],
        ),
        (
            GGML_TYPE_Q2_0,
            64,
            _f16(0.5) + bytes(range(16)),
            [
                ((byte >> (2 * index)) & 3) - 1
                for byte in bytes(range(16))
                for index in range(4)
            ],
        ),
    ],
)
def test_llama_reference_dequantize_on_cpu(
    quant_type: int, qk: int, block: bytes, expected: list[float | int]
):
    weight = torch.frombuffer(block, dtype=torch.uint8).clone()

    output = ops.ggml_dequantize(weight, quant_type, 1, qk, dtype=torch.float32)

    torch.testing.assert_close(output, torch.tensor([expected], dtype=torch.float32))


@pytest.mark.parametrize("quant_type", [GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0])
def test_tq_reference_dequantize_on_cpu(quant_type: int):
    qk = 256
    block = (
        bytes(range(48)) + bytes(range(4)) + _f16(2.0)
        if quant_type == GGML_TYPE_TQ1_0
        else bytes(range(64)) + _f16(0.5)
    )
    output = ggml_dequantize_triton(
        torch.frombuffer(block, dtype=torch.uint8).clone(),
        quant_type,
        1,
        qk,
        dtype=torch.float32,
    )
    assert output.shape == (1, qk)
    assert torch.isfinite(output).all()


def _dequantize(block: bytes, quant_type: int, n: int) -> torch.Tensor:
    return ops.ggml_dequantize(
        torch.tensor(np.frombuffer(block, dtype=np.uint8).copy(), device="cuda"),
        quant_type,
        1,
        n,
        dtype=torch.float32,
    ).cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dequantize_tq1_0():
    qs = bytes(range(48))
    qh = bytes(range(4))
    output = _dequantize(qs + qh + _f16(2.0), GGML_TYPE_TQ1_0, 256)
    pow3 = (1, 3, 9, 27, 81)
    expected = []
    for group in (qs[:32], qs[32:]):
        for p in pow3:
            expected.extend((((q * p) & 0xFF) * 3 >> 8) - 1 for q in group)
    for p in pow3[:4]:
        expected.extend((((q * p) & 0xFF) * 3 >> 8) - 1 for q in qh)
    torch.testing.assert_close(output, torch.tensor([expected]) * 2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("quant_type", "qk", "block"),
    [
        (GGML_TYPE_TQ2_0, 256, bytes(range(64)) + _f16(0.5)),
        (GGML_TYPE_Q2_0, 64, _f16(0.5) + bytes(range(16))),
    ],
)
def test_dequantize_2bit(quant_type: int, qk: int, block: bytes):
    output = _dequantize(block, quant_type, qk)
    qs = block[:64] if quant_type == GGML_TYPE_TQ2_0 else block[2:]
    if quant_type == GGML_TYPE_TQ2_0:
        expected = [
            ((qs[j + m] >> (2 * shift)) & 3) - 1
            for j in (0, 32)
            for shift in range(4)
            for m in range(32)
        ]
    else:
        expected = [((qs[j // 4] >> (2 * (j % 4))) & 3) - 1 for j in range(64)]
    torch.testing.assert_close(output, torch.tensor([expected]) * 0.5)
