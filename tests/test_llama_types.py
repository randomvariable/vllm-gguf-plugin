import struct

import gguf
import numpy as np
import pytest
import torch

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.llama_types import (
    GGML_TYPE_Q2_0,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
)


def test_llama_quant_types_registered():
    assert gguf.GGMLQuantizationType.TQ1_0 == GGML_TYPE_TQ1_0
    assert gguf.GGMLQuantizationType.TQ2_0 == GGML_TYPE_TQ2_0
    assert gguf.GGMLQuantizationType.Q2_0 == GGML_TYPE_Q2_0
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_TQ1_0] == (256, 54)
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_TQ2_0] == (256, 66)
    assert gguf.GGML_QUANT_SIZES[GGML_TYPE_Q2_0] == (64, 18)


def _f16(value: float) -> bytes:
    return struct.pack("<e", value)


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
