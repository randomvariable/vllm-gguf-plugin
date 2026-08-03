"""RED contract gaps for type-101 dense GEMM."""

from collections.abc import Sequence
from math import prod

import pytest
import torch

from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4_FAST
from vllm_gguf_plugin.triton.gemm.standard_quant.q4_0_rocmfp4_fast import (
    ggml_gemm_q4_0_rocmfp4_fast_triton,
)

BLOCK_BYTES = 17
BLOCK_SIZE = 32
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)


def _scale(scale_byte: int) -> float:
    if not 0 <= scale_byte <= 0x7E:
        return 0.0
    exponent, mantissa = scale_byte >> 3, scale_byte & 0x07
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _pack_block(codes: Sequence[int], scale_byte: int) -> list[int]:
    assert len(codes) == BLOCK_SIZE
    return [codes[index] | (codes[index + 16] << 4) for index in range(16)] + [
        scale_byte
    ]


def _weights(rows: int, blocks: int) -> torch.Tensor:
    packed: list[int] = []
    for row in range(rows):
        for block in range(blocks):
            codes = [
                (row + block + index) % len(CODEBOOK) for index in range(BLOCK_SIZE)
            ]
            packed.extend(_pack_block(codes, 0x40 + block))
    return torch.tensor(packed, dtype=torch.uint8).reshape(rows, blocks * BLOCK_BYTES)


def _oracle(weights: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    decoded_rows: list[list[float]] = []
    for packed_row in weights.tolist():
        decoded: list[float] = []
        for start in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[start : start + BLOCK_BYTES]
            scale = _scale(block[16])
            decoded.extend(CODEBOOK[value & 0x0F] * scale for value in block[:16])
            decoded.extend(CODEBOOK[value >> 4] * scale for value in block[:16])
        decoded_rows.append(decoded)
    dense = torch.tensor(decoded_rows, dtype=torch.float32)
    return (activations.float() @ dense.T).to(activations.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation_shape", [(17, 96), (2, 3, 96)])
def test_type101_cpu_dense_gemm_matches_independent_oracle_for_contract_shapes(
    dtype: torch.dtype, activation_shape: tuple[int, ...]
) -> None:
    """Rank, tail, multi-block, and dtype behavior have no decoder dependency."""
    weights = _weights(rows=17, blocks=3)
    activations = (
        torch.arange(prod(activation_shape), dtype=dtype).reshape(activation_shape) / 31
    )

    output = ggml_gemm_q4_0_rocmfp4_fast_triton(weights, activations, row=17)

    assert output.shape == (*activation_shape[:-1], 17)
    assert output.dtype == dtype
    torch.testing.assert_close(output, _oracle(weights, activations), atol=1, rtol=8e-3)


@pytest.mark.parametrize("activation_shape", [(0, BLOCK_SIZE), (1, 0, BLOCK_SIZE)])
def test_type101_dense_gemm_rejects_empty_activation_dimensions(
    activation_shape: tuple[int, ...],
) -> None:
    """Malformed activations must fail rather than silently return an empty GEMM."""
    with pytest.raises(ValueError, match="positive"):
        ggml_gemm_q4_0_rocmfp4_fast_triton(
            _weights(rows=1, blocks=1),
            torch.empty(activation_shape, dtype=torch.float32),
            row=1,
        )


def test_type101_public_type_id_remains_explicit_in_contract() -> None:
    """Test fixture must target only packed type-101 dispatch."""
    assert GGML_TYPE_Q4_0_ROCMFP4_FAST == 101
