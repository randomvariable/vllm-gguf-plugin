"""RED contract for GGML type-100 Q4_0_ROCMFP4 dense GEMM."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4

BLOCK_BYTES = 18
BLOCK_SIZE = 32
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)
ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _ue4m3_scale(scale_byte: int) -> float:
    """Independent UE4M3 bias-8 scale oracle for type-100 ABI bytes."""
    if scale_byte >= 0x7F:
        return 0.0
    exponent, mantissa = divmod(scale_byte, 8)
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _pack_block(codes: Sequence[int], low_scale: int, high_scale: int) -> list[int]:
    assert len(codes) == BLOCK_SIZE
    return [codes[index] | (codes[index + 16] << 4) for index in range(16)] + [
        low_scale,
        high_scale,
    ]


def _make_weights(rows: int, blocks: int) -> torch.Tensor:
    packed_blocks: list[list[int]] = []
    scales = ((0x40, 0x48), (0x7F, 0x40), (0x40, 0xFF))
    for row in range(rows):
        for block in range(blocks):
            codes = [
                (row * 3 + block * 5 + index) % len(CODEBOOK)
                for index in range(BLOCK_SIZE)
            ]
            packed_blocks.append(_pack_block(codes, *scales[block % len(scales)]))
    return torch.tensor(packed_blocks, dtype=torch.uint8).reshape(
        rows, blocks * BLOCK_BYTES
    )


def _reference_decode(weights: torch.Tensor) -> torch.Tensor:
    """Decode contiguous low then high halves without plugin decoder helpers."""
    assert weights.dtype == torch.uint8
    assert weights.ndim == 2
    assert weights.shape[1] % BLOCK_BYTES == 0

    decoded_rows: list[list[float]] = []
    for packed_row in weights.cpu().tolist():
        decoded: list[float] = []
        for start in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[start : start + BLOCK_BYTES]
            low_scale, high_scale = map(_ue4m3_scale, block[16:18])
            decoded.extend(CODEBOOK[value & 0x0F] * low_scale for value in block[:16])
            decoded.extend(CODEBOOK[value >> 4] * high_scale for value in block[:16])
        decoded_rows.append(decoded)
    return torch.tensor(decoded_rows, dtype=torch.float32, device=weights.device)


def _reference_gemm(weights: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    return (activations.float() @ _reference_decode(weights).T).to(activations.dtype)


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
    elif dtype == torch.float16:
        torch.testing.assert_close(actual, expected, atol=0.125, rtol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, atol=1.0, rtol=8e-3)


@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
@pytest.mark.parametrize(
    ("activation_shape", "rows"),
    [
        ((7, 96), 17),
        ((2, 3, 96), 19),
    ],
)
def test_type100_public_dense_gemm_matches_independent_abi_oracle(
    dtype: torch.dtype, activation_shape: tuple[int, ...], rows: int
) -> None:
    """Dense GEMM must preserve rank, tail rows, and contiguous scale halves."""
    weights = _make_weights(rows, blocks=3)
    numel = 1
    for size in activation_shape:
        numel *= size
    activations = torch.arange(numel, dtype=dtype).reshape(activation_shape).div(31)

    output = ops.ggml_mul_mat_a8(weights, activations, GGML_TYPE_Q4_0_ROCMFP4, rows)

    assert output.shape == (*activation_shape[:-1], rows)
    assert output.dtype == dtype
    _assert_close(output, _reference_gemm(weights, activations), dtype)


@pytest.mark.parametrize(
    ("weights", "activations", "row"),
    [
        (
            torch.empty((0, BLOCK_BYTES), dtype=torch.uint8),
            torch.ones((1, BLOCK_SIZE), dtype=torch.float32),
            0,
        ),
        (
            torch.empty((1, 0), dtype=torch.uint8),
            torch.empty((1, 0), dtype=torch.float32),
            1,
        ),
    ],
)
def test_type100_public_dense_gemm_rejects_empty_dimensions(
    weights: torch.Tensor, activations: torch.Tensor, row: int
) -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        ops.ggml_mul_mat_a8(weights, activations, GGML_TYPE_Q4_0_ROCMFP4, row)
