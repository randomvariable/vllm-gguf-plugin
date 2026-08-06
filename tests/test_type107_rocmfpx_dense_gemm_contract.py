# SPDX-License-Identifier: Apache-2.0
"""RED contract for GGML type-107 Q2_0_ROCMFPX dense GEMM.

The oracle in this module is written directly from the ``block_rocmfp2`` ABI
in ``vllm_gguf_plugin/csrc/gguf/ggml-common.h``: 32 weights in 10 bytes as
``uint8_t qs[8]`` (four 2-bit indices per byte, index ``i`` at bit ``2 * i``)
plus ``uint8_t e[2]`` UE4M3 half-scales, one per 16 weights. It deliberately
shares no helper with the plugin decoders so that a shared assumption cannot
mask a kernel error.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from tests.numerics import assert_gemm_close
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

GGML_TYPE_Q2_0_ROCMFPX = 107
BLOCK_BYTES = 10
BLOCK_SIZE = 32
QS_BYTES = 8
CODEBOOK = (-4, -1, 1, 4)
ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _ue4m3_scale(scale_byte: int) -> float:
    """Independent UE4M3 bias-8 scale oracle for type-107 ABI bytes."""
    if scale_byte > 0x7E:
        return 0.0
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _pack_block(codes: Sequence[int], low_scale: int, high_scale: int) -> list[int]:
    """Pack 32 2-bit indices little-endian within each byte, then both scales."""
    assert len(codes) == BLOCK_SIZE
    qs = [
        sum((codes[byte * 4 + i] & 3) << (2 * i) for i in range(4))
        for byte in range(QS_BYTES)
    ]
    return qs + [low_scale, high_scale]


def _make_weights(rows: int, blocks: int) -> torch.Tensor:
    packed_blocks: list[list[int]] = []
    scales = ((0x40, 0x48), (0x38, 0x44), (0x50, 0x3C))
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
    """Decode 4-consecutive-outputs-per-byte without plugin decoder helpers."""
    assert weights.dtype == torch.uint8
    assert weights.ndim == 2
    assert weights.shape[1] % BLOCK_BYTES == 0

    decoded_rows: list[list[float]] = []
    for packed_row in weights.cpu().tolist():
        decoded: list[float] = []
        for start in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[start : start + BLOCK_BYTES]
            low_scale, high_scale = map(_ue4m3_scale, block[QS_BYTES:BLOCK_BYTES])
            values = [
                CODEBOOK[(byte >> (2 * i)) & 3]
                for byte in block[:QS_BYTES]
                for i in range(4)
            ]
            decoded.extend(value * low_scale for value in values[:16])
            decoded.extend(value * high_scale for value in values[16:])
        decoded_rows.append(decoded)
    return torch.tensor(decoded_rows, dtype=torch.float32, device=weights.device)


def _reference_gemm(weights: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    return (activations.float() @ _reference_decode(weights).T).to(activations.dtype)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    activations: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Compare against the reference using bounds derived from the tensors.

    See tests/numerics.py. The bound comes from the activation dtype's mantissa
    width, the reduction length, and the measured cancellation between the
    activations and the decoded weights -- all read off the tensors under test
    rather than fitted to an observed failure.
    """
    assert_gemm_close(actual, expected, activations, _reference_decode(weights))


@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
@pytest.mark.parametrize(
    ("activation_shape", "rows"),
    [
        ((7, 96), 17),
        ((2, 3, 96), 19),
    ],
)
def test_type107_public_dense_gemm_matches_independent_abi_oracle(
    dtype: torch.dtype, activation_shape: tuple[int, ...], rows: int
) -> None:
    """Dense GEMM must preserve rank, tail rows, and per-half 2-bit scaling."""
    weights = _make_weights(rows, blocks=3)
    numel = 1
    for size in activation_shape:
        numel *= size
    activations = torch.arange(numel, dtype=dtype).reshape(activation_shape).div(31)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, rows)

    assert output.shape == (*activation_shape[:-1], rows)
    assert output.dtype == dtype
    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


def test_type107_public_dense_gemm_decodes_four_outputs_per_qs_byte() -> None:
    """An identity activation exposes the packed index order of each qs byte."""
    codes = [index % len(CODEBOOK) for index in range(BLOCK_SIZE)]
    weights = torch.tensor([_pack_block(codes, 0x40, 0x40)], dtype=torch.uint8).reshape(
        1, BLOCK_BYTES
    )
    activations = torch.eye(BLOCK_SIZE, dtype=torch.float32)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, 1)

    assert output.shape == (BLOCK_SIZE, 1)
    torch.testing.assert_close(
        output[:, 0], _reference_decode(weights)[0], atol=2e-5, rtol=2e-6
    )


@pytest.mark.parametrize(
    ("low_scale", "high_scale"),
    [(0x7F, 0x40), (0x40, 0xFF), (0xFF, 0x7F)],
)
def test_type107_public_dense_gemm_zeroes_reserved_scale_half(
    low_scale: int, high_scale: int
) -> None:
    """Reserved scale bytes above 0x7E zero only their own 16-weight half."""
    codes = [(index + 1) % len(CODEBOOK) for index in range(BLOCK_SIZE)]
    weights = torch.tensor(
        [_pack_block(codes, low_scale, high_scale)], dtype=torch.uint8
    ).reshape(1, BLOCK_BYTES)
    activations = torch.eye(BLOCK_SIZE, dtype=torch.float32)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, 1)

    expected = _reference_decode(weights)[0]
    if low_scale > 0x7E:
        assert torch.all(expected[:16] == 0)
        assert torch.all(output[:16, 0] == 0)
    if high_scale > 0x7E:
        assert torch.all(expected[16:] == 0)
        assert torch.all(output[16:, 0] == 0)
    torch.testing.assert_close(output[:, 0], expected, atol=2e-5, rtol=2e-6)


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
def test_type107_public_dense_gemm_rejects_empty_dimensions(
    weights: torch.Tensor, activations: torch.Tensor, row: int
) -> None:
    """Zero rows or zero packed width must fail instead of returning empties."""
    with pytest.raises(ValueError, match="positive"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, row)


def test_type107_public_dense_gemm_rejects_non_uint8_weights() -> None:
    """Packed type-107 blocks are raw bytes; other weight dtypes are invalid."""
    weights = _make_weights(rows=2, blocks=1).to(torch.int8)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    with pytest.raises(TypeError, match="uint8"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, 2)


def test_type107_public_dense_gemm_rejects_hidden_size_mismatch() -> None:
    """Hidden size is packed width // 10 * 32 and must match the activations."""
    weights = _make_weights(rows=2, blocks=2)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="hidden size"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, 2)


def test_type107_public_dense_gemm_rejects_rank1_activations() -> None:
    """Only rank-2 and rank-3 activations are part of the dense GEMM contract."""
    weights = _make_weights(rows=2, blocks=1)
    activations = torch.ones(BLOCK_SIZE, dtype=torch.float32)

    with pytest.raises(ValueError, match="2D or 3D"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, 2)


def test_type107_public_dense_gemm_rejects_row_mismatch() -> None:
    """``row`` must agree with the packed weight row count."""
    weights = _make_weights(rows=2, blocks=1)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="row"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q2_0_ROCMFPX, 3)
