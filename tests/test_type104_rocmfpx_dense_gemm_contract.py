"""RED contract for GGML type-104 Q3_0_ROCMFPX dense GEMM.

The oracle in this module is written directly from the authoritative
``block_rocmfp3`` ABI (``vllm_gguf_plugin/csrc/gguf/ggml-common.h``) and is
deliberately independent of every plugin decoder helper, so a shared bug
cannot mask a kernel error.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from tests.numerics import assert_gemm_close
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q3_0_ROCMFPX
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

BLOCK_BYTES = 14
BLOCK_SIZE = 32
CODE_BYTES = 12
GROUP_CODES = 8
CODEBOOK = (0, 1, 2, 4, 0, -1, -2, -4)
ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _ue4m3_scale(scale_byte: int) -> float:
    """Independent UE4M3 bias-8 scale oracle for type-104 ABI bytes."""
    if scale_byte > 0x7E:
        return 0.0
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _pack_group(codes: Sequence[int]) -> list[int]:
    """Pack 8 three-bit codes into 3 bytes, inverting the ABI extraction."""
    assert len(codes) == GROUP_CODES
    c0, c1, c2, c3, c4, c5, c6, c7 = (int(code) & 7 for code in codes)
    return [
        (c0 | (c1 << 3) | ((c2 & 3) << 6)) & 0xFF,
        ((c2 >> 2) | (c3 << 1) | (c4 << 4) | ((c5 & 1) << 7)) & 0xFF,
        ((c5 >> 1) | (c6 << 2) | (c7 << 5)) & 0xFF,
    ]


def _pack_block(codes: Sequence[int], low_scale: int, high_scale: int) -> list[int]:
    assert len(codes) == BLOCK_SIZE
    packed: list[int] = []
    for group in range(4):
        packed += _pack_group(codes[group * GROUP_CODES : (group + 1) * GROUP_CODES])
    assert len(packed) == CODE_BYTES
    return packed + [low_scale, high_scale]


def _extract_codes(block: Sequence[int]) -> list[int]:
    """Extract 32 codes exactly as the ABI specifies, group by group."""
    codes: list[int] = []
    for group in range(4):
        src = block[group * 3 : group * 3 + 3]
        codes += [
            src[0] & 7,
            (src[0] >> 3) & 7,
            ((src[0] >> 6) | (src[1] << 2)) & 7,
            (src[1] >> 1) & 7,
            (src[1] >> 4) & 7,
            ((src[1] >> 7) | (src[2] << 1)) & 7,
            (src[2] >> 2) & 7,
            (src[2] >> 5) & 7,
        ]
    return codes


def _make_weights(rows: int, blocks: int) -> torch.Tensor:
    packed_blocks: list[list[int]] = []
    scales = ((0x40, 0x48), (0x7F, 0x40), (0x40, 0xFF))
    for row in range(rows):
        for block in range(blocks):
            codes = [
                (row * 3 + block * 5 + index) % len(CODEBOOK)
                for index in range(BLOCK_SIZE)
            ]
            packed_blocks.append(
                _pack_block(codes, *scales[block % len(scales)]),
            )
    return torch.tensor(packed_blocks, dtype=torch.uint8).reshape(
        rows, blocks * BLOCK_BYTES
    )


def _reference_decode(weights: torch.Tensor) -> torch.Tensor:
    """Decode 3-bit codes with repeat_interleave(16) half-scales."""
    assert weights.dtype == torch.uint8
    assert weights.ndim == 2
    assert weights.shape[1] % BLOCK_BYTES == 0

    decoded_rows: list[list[float]] = []
    for packed_row in weights.cpu().tolist():
        decoded: list[float] = []
        for start in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[start : start + BLOCK_BYTES]
            low_scale, high_scale = map(_ue4m3_scale, block[CODE_BYTES:BLOCK_BYTES])
            codes = _extract_codes(block[:CODE_BYTES])
            decoded += [CODEBOOK[code] * low_scale for code in codes[:16]]
            decoded += [CODEBOOK[code] * high_scale for code in codes[16:]]
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


def test_type104_packing_oracle_round_trips_every_code() -> None:
    """The independent packer must invert the ABI extraction for all codes."""
    for offset in range(8):
        codes = [(offset + index) % 8 for index in range(BLOCK_SIZE)]
        block = _pack_block(codes, 0x40, 0x40)
        assert len(block) == BLOCK_BYTES
        assert _extract_codes(block[:CODE_BYTES]) == codes


@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
@pytest.mark.parametrize(
    ("activation_shape", "rows"),
    [
        ((7, 96), 17),
        ((2, 3, 96), 19),
    ],
)
def test_type104_public_dense_gemm_matches_independent_abi_oracle(
    dtype: torch.dtype, activation_shape: tuple[int, ...], rows: int
) -> None:
    """Dense GEMM must preserve rank, tail rows, and per-half 3-bit scales."""
    weights = _make_weights(rows, blocks=3)
    numel = 1
    for size in activation_shape:
        numel *= size
    activations = torch.arange(numel, dtype=dtype).reshape(activation_shape).div(31)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, rows)

    assert output.shape == (*activation_shape[:-1], rows)
    assert output.dtype == dtype
    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
def test_type104_public_dense_gemm_handles_cross_byte_code_boundaries(
    dtype: torch.dtype,
) -> None:
    """Codes c2 and c5 span two packed bytes and must survive the split."""
    rows = 17
    packed_blocks: list[list[int]] = []
    for row in range(rows):
        # Drive c2/c5 through every value, including the ones whose high bits
        # live in the following byte (c2 >= 4, c5 >= 2).
        codes = [0] * BLOCK_SIZE
        for group in range(4):
            base = group * GROUP_CODES
            codes[base + 2] = (row + group) % 8
            codes[base + 5] = (row + group + 3) % 8
            codes[base + 0] = 7
            codes[base + 1] = 7
            codes[base + 3] = 7
            codes[base + 4] = 7
            codes[base + 6] = 7
            codes[base + 7] = 7
        packed_blocks.append(_pack_block(codes, 0x40, 0x48))
    weights = torch.tensor(packed_blocks, dtype=torch.uint8).reshape(rows, BLOCK_BYTES)
    activations = torch.arange(5 * BLOCK_SIZE, dtype=dtype).reshape(5, BLOCK_SIZE)
    activations = activations.div(17)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, rows)

    assert output.shape == (5, rows)
    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


@pytest.mark.parametrize(
    ("low_scale", "high_scale"),
    [
        (0x7F, 0x40),
        (0x40, 0xFF),
        (0xFF, 0x7F),
    ],
)
def test_type104_public_dense_gemm_zeroes_reserved_scale_per_half(
    low_scale: int, high_scale: int
) -> None:
    """Reserved scale bytes above 0x7E zero only their own 16-weight half."""
    rows = 17
    codes = [index % 8 for index in range(BLOCK_SIZE)]
    weights = torch.tensor(
        [_pack_block(codes, low_scale, high_scale) for _ in range(rows)],
        dtype=torch.uint8,
    ).reshape(rows, BLOCK_BYTES)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, rows)

    decoded = _reference_decode(weights)
    if low_scale > 0x7E:
        assert torch.all(decoded[:, :16] == 0.0)
    if high_scale > 0x7E:
        assert torch.all(decoded[:, 16:] == 0.0)
    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


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
def test_type104_public_dense_gemm_rejects_empty_dimensions(
    weights: torch.Tensor, activations: torch.Tensor, row: int
) -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, row)


def test_type104_public_dense_gemm_rejects_non_uint8_weights() -> None:
    weights = _make_weights(17, blocks=1).to(torch.int8)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)
    with pytest.raises(TypeError, match="uint8"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, 17)


def test_type104_public_dense_gemm_rejects_hidden_size_mismatch() -> None:
    weights = _make_weights(17, blocks=3)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="hidden size"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, 17)


def test_type104_public_dense_gemm_rejects_rank1_activations() -> None:
    weights = _make_weights(17, blocks=1)
    activations = torch.ones(BLOCK_SIZE, dtype=torch.float32)
    with pytest.raises(ValueError, match="2D or 3D"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, 17)


def test_type104_public_dense_gemm_rejects_unaligned_row_width() -> None:
    weights = torch.zeros((17, BLOCK_BYTES + 1), dtype=torch.uint8)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q3_0_ROCMFPX, 17)
