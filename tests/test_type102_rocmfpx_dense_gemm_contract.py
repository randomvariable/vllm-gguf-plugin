# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RED contract for GGML type-102 Q6_0_ROCMFPX dense GEMM.

The oracle in this module is written directly from the ``block_rocmfp6``
ABI (24 packed 6-bit sign-magnitude codes + two UE4M3 half-scales in
26 bytes) and deliberately shares no code with any plugin decoder, so a
shared wrong assumption cannot mask a kernel error.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q6_0_ROCMFPX
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

BLOCK_BYTES = 26
BLOCK_SIZE = 32
CODE_BYTES = 24
GROUP_BYTES = 3
GROUP_CODES = 4
HALF = 16
ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Codes whose c1/c2 lanes straddle a byte boundary: c1 owns src[0] bits 6-7
# plus src[1] bits 0-3, and c2 owns src[1] bits 4-7 plus src[2] bits 0-1.
# Every value below sets bits on both sides of its boundary.
CROSS_BYTE_C1 = (0x3F, 0x21, 0x33, 0x1E, 0x2B, 0x07, 0x39, 0x15)
CROSS_BYTE_C2 = (0x3F, 0x31, 0x2A, 0x13, 0x3C, 0x25, 0x1F, 0x1D)


def _ue4m3_scale(scale_byte: int) -> float:
    """Independent UE4M3 bias-8 scale oracle for type-102 ABI bytes."""
    if scale_byte > 0x7E:
        return 0.0
    exponent, mantissa = divmod(scale_byte, 8)
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _decode_sign_magnitude(code: int) -> int:
    """Decode one 6-bit sign-magnitude code; ``0x20`` means -32, not -0."""
    magnitude = code & 31
    if code & 32:
        return -(magnitude if magnitude != 0 else 32)
    return magnitude


def _pack_codes(codes: Sequence[int]) -> list[int]:
    """Invert the 4-codes-per-3-bytes packing for one 32-weight block."""
    assert len(codes) == BLOCK_SIZE
    packed: list[int] = []
    for group in range(BLOCK_SIZE // GROUP_CODES):
        c0, c1, c2, c3 = codes[group * GROUP_CODES : group * GROUP_CODES + GROUP_CODES]
        packed.append((c0 | ((c1 & 0x03) << 6)) & 0xFF)
        packed.append(((c1 >> 2) | ((c2 & 0x0F) << 4)) & 0xFF)
        packed.append(((c2 >> 4) | (c3 << 2)) & 0xFF)
    assert len(packed) == CODE_BYTES
    return packed


def _pack_block(codes: Sequence[int], low_scale: int, high_scale: int) -> list[int]:
    return [*_pack_codes(codes), low_scale, high_scale]


def _make_weights(rows: int, blocks: int) -> torch.Tensor:
    packed_blocks: list[list[int]] = []
    scales = ((0x40, 0x44), (0x7F, 0x40), (0x40, 0xFF), (0x3A, 0x41))
    for row in range(rows):
        for block in range(blocks):
            codes = [
                (row * 7 + block * 11 + index * 5) % 64 for index in range(BLOCK_SIZE)
            ]
            packed_blocks.append(_pack_block(codes, *scales[block % len(scales)]))
    return torch.tensor(packed_blocks, dtype=torch.uint8).reshape(
        rows, blocks * BLOCK_BYTES
    )


def _reference_decode(weights: torch.Tensor) -> torch.Tensor:
    """Re-extract 6-bit codes straight from the packed bytes per the ABI."""
    assert weights.dtype == torch.uint8
    assert weights.ndim == 2
    assert weights.shape[1] % BLOCK_BYTES == 0

    decoded_rows: list[list[float]] = []
    for packed_row in weights.cpu().tolist():
        decoded: list[float] = []
        for start in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[start : start + BLOCK_BYTES]
            low_scale, high_scale = (_ue4m3_scale(byte) for byte in block[24:26])
            values: list[int] = []
            for group in range(CODE_BYTES // GROUP_BYTES):
                src = block[group * GROUP_BYTES : group * GROUP_BYTES + GROUP_BYTES]
                values.append(src[0] & 0x3F)
                values.append(((src[0] >> 6) | (src[1] << 2)) & 0x3F)
                values.append(((src[1] >> 4) | (src[2] << 4)) & 0x3F)
                values.append((src[2] >> 2) & 0x3F)
            assert len(values) == BLOCK_SIZE
            decoded.extend(
                _decode_sign_magnitude(code) * low_scale for code in values[:HALF]
            )
            decoded.extend(
                _decode_sign_magnitude(code) * high_scale for code in values[HALF:]
            )
        decoded_rows.append(decoded)
    return torch.tensor(decoded_rows, dtype=torch.float32, device=weights.device)


def _reference_gemm(weights: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    return (activations.float() @ _reference_decode(weights).T).to(activations.dtype)


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-6)
    elif dtype == torch.float16:
        torch.testing.assert_close(actual, expected, atol=4.0, rtol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, atol=32.0, rtol=8e-3)


def _decode_via_public_gemm(weights: torch.Tensor) -> torch.Tensor:
    """Read one packed block back out through the public GEMM entry point."""
    identity = torch.eye(BLOCK_SIZE, dtype=torch.float32)
    output = ggml_mul_mat_a8_triton(
        weights, identity, GGML_TYPE_Q6_0_ROCMFPX, weights.shape[0]
    )
    return output.reshape(BLOCK_SIZE, weights.shape[0])[:, 0]


@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
@pytest.mark.parametrize(
    ("activation_shape", "rows"),
    [
        ((7, 96), 17),
        ((2, 3, 96), 19),
    ],
)
def test_type102_public_dense_gemm_matches_independent_abi_oracle(
    dtype: torch.dtype, activation_shape: tuple[int, ...], rows: int
) -> None:
    """Dense GEMM must preserve rank, tail rows, and both UE4M3 half-scales."""
    weights = _make_weights(rows, blocks=3)
    numel = 1
    for size in activation_shape:
        numel *= size
    activations = torch.arange(numel, dtype=dtype).reshape(activation_shape).div(numel)

    output = ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q6_0_ROCMFPX, rows)

    assert output.shape == (*activation_shape[:-1], rows)
    assert output.dtype == dtype
    _assert_close(output, _reference_gemm(weights, activations), dtype)


def test_type102_public_dense_gemm_decodes_cross_byte_code_boundaries() -> None:
    """Codes c1 and c2 straddle two packed bytes and must be reassembled."""
    codes: list[int] = []
    for group in range(BLOCK_SIZE // GROUP_CODES):
        codes.extend(
            (
                (group * 9) % 64,
                CROSS_BYTE_C1[group],
                CROSS_BYTE_C2[group],
                (group * 13 + 3) % 64,
            )
        )
    # Guard the fixture itself: every c1/c2 really does span a byte boundary.
    for group in range(BLOCK_SIZE // GROUP_CODES):
        assert CROSS_BYTE_C1[group] & 0x03 and CROSS_BYTE_C1[group] >> 2
        assert CROSS_BYTE_C2[group] & 0x0F and CROSS_BYTE_C2[group] >> 4

    weights = torch.tensor([_pack_block(codes, 0x40, 0x44)], dtype=torch.uint8)

    decoded = _decode_via_public_gemm(weights)

    torch.testing.assert_close(
        decoded, _reference_decode(weights).reshape(BLOCK_SIZE), atol=1e-4, rtol=1e-6
    )


def test_type102_public_dense_gemm_decodes_zero_magnitude_sign_bit_as_minus_32() -> (
    None
):
    """``0x20`` is magnitude 0 with the sign bit set and decodes to -32."""
    codes = [0x20] * BLOCK_SIZE
    weights = torch.tensor([_pack_block(codes, 0x40, 0x40)], dtype=torch.uint8)
    assert all(_decode_sign_magnitude(code) == -32 for code in codes)

    decoded = _decode_via_public_gemm(weights)

    torch.testing.assert_close(
        decoded,
        torch.full((BLOCK_SIZE,), -32.0),
        atol=1e-4,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("low_scale_byte", "high_scale_byte", "zero_low", "zero_high"),
    [
        (0x7F, 0x40, True, False),
        (0x40, 0xFF, False, True),
        (0xFF, 0x7F, True, True),
    ],
)
def test_type102_public_dense_gemm_zeroes_reserved_scale_halves(
    low_scale_byte: int, high_scale_byte: int, zero_low: bool, zero_high: bool
) -> None:
    """Reserved scale bytes above ``0x7E`` zero only their own 16-weight half."""
    codes = [(index * 3 + 1) % 64 for index in range(BLOCK_SIZE)]
    weights = torch.tensor(
        [_pack_block(codes, low_scale_byte, high_scale_byte)], dtype=torch.uint8
    )

    decoded = _decode_via_public_gemm(weights)

    assert bool(decoded[:HALF].eq(0).all()) == zero_low
    assert bool(decoded[HALF:].eq(0).all()) == zero_high
    torch.testing.assert_close(
        decoded, _reference_decode(weights).reshape(BLOCK_SIZE), atol=1e-4, rtol=1e-6
    )


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
def test_type102_public_dense_gemm_rejects_empty_dimensions(
    weights: torch.Tensor, activations: torch.Tensor, row: int
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q6_0_ROCMFPX, row)


def test_type102_public_dense_gemm_rejects_non_uint8_weights() -> None:
    weights = _make_weights(rows=2, blocks=1).to(torch.int8)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    with pytest.raises(TypeError, match="uint8"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q6_0_ROCMFPX, 2)


def test_type102_public_dense_gemm_rejects_hidden_size_mismatch() -> None:
    weights = _make_weights(rows=2, blocks=2)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="hidden size"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q6_0_ROCMFPX, 2)


def test_type102_public_dense_gemm_rejects_unaligned_packed_width() -> None:
    weights = torch.zeros((2, BLOCK_BYTES + 1), dtype=torch.uint8)
    activations = torch.ones((3, BLOCK_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="divisible"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q6_0_ROCMFPX, 2)


def test_type102_public_dense_gemm_rejects_rank1_activations() -> None:
    weights = _make_weights(rows=2, blocks=1)
    activations = torch.ones(BLOCK_SIZE, dtype=torch.float32)

    with pytest.raises(ValueError, match="2D or 3D"):
        ggml_mul_mat_a8_triton(weights, activations, GGML_TYPE_Q6_0_ROCMFPX, 2)
