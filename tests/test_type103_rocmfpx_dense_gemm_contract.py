"""Dense GEMM contract for GGML type 103 (Q8_0_ROCMFPX).

The oracle here is written directly from the ROCmFPX ABI: 32 signed int8
codes followed by one UE4M3 scale byte, 33 bytes per block. Reserved scale
bytes ``0x7F..0xFF`` decode to zero.
"""

from collections.abc import Sequence

import pytest
import torch

from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

QUANT_TYPE = 103
BLOCK_BYTES = 33
BLOCK_SIZE = 32


def _oracle_scale(byte: int) -> float:
    if byte > 0x7E:
        return 0.0
    exponent, mantissa = byte >> 3, byte & 0x07
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _pack_block(codes: Sequence[int], scale_byte: int) -> list[int]:
    assert len(codes) == BLOCK_SIZE
    return [code & 0xFF for code in codes] + [scale_byte]


def _oracle_dense(packed: torch.Tensor) -> torch.Tensor:
    rows = []
    for row in packed.tolist():
        values: list[float] = []
        for start in range(0, len(row), BLOCK_BYTES):
            block = row[start : start + BLOCK_BYTES]
            scale = _oracle_scale(block[32])
            for byte in block[:32]:
                signed = byte if byte < 128 else byte - 256
                values.append(signed * scale)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def _weights(rows: int, blocks: int) -> torch.Tensor:
    packed: list[int] = []
    for row in range(rows):
        for block in range(blocks):
            codes = [((row * 7 + block * 3 + i) % 256) - 128 for i in range(BLOCK_SIZE)]
            packed.extend(_pack_block(codes, 0x40 + block))
    return torch.tensor(packed, dtype=torch.uint8).reshape(rows, blocks * BLOCK_BYTES)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation_shape", [(7, 96), (2, 3, 96)])
@pytest.mark.parametrize("rows", [17, 19])
def test_type103_dense_gemm_matches_independent_abi_oracle(
    dtype: torch.dtype, activation_shape: tuple[int, ...], rows: int
) -> None:
    """Signed int8 decode with one UE4M3 scale per 32-value block."""
    weights = _weights(rows=rows, blocks=3)
    activations = (
        torch.arange(
            int(torch.tensor(activation_shape).prod()), dtype=torch.float32
        ).reshape(activation_shape)
        / 512
    ).to(dtype)

    output = ggml_mul_mat_a8_triton(weights, activations, QUANT_TYPE, rows)

    dense = _oracle_dense(weights)
    expected = (activations.float() @ dense.T).to(dtype)

    assert output.shape == (*activation_shape[:-1], rows)
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, atol=1.0, rtol=8e-3)


def test_type103_dense_gemm_zeroes_reserved_scale_bytes() -> None:
    """Reserved UE4M3 bytes decode to a zero scale for the whole block."""
    codes = [1] * BLOCK_SIZE
    packed = torch.tensor(
        [_pack_block(codes, 0x40) + _pack_block(codes, 0x7F)], dtype=torch.uint8
    )
    activations = torch.ones((1, 64), dtype=torch.float32)

    output = ggml_mul_mat_a8_triton(packed, activations, QUANT_TYPE, 1)

    # Only the first block contributes: 32 codes of value 1 at scale 1.0.
    torch.testing.assert_close(output, torch.tensor([[32.0]]))


def test_type103_dense_gemm_decodes_negative_codes() -> None:
    """Bytes >= 128 are two's-complement negative int8 codes."""
    codes = [-1] * 16 + [127] * 16
    packed = torch.tensor([_pack_block(codes, 0x40)], dtype=torch.uint8)
    activations = torch.ones((1, BLOCK_SIZE), dtype=torch.float32)

    output = ggml_mul_mat_a8_triton(packed, activations, QUANT_TYPE, 1)

    torch.testing.assert_close(output, torch.tensor([[16 * -1.0 + 16 * 127.0]]))


@pytest.mark.parametrize(
    "weights,activations,error,message",
    [
        (
            torch.zeros((0, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, BLOCK_SIZE), dtype=torch.float32),
            ValueError,
            "positive",
        ),
        (
            torch.zeros((1, 0), dtype=torch.uint8),
            torch.zeros((1, BLOCK_SIZE), dtype=torch.float32),
            ValueError,
            "positive",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.float32),
            torch.zeros((1, BLOCK_SIZE), dtype=torch.float32),
            TypeError,
            "uint8",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((BLOCK_SIZE,), dtype=torch.float32),
            ValueError,
            "2D or 3D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, BLOCK_SIZE - 1), dtype=torch.float32),
            ValueError,
            "hidden size",
        ),
        (
            torch.zeros((1, BLOCK_BYTES - 1), dtype=torch.uint8),
            torch.zeros((1, BLOCK_SIZE), dtype=torch.float32),
            ValueError,
            "divisible by 33",
        ),
    ],
)
def test_type103_dense_gemm_rejects_malformed_inputs(
    weights: torch.Tensor,
    activations: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed type-103 storage fails closed before any decode."""
    with pytest.raises(error, match=message):
        ggml_mul_mat_a8_triton(
            weights, activations, QUANT_TYPE, max(weights.shape[0], 1)
        )
