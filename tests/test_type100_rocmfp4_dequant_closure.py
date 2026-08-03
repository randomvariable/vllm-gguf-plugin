"""RED contracts for GGML type-100 Q4_0_ROCMFP4 dequant-only fallback."""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import linear
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton

QK = 32
BLOCK_BYTES = 18
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)


def _ue4m3(byte: int) -> float:
    """Independent UE4M3 bias-8 scale oracle."""
    if byte > 0x7E:
        return 0.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 8))


def _block(qs: bytes, low_scale: int, high_scale: int) -> bytes:
    assert len(qs) == 16
    return qs + bytes((low_scale, high_scale))


def _reference(qweight: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    assert columns % QK == 0
    raw = qweight.contiguous().reshape(rows, -1).tolist()
    decoded_rows: list[list[float]] = []
    for packed_row in raw:
        decoded: list[float] = []
        for offset in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[offset : offset + BLOCK_BYTES]
            low_scale, high_scale = map(_ue4m3, block[16:18])
            decoded.extend(CODEBOOK[byte & 0x0F] * low_scale for byte in block[:16])
            decoded.extend(CODEBOOK[byte >> 4] * high_scale for byte in block[:16])
        decoded_rows.append(decoded)
    return torch.tensor(decoded_rows, dtype=torch.float32)


def _random_qweight(rows: int = 2, blocks_per_row: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(100)
    blocks: list[torch.Tensor] = []
    for _ in range(rows * blocks_per_row):
        qs = torch.randint(0, 256, (16,), generator=generator, dtype=torch.uint8)
        scales = torch.tensor((0x40, 0x4B), dtype=torch.uint8)
        blocks.append(torch.cat((qs, scales)))
    return torch.stack(blocks).reshape(rows, blocks_per_row * BLOCK_BYTES)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_type100_random_blocks_match_independent_contiguous_half_scale_oracle(
    dtype: torch.dtype,
) -> None:
    qweight = _random_qweight()
    expected = _reference(qweight, rows=2, columns=64).to(dtype)

    decoded = ggml_dequantize_triton(
        qweight, GGML_TYPE_Q4_0_ROCMFP4, 2, 64, dtype=dtype
    )

    assert decoded.dtype is dtype
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_scale", [0x7F, 0xFF])
def test_type100_invalid_half_scale_zeroes_only_its_contiguous_half(
    invalid_scale: int,
) -> None:
    qweight = torch.tensor(
        list(_block(bytes([0x21] * 16), invalid_scale, 0x48)), dtype=torch.uint8
    )

    decoded = ggml_dequantize_triton(
        qweight, GGML_TYPE_Q4_0_ROCMFP4, 1, QK, dtype=torch.float32
    )

    torch.testing.assert_close(decoded, _reference(qweight, 1, QK))


def test_type100_decodes_contiguous_nibble_halves() -> None:
    qweight = torch.tensor(
        list(_block(bytes([0x21] * 16), 0x40, 0x48)), dtype=torch.uint8
    )

    decoded = ggml_dequantize_triton(
        qweight, GGML_TYPE_Q4_0_ROCMFP4, 1, QK, dtype=torch.float32
    )

    torch.testing.assert_close(decoded, torch.tensor([[1.0] * 16 + [4.0] * 16]))


@pytest.mark.parametrize("byte_count", [BLOCK_BYTES - 1, BLOCK_BYTES + 1])
def test_type100_requires_exact_packed_byte_count(byte_count: int) -> None:
    with pytest.raises(ValueError, match="requires 18 bytes"):
        ggml_dequantize_triton(
            torch.zeros(byte_count, dtype=torch.uint8),
            GGML_TYPE_Q4_0_ROCMFP4,
            1,
            QK,
        )


@pytest.mark.parametrize("shape", [(1, 31), (1, 33), (0, QK), (1, 0)])
def test_type100_rejects_malformed_dequant_shape(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        ggml_dequantize_triton(
            torch.zeros(BLOCK_BYTES, dtype=torch.uint8),
            GGML_TYPE_Q4_0_ROCMFP4,
            *shape,
        )


def test_type100_rejects_non_uint8_packed_weights() -> None:
    with pytest.raises(TypeError, match="torch.uint8"):
        ggml_dequantize_triton(
            torch.zeros(BLOCK_BYTES, dtype=torch.int8),
            GGML_TYPE_Q4_0_ROCMFP4,
            1,
            QK,
        )


def test_type100_noncontiguous_input_matches_contiguous_oracle() -> None:
    qweight = _random_qweight(rows=1, blocks_per_row=2)
    backing = torch.zeros((1, qweight.shape[1] * 2), dtype=torch.uint8)
    backing[:, ::2] = qweight
    noncontiguous = backing[:, ::2]
    assert not noncontiguous.is_contiguous()

    decoded = ggml_dequantize_triton(
        noncontiguous, GGML_TYPE_Q4_0_ROCMFP4, 1, 64, dtype=torch.float32
    )

    torch.testing.assert_close(decoded, _reference(qweight, 1, 64))


@pytest.mark.parametrize("activation_shape", [(2, 64), (2, 3, 64)])
def test_type100_public_dense_fallback_preserves_shape_and_never_selects_cpu_gemv(
    monkeypatch: pytest.MonkeyPatch, activation_shape: tuple[int, ...]
) -> None:
    qweight = _random_qweight()
    x = torch.arange(
        int(torch.tensor(activation_shape).prod()), dtype=torch.float32
    ).reshape(activation_shape)

    def unexpected_fused_op(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("type-100 CPU dequant-only fallback selected fused GEMV/GEMM")

    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", unexpected_fused_op)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", unexpected_fused_op)

    output = linear._fused_mul_mat_gguf(x, qweight, GGML_TYPE_Q4_0_ROCMFP4)

    assert GGML_TYPE_Q4_0_ROCMFP4 not in ops._CUDA_GEMV_QUANT_TYPES
    assert output.shape == (*activation_shape[:-1], qweight.shape[0])
    torch.testing.assert_close(output, x @ _reference(qweight, 2, 64).T)
