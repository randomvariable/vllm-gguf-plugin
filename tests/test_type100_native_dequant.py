"""Public CUDA/HIP tests for native GGML type-100 dequantization."""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin import ops

QK = 32
BLOCK_BYTES = 18
TYPE100 = 100
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)


def _ue4m3(value: int) -> float:
    if value > 0x7E:
        return 0.0
    exponent, mantissa = (value >> 3) & 0x0F, value & 0x07
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _oracle(weight: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """Decode each type-100 low/high nibble as a contiguous 16-value half."""
    rows: list[list[float]] = []
    for row in weight.cpu().contiguous().reshape(m, -1).tolist():
        decoded: list[float] = []
        for offset in range(0, len(row), BLOCK_BYTES):
            block = row[offset : offset + BLOCK_BYTES]
            low, high = _ue4m3(block[16]), _ue4m3(block[17])
            decoded.extend(CODEBOOK[b & 0x0F] * low for b in block[:16])
            decoded.extend(CODEBOOK[b >> 4] * high for b in block[:16])
        rows.append(decoded)
    return torch.tensor(rows, dtype=torch.float32, device=weight.device)


def _require_native() -> None:
    if not torch.cuda.is_available() or not ops._cuda_kernel_available(
        "ggml_dequantize", TYPE100
    ):
        pytest.skip("native CUDA/HIP GGML type-100 kernel unavailable")


@pytest.fixture
def packed() -> torch.Tensor:
    _require_native()
    generator = torch.Generator().manual_seed(100)
    qs = torch.randint(0, 256, (2, 2, 16), generator=generator, dtype=torch.uint8)
    scales = torch.tensor([0x40, 0x4B], dtype=torch.uint8).expand(2, 2, 2)
    return torch.cat((qs, scales), dim=-1).reshape(2, 36).cuda()


@pytest.mark.parametrize("byte_count", [BLOCK_BYTES - 1, BLOCK_BYTES + 1])
def test_type100_native_rejects_wrong_byte_count(byte_count: int) -> None:
    _require_native()
    with pytest.raises((RuntimeError, ValueError), match="exactly"):
        ops.ggml_dequantize(
            torch.zeros(byte_count, dtype=torch.uint8, device="cuda"),
            TYPE100,
            1,
            QK,
            None,
        )


def test_type100_native_rejects_noncontiguous(packed: torch.Tensor) -> None:
    noncontiguous = torch.zeros((2, 72), dtype=torch.uint8, device="cuda")[:, ::2]
    noncontiguous.copy_(packed)
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.ggml_dequantize(noncontiguous, TYPE100, 2, 64, None)


def test_type100_native_rejects_invalid_dtype(packed: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="uint8"):
        ops.ggml_dequantize(packed.view(torch.int8), TYPE100, 2, 64, None)


@pytest.mark.parametrize("shape", [(1, 31), (1, 33), (0, 32), (1, 0)])
def test_type100_native_rejects_malformed_dimensions(
    packed: torch.Tensor, shape: tuple[int, int]
) -> None:
    with pytest.raises(RuntimeError, match="(positive|divisible)"):
        ops.ggml_dequantize(packed[:1, :18], TYPE100, *shape, None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_type100_native_matches_independent_oracle(
    packed: torch.Tensor, dtype: torch.dtype
) -> None:
    decoded = ops.ggml_dequantize(packed, TYPE100, 2, 64, dtype)
    torch.testing.assert_close(decoded.float(), _oracle(packed, 2, 64), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("low_scale", "high_scale"),
    [(0x7F, 0x48), (0x40, 0x7F), (0x80, 0x48), (0x40, 0xFE)],
)
def test_type100_native_reserved_scales_are_half_local(
    low_scale: int, high_scale: int
):
    _require_native()
    packed = torch.tensor(
        list(bytes([0x21] * 16) + bytes([low_scale, high_scale])),
        dtype=torch.uint8,
        device="cuda",
    )
    decoded = ops.ggml_dequantize(packed, TYPE100, 1, QK, torch.float32)
    expected = torch.tensor(
        [[1.0] * 16 + [4.0] * 16], dtype=torch.float32, device="cuda"
    )
    if low_scale > 0x7E:
        expected[:, :16] = 0.0
    if high_scale > 0x7E:
        expected[:, 16:] = 0.0
    torch.testing.assert_close(decoded, expected)


def test_type100_native_decodes_contiguous_nibble_halves() -> None:
    _require_native()
    packed = torch.tensor(
        list(bytes([0x21] * 16) + bytes([0x40, 0x48])),
        dtype=torch.uint8,
        device="cuda",
    )

    decoded = ops.ggml_dequantize(packed, TYPE100, 1, QK, torch.float32)

    expected = torch.tensor([[1.0] * 16 + [4.0] * 16], device="cuda")
    torch.testing.assert_close(decoded, expected)
