"""Coverage for a dedicated type-101 ROCmFP4_FAST Triton GEMM."""

import importlib
from collections.abc import Sequence
from math import prod

import pytest
import torch

from tests.numerics import assert_gemm_close
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4_FAST

BLOCK_BYTES = 17
BLOCK_SIZE = 32
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)
ACTIVATION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _ue4m3_scale(scale_byte: int) -> float:
    """Independent UE4M3 bias-8 decoder; invalid encodings decode to zero."""
    if not 0 <= scale_byte <= 0x7E:
        return 0.0
    exponent, mantissa = (scale_byte >> 3) & 0x0F, scale_byte & 0x07
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8)


def _pack_block(codes: Sequence[int], scale_byte: int) -> list[int]:
    assert len(codes) == BLOCK_SIZE
    return [codes[index] | (codes[index + 16] << 4) for index in range(16)] + [
        scale_byte
    ]


def _make_weights(rows: int, blocks: int, scale_byte: int = 0x40) -> torch.Tensor:
    packed_blocks = []
    for row in range(rows):
        for block in range(blocks):
            codes = [
                (row + block + index) % len(CODEBOOK) for index in range(BLOCK_SIZE)
            ]
            packed_blocks.extend(_pack_block(codes, scale_byte))
    return torch.tensor(packed_blocks, dtype=torch.uint8).reshape(
        rows, blocks * BLOCK_BYTES
    )


def _reference_decode(weights: torch.Tensor) -> torch.Tensor:
    """Decode type-101 bytes without importing any plugin decoder."""
    assert weights.dtype == torch.uint8
    assert weights.dim() == 2
    assert weights.shape[1] % BLOCK_BYTES == 0

    decoded_rows = []
    for packed_row in weights.cpu().tolist():
        decoded = []
        for start in range(0, len(packed_row), BLOCK_BYTES):
            block = packed_row[start : start + BLOCK_BYTES]
            scale = _ue4m3_scale(block[16])
            # ABI order: all low nibbles first, then all high nibbles.
            decoded.extend(CODEBOOK[value & 0x0F] * scale for value in block[:16])
            decoded.extend(CODEBOOK[value >> 4] * scale for value in block[:16])
        decoded_rows.append(decoded)
    return torch.tensor(decoded_rows, dtype=torch.float32, device=weights.device)


def _reference_gemm(weights: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    result = activations.float() @ _reference_decode(weights).T
    return result.to(activations.dtype)


def _kernel():
    """Return dedicated public type-101 Triton GEMM kernel."""
    module = importlib.import_module(
        "vllm_gguf_plugin.triton.gemm.standard_quant.q4_0_rocmfp4_fast"
    )
    return module.ggml_gemm_q4_0_rocmfp4_fast_triton


def _cuda_or_rocm() -> bool:
    return torch.cuda.is_available()


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


@pytest.mark.parametrize(
    ("scale_byte", "expected"),
    [
        (0x00, 0.0),
        (0x01, 1.0 / 1024.0),
        (0x07, 7.0 / 1024.0),
        (0x08, 2.0**-7),
        (0x7D, 208.0),
        (0x7E, 224.0),
        (0x7F, 0.0),
        (0xFF, 0.0),
    ],
)
def test_type_101_oracle_covers_ue4m3_boundaries(
    scale_byte: int, expected: float
) -> None:
    assert _ue4m3_scale(scale_byte) == expected


def test_type_101_gemm_is_dedicated_and_dispatchable() -> None:
    """Type 101 reaches until type 101 reaches dedicated Triton GEMM dispatch."""
    from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

    weights = _make_weights(rows=1, blocks=1)
    activations = torch.ones((1, BLOCK_SIZE), dtype=torch.float32)
    output = ggml_mul_mat_a8_triton(
        weights, activations, GGML_TYPE_Q4_0_ROCMFP4_FAST, row=1
    )
    torch.testing.assert_close(output, _reference_gemm(weights, activations))


@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
def test_type_101_gemm_scales_both_nibble_halves() -> None:
    """A row/block with identical nibbles must scale all 32 weights."""
    weights = torch.tensor(
        [0x11] * 16 + [0x40], dtype=torch.uint8, device="cuda"
    ).reshape(1, -1)
    activations = torch.ones((1, BLOCK_SIZE), dtype=torch.float32, device="cuda")

    output = _kernel()(weights, activations, row=1)

    torch.testing.assert_close(output, _reference_gemm(weights, activations))


@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
@pytest.mark.parametrize("activation_shape,rows", [((33, 96), 129), ((2, 3, 96), 33)])
def test_type_101_gemm_matches_independent_oracle_for_tails_and_ranks(
    dtype: torch.dtype, activation_shape: tuple[int, ...], rows: int
) -> None:
    weights = _make_weights(rows, blocks=3, scale_byte=0x7E).cuda()
    activations = (
        torch.arange(prod(activation_shape), dtype=dtype)
        .reshape(activation_shape)
        .div(31)
        .cuda()
    )

    output = _kernel()(weights, activations, row=rows)

    assert output.shape == (*activation_shape[:-1], rows)
    assert output.dtype == dtype
    assert output.device == activations.device
    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
@pytest.mark.parametrize("scale_byte", [0x40, 0x7F, 0xFF])
def test_type_101_gemm_decodes_all_codes_low_then_high_and_invalid_scale(
    scale_byte: int,
) -> None:
    # Distinct low/high sequences prove nibble order with a nonzero scale.
    codes = list(range(16)) + list(reversed(range(16)))
    weights = (
        torch.tensor(_pack_block(codes, scale_byte), dtype=torch.uint8)
        .reshape(1, -1)
        .cuda()
    )
    activations = torch.eye(BLOCK_SIZE, dtype=torch.float32, device="cuda")

    output = _kernel()(weights, activations, row=1)

    torch.testing.assert_close(
        output, _reference_gemm(weights, activations), atol=0, rtol=0
    )


@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
def test_type_101_gemm_broadcasts_scales_across_n_boundary() -> None:
    """Scale bytes vary by output row and block across the BLOCK_N boundary."""
    rows = 17
    blocks = 2
    packed_blocks = []
    for row in range(rows):
        for block in range(blocks):
            # Code 1 makes each block's contribution easy to verify with ones.
            packed_blocks.extend(_pack_block([1] * BLOCK_SIZE, 0x20 + row * 2 + block))
    weights = (
        torch.tensor(packed_blocks, dtype=torch.uint8)
        .reshape(rows, blocks * BLOCK_BYTES)
        .cuda()
    )
    activations = torch.ones(
        (1, blocks * BLOCK_SIZE), dtype=torch.float32, device="cuda"
    )

    output = _kernel()(weights, activations, row=rows)

    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
@pytest.mark.parametrize("m", [16, 17, 64])
def test_type_101_public_gemm_dispatch_handles_large_m(monkeypatch, m: int) -> None:
    """Public dispatch reaches dedicated type-101 GEMM beyond one tile."""
    import vllm_gguf_plugin.triton.gemm.interface as interface

    calls: list[tuple[int, int]] = []
    original = interface.ggml_gemm_q4_0_rocmfp4_fast_triton

    def spy(weights: torch.Tensor, activations: torch.Tensor, row: int) -> torch.Tensor:
        calls.append((activations.shape[-2], row))
        return original(weights, activations, row)

    monkeypatch.setattr(interface, "ggml_gemm_q4_0_rocmfp4_fast_triton", spy)
    weights = _make_weights(rows=2, blocks=1, scale_byte=0x40).cuda()
    activations = torch.ones((m, BLOCK_SIZE), dtype=torch.float32, device="cuda")

    output = interface.ggml_mul_mat_a8_triton(
        weights, activations, GGML_TYPE_Q4_0_ROCMFP4_FAST, row=2
    )

    assert output.shape == (m, 2)
    assert calls == [(m, 2)]


@pytest.mark.parametrize(
    ("weights", "activations", "row", "error", "message"),
    [
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.float32),
            torch.zeros((1, 32), dtype=torch.float32),
            1,
            TypeError,
            "uint8",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 32), dtype=torch.int32),
            1,
            TypeError,
            "float16",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros(32, dtype=torch.float32),
            1,
            ValueError,
            "2D or 3D",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 32), dtype=torch.float32),
            2,
            ValueError,
            "row.*W.shape",
        ),
        (
            torch.zeros((1, BLOCK_BYTES - 1), dtype=torch.uint8),
            torch.zeros((1, 32), dtype=torch.float32),
            1,
            ValueError,
            "divisible by 17",
        ),
        (
            torch.zeros((1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 31), dtype=torch.float32),
            1,
            ValueError,
            "hidden size",
        ),
        (
            torch.zeros((0, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, 32), dtype=torch.float32),
            0,
            ValueError,
            "positive",
        ),
        (
            torch.zeros((1, 0), dtype=torch.uint8),
            torch.zeros((1, 0), dtype=torch.float32),
            1,
            ValueError,
            "positive",
        ),
    ],
)
@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
def test_type_101_gemm_rejects_invalid_metadata_before_launch(
    weights: torch.Tensor,
    activations: torch.Tensor,
    row: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _kernel()(weights, activations, row=row)


@pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)
def test_type_101_gemm_accepts_noncontiguous_inputs_by_materializing_contiguous() -> (
    None
):
    weights = _make_weights(rows=2, blocks=2).cuda()
    # The stride-2 slice must still be as wide as the weights, so the source has
    # to be twice the hidden size. A same-width source yields a half-width view
    # and the call is rejected for a hidden-size mismatch before the kernel is
    # ever reached, which would leave non-contiguous handling untested.
    hidden = 2 * BLOCK_SIZE
    padded = torch.randn((3, 2 * hidden), dtype=torch.float32, device="cuda")
    activations = padded[:, ::2]
    assert activations.shape[-1] == hidden
    assert not activations.is_contiguous()

    output = _kernel()(weights, activations, row=2)

    _assert_close(output, _reference_gemm(weights, activations), activations, weights)


def test_type_101_public_gemm_uses_dedicated_triton_route(monkeypatch) -> None:
    weights = _make_weights(rows=1, blocks=1)
    activations = torch.ones((1, BLOCK_SIZE), dtype=torch.float32)
    expected = _reference_gemm(weights, activations)
    calls: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []

    def dedicated_gemm(
        W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
    ) -> torch.Tensor:
        calls.append((W, X, quant_type, row))
        return expected

    monkeypatch.setattr(ops, "_type101_triton_gemm_available", lambda W, X: True)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8_triton", dedicated_gemm)
    monkeypatch.setattr(
        ops,
        "ggml_dequantize",
        lambda *args: pytest.fail("dedicated Triton route must not dequantize"),
    )

    output = ops.ggml_mul_mat_a8(
        weights, activations, GGML_TYPE_Q4_0_ROCMFP4_FAST, row=1
    )

    torch.testing.assert_close(output, expected)
    assert calls == [(weights, activations, GGML_TYPE_Q4_0_ROCMFP4_FAST, 1)]


def test_type_101_public_gemm_falls_back_to_dequantize_then_dense(monkeypatch) -> None:
    weights = _make_weights(rows=2, blocks=1)
    activations = torch.arange(BLOCK_SIZE, dtype=torch.float32).reshape(1, -1)
    decoded = _reference_decode(weights).to(activations.dtype)
    calls: list[tuple[torch.Tensor, int, int, int, torch.dtype]] = []

    def dequantize(
        W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype
    ) -> torch.Tensor:
        calls.append((W, quant_type, m, n, dtype))
        return decoded

    monkeypatch.setattr(ops, "_type101_triton_gemm_available", lambda W, X: False)
    monkeypatch.setattr(
        ops,
        "ggml_mul_mat_a8_triton",
        lambda *args: pytest.fail("type 101 fallback must not use Triton GEMM"),
    )
    monkeypatch.setattr(ops, "ggml_dequantize", dequantize)

    output = ops.ggml_mul_mat_a8(
        weights, activations, GGML_TYPE_Q4_0_ROCMFP4_FAST, row=2
    )

    torch.testing.assert_close(output, activations @ decoded.T)
    assert calls == [
        (weights, GGML_TYPE_Q4_0_ROCMFP4_FAST, 2, BLOCK_SIZE, torch.float32)
    ]


def test_type_101_gemm_rejects_rank_three_weights_before_launch(monkeypatch) -> None:
    kernel = _kernel()
    module = importlib.import_module(kernel.__module__)
    monkeypatch.setattr(
        module,
        "_gemm_kernel",
        lambda *args, **kwargs: pytest.fail("invalid weights must not launch"),
    )

    with pytest.raises(ValueError, match="weights must be 2D, got 3D"):
        kernel(
            torch.zeros((1, 1, BLOCK_BYTES), dtype=torch.uint8),
            torch.zeros((1, BLOCK_SIZE), dtype=torch.float32),
            row=1,
        )
