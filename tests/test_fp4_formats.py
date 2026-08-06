# SPDX-License-Identifier: Apache-2.0
"""Contract for the GGML FP4 formats: MXFP4 (39) and NVFP4 (40).

The oracles here are written from the ABI in llama.cpp's ``ggml-common.h`` and
``ggml-quants.c``, and share no code with the plugin decoders under test, so a
shared misreading cannot hide itself (see AGENTS.md).

Both formats pack E2M1 codes through the same ``kvalues_fp4`` codebook and
differ only in blocking and scale encoding:

===========  ==========  ===========  ==================================
format       block       bytes        scale
===========  ==========  ===========  ==================================
MXFP4 (39)   32 weights  17           one E8M0 byte, trailing
NVFP4 (40)   64 weights  36           four UE4M3 bytes, leading, one per
                                      16-weight sub-block
===========  ==========  ===========  ==================================

A note on NVFP4, because the name is overloaded. NVIDIA's ModelOpt and
compressed-tensors NVFP4 carries per-tensor ``weight_global_scale`` and
``input_global_scale`` alongside the block scales. The GGUF format does not:
``quantize_row_nvfp4_ref`` derives each sub-block scale from that sub-block's
own absmax, and ``dequantize_row_nvfp4`` reads nothing else. The GGUF encoding
is therefore self-contained, and no companion tensors are required.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40

MXFP4_BLOCK_SIZE = 32
MXFP4_BLOCK_BYTES = 17

NVFP4_BLOCK_SIZE = 64
NVFP4_SUB_SIZE = 16
NVFP4_SUB_COUNT = NVFP4_BLOCK_SIZE // NVFP4_SUB_SIZE
NVFP4_BLOCK_BYTES = 36

# kvalues_fp4 from ggml-common.h: E2M1 values doubled, so the codebook is
# integral. The doubling is undone by the scale decoders, which both return
# half their raw value.
CODEBOOK = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _e8m0_scale(byte: int) -> float:
    """MXFP4 scale oracle: ``ggml_e8m0_to_fp32_half``.

    A pure exponent, no mantissa. Values below 2 are subnormal patterns; the
    rest are ``2 ** (byte - 128)``, already halved to match the doubled
    codebook.
    """
    if byte < 2:
        bits = 0x00200000 << byte
        return torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
    return 2.0 ** (byte - 128)


def _ue4m3_scale(byte: int) -> float:
    """NVFP4 scale oracle: ``ggml_ue4m3_to_fp32``.

    Four exponent bits at bias 7 and three mantissa bits, halved on return to
    match the doubled codebook -- so the effective bias is 8, as in ROCmFPX.
    ``0x00`` and ``0x7F`` both mean zero.
    """
    if byte in (0x00, 0x7F):
        return 0.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:
        raw = mantissa * 2.0**-9
    else:
        raw = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return raw * 0.5


def decode_mxfp4_block(block: Sequence[int]) -> list[float]:
    """Decode one 17-byte MXFP4 block to 32 floats.

    Layout is ``qs[16]`` then the scale byte. Low nibbles fill outputs 0..15 and
    high nibbles fill 16..31 -- contiguous halves, not interleaved.
    """
    if len(block) != MXFP4_BLOCK_BYTES:
        raise ValueError(f"expected {MXFP4_BLOCK_BYTES} bytes, got {len(block)}")
    scale = _e8m0_scale(block[16])
    out = [0.0] * MXFP4_BLOCK_SIZE
    for j in range(16):
        packed = block[j]
        out[j] = CODEBOOK[packed & 0xF] * scale
        out[j + 16] = CODEBOOK[packed >> 4] * scale
    return out


def decode_nvfp4_block(block: Sequence[int]) -> list[float]:
    """Decode one 36-byte NVFP4 block to 64 floats.

    Layout is ``d[4]`` then ``qs[32]``: the scales lead, unlike MXFP4. Each
    16-weight sub-block owns one scale and eight payload bytes, and within a
    sub-block the low nibbles fill the first eight outputs and the high nibbles
    the second eight.
    """
    if len(block) != NVFP4_BLOCK_BYTES:
        raise ValueError(f"expected {NVFP4_BLOCK_BYTES} bytes, got {len(block)}")
    out = [0.0] * NVFP4_BLOCK_SIZE
    for sub in range(NVFP4_SUB_COUNT):
        scale = _ue4m3_scale(block[sub])
        payload = NVFP4_SUB_COUNT + sub * (NVFP4_SUB_SIZE // 2)
        base = sub * NVFP4_SUB_SIZE
        for j in range(NVFP4_SUB_SIZE // 2):
            packed = block[payload + j]
            out[base + j] = CODEBOOK[packed & 0xF] * scale
            out[base + NVFP4_SUB_SIZE // 2 + j] = CODEBOOK[packed >> 4] * scale
    return out


GEOMETRY = {
    GGML_TYPE_MXFP4: (MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES),
    GGML_TYPE_NVFP4: (NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES),
}
FP4_TYPES = tuple(sorted(GEOMETRY))


class TestScaleOracles:
    """The two scale encodings, which are the formats' only real difference."""

    def test_e8m0_is_a_pure_power_of_two(self) -> None:
        """``bits = (x - 1) << 23`` gives ``2 ** (x - 128)``.

        The exponent field is one below the byte, and fp32 subtracts its own
        bias of 127, so the halving that matches the doubled codebook is already
        folded into the encoding rather than applied afterwards.
        """
        assert _e8m0_scale(128) == 1.0
        assert _e8m0_scale(129) == 2.0
        assert _e8m0_scale(127) == 0.5

    def test_e8m0_subnormals_are_tiny(self) -> None:
        assert 0.0 < _e8m0_scale(0) < 1e-30
        assert _e8m0_scale(1) == pytest.approx(2.0 * _e8m0_scale(0))

    def test_ue4m3_matches_rocmfpx_half_scale(self) -> None:
        """NVFP4 uses the same effective bias-8 half-scale as ROCmFPX."""
        assert _ue4m3_scale(0x40) == 1.0
        assert _ue4m3_scale(0x48) == 2.0
        assert _ue4m3_scale(0x38) == 0.5

    def test_ue4m3_reserved_bytes_are_zero(self) -> None:
        assert _ue4m3_scale(0x00) == 0.0
        assert _ue4m3_scale(0x7F) == 0.0

    def test_ue4m3_subnormals_use_the_mantissa_alone(self) -> None:
        assert _ue4m3_scale(0x01) == pytest.approx(1 / 1024.0)
        assert _ue4m3_scale(0x07) == pytest.approx(7 / 1024.0)


class TestBlockGeometry:
    """Block sizes must agree with what gguf reports, or loading misaddresses."""

    @pytest.mark.parametrize(
        ("quant_type", "block_size", "block_bytes"),
        [
            (GGML_TYPE_MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES),
            (GGML_TYPE_NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES),
        ],
    )
    def test_geometry_matches_gguf(
        self, quant_type: int, block_size: int, block_bytes: int
    ) -> None:
        import gguf

        reported = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(quant_type)]
        assert reported == (block_size, block_bytes)


class TestOracleDecoding:
    """Properties that separate a correct decoder from a plausible one."""

    def test_mxfp4_identity_block_exposes_nibble_order(self) -> None:
        """Scale 1.0 with ascending codes shows exactly where each nibble lands."""
        block = [(j % 16) | (((j + 1) % 16) << 4) for j in range(16)] + [128]
        decoded = decode_mxfp4_block(block)
        for j in range(16):
            assert decoded[j] == CODEBOOK[j % 16]
            assert decoded[j + 16] == CODEBOOK[(j + 1) % 16]

    def test_nvfp4_sub_blocks_scale_independently(self) -> None:
        """Each 16-weight sub-block must use its own scale, not a shared one."""
        # All codes are 1 (codebook value 1); scales double per sub-block.
        block = [0x38, 0x40, 0x48, 0x50] + [0x11] * 32
        decoded = decode_nvfp4_block(block)
        for sub, expected in enumerate((0.5, 1.0, 2.0, 4.0)):
            values = decoded[sub * 16 : (sub + 1) * 16]
            assert all(v == expected for v in values), f"sub-block {sub}"

    def test_nvfp4_reserved_scale_zeroes_only_its_sub_block(self) -> None:
        """A reserved scale must not zero the whole block."""
        block = [0x7F, 0x40, 0x40, 0x40] + [0x11] * 32
        decoded = decode_nvfp4_block(block)
        assert all(v == 0.0 for v in decoded[:16])
        assert all(v == 1.0 for v in decoded[16:])

    def test_mxfp4_and_nvfp4_share_the_codebook(self) -> None:
        """Same E2M1 table; only blocking and scale encoding differ."""
        mx = decode_mxfp4_block([0x10] * 16 + [128])
        nv = decode_nvfp4_block([0x40] * 4 + [0x10] * 32)
        # Code 0 -> 0.0, code 1 -> 1.0 in both.
        assert mx[0] == 0.0 and mx[16] == 1.0
        assert nv[0] == 0.0 and nv[8] == 1.0

    @pytest.mark.parametrize("quant_type", [GGML_TYPE_MXFP4, GGML_TYPE_NVFP4])
    def test_oracle_rejects_wrong_block_length(self, quant_type: int) -> None:
        decode = (
            decode_mxfp4_block if quant_type == GGML_TYPE_MXFP4 else decode_nvfp4_block
        )
        with pytest.raises(ValueError, match="expected"):
            decode([0] * 5)


# ---------------------------------------------------------------------------
# Dispatch contracts (RED until the kernels land)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_triton_gemm_dispatch_registered(quant_type: int) -> None:
    """Each FP4 format must resolve to a dedicated Triton GEMM kernel."""
    from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

    block_size, block_bytes = GEOMETRY[quant_type]
    w = torch.zeros((2, block_bytes), dtype=torch.uint8)
    x = torch.zeros((1, block_size), dtype=torch.float32)

    # An unregistered type raises "Unsupported Triton quant type" from the
    # dispatch table. A registered one reaches its kernel, which on CPU falls
    # back to a reference decode -- so reaching a result at all is the signal.
    try:
        out = ggml_mul_mat_a8_triton(w, x, quant_type, 2)
    except (ValueError, TypeError) as exc:
        assert "Unsupported Triton quant type" not in str(exc), (
            f"type {quant_type} has no dedicated Triton GEMM kernel"
        )
    else:
        assert out.shape == (1, 2)


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_triton_moe_dispatch_registered(quant_type: int) -> None:
    """Each FP4 format must appear in the fused-MoE dispatch table."""
    from vllm_gguf_plugin.triton.fused_moe.interface import TRITON_MOE_DISPATCH

    assert quant_type in TRITON_MOE_DISPATCH, (
        f"type {quant_type} missing from TRITON_MOE_DISPATCH"
    )


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_registration_tables_agree(quant_type: int) -> None:
    """All five registration tables must carry the type consistently.

    A missing entry surfaces as a KeyError inside shared validation, or -- worse
    for BLOCK_M -- as a silent default that produces working but mistuned code.
    """
    from vllm_gguf_plugin.triton.fused_moe.interface import TRITON_MOE_DISPATCH
    from vllm_gguf_plugin.triton.fused_moe.utils import (
        TRITON_FUSED_MOE_SUPPORTED_TYPES,
        TRITON_MOE_BLOCK_M_BY_TYPE,
    )
    from vllm_gguf_plugin.triton.gemm.utils import (
        BLOCK_BYTES_BY_TYPE,
        BLOCK_QK_BY_TYPE,
    )

    block_size, block_bytes = GEOMETRY[quant_type]

    assert BLOCK_BYTES_BY_TYPE[quant_type] == block_bytes
    assert BLOCK_QK_BY_TYPE[quant_type] == block_size
    assert quant_type in TRITON_FUSED_MOE_SUPPORTED_TYPES
    assert quant_type in TRITON_MOE_DISPATCH
    assert TRITON_MOE_BLOCK_M_BY_TYPE.get(quant_type) is not None, (
        f"type {quant_type} would silently inherit the default BLOCK_M"
    )


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_dense_linear_routes_to_gemm(quant_type: int) -> None:
    """Large-batch dense linear must reach the GEMM path, not dequant+dense."""
    from vllm_gguf_plugin.quantization.utils import MMQ_QUANT_TYPES

    assert quant_type in MMQ_QUANT_TYPES, (
        f"type {quant_type} would fall back to dequantize-plus-dense matmul"
    )


# ---------------------------------------------------------------------------
# Numerical parity against the independent oracle
# ---------------------------------------------------------------------------


def _pack_random(
    quant_type: int, rows: int, blocks: int, seed: int = 0
) -> torch.Tensor:
    """Build packed weights with scale bytes drawn from the usable range."""
    _, block_bytes = GEOMETRY[quant_type]
    generator = torch.Generator().manual_seed(
        quant_type * 1000 + rows + blocks + seed
    )
    packed = torch.randint(
        0, 256, (rows, blocks, block_bytes), dtype=torch.uint8, generator=generator
    )
    if quant_type == GGML_TYPE_MXFP4:
        # E8M0 spans the full byte range, but exponents far from 128 overflow
        # the reference in float32. Keep the scale within a usable window.
        packed[:, :, 16] = torch.randint(
            120, 136, (rows, blocks), dtype=torch.uint8, generator=generator
        )
    else:
        # UE4M3: 0x00 and 0x7F are reserved, and 0x80+ has the top bit set,
        # which the encoding does not define. Stay in the normal range.
        packed[:, :, :NVFP4_SUB_COUNT] = torch.randint(
            0x30,
            0x50,
            (rows, blocks, NVFP4_SUB_COUNT),
            dtype=torch.uint8,
            generator=generator,
        )
    return packed.reshape(rows, blocks * block_bytes)


def _oracle_decode(quant_type: int, packed: torch.Tensor) -> torch.Tensor:
    """Decode packed weights with the independent oracle."""
    _, block_bytes = GEOMETRY[quant_type]
    decode = decode_mxfp4_block if quant_type == GGML_TYPE_MXFP4 else decode_nvfp4_block
    rows = []
    for row in packed.tolist():
        values: list[float] = []
        for start in range(0, len(row), block_bytes):
            values.extend(decode(row[start : start + block_bytes]))
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


@pytest.mark.parametrize("quant_type", FP4_TYPES)
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
@pytest.mark.parametrize("blocks", [1, 3])
@pytest.mark.parametrize("rows", [2, 17])
def test_gemm_matches_oracle(
    quant_type: int, dtype: torch.dtype, blocks: int, rows: int
) -> None:
    """The GEMM must reproduce a dense matmul against oracle-decoded weights.

    Row counts include a tile tail (17 against BLOCK_N=64) so masked lanes are
    exercised rather than only full tiles.
    """
    from tests.numerics import assert_gemm_close
    from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

    block_size, _ = GEOMETRY[quant_type]
    hidden = blocks * block_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    packed = _pack_random(quant_type, rows, blocks)
    reference = _oracle_decode(quant_type, packed).to(device)

    generator = torch.Generator().manual_seed(quant_type + rows)
    x = torch.randn(4, hidden, generator=generator).to(device=device, dtype=dtype)
    w = packed.to(device)

    actual = ggml_mul_mat_a8_triton(w, x, quant_type, rows)
    expected = x.to(torch.float32) @ reference.T

    assert_gemm_close(
        actual.to(torch.float32),
        expected,
        x,
        dense_weights=reference,
        label=f"type {quant_type} rows={rows} blocks={blocks} {dtype}",
    )


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_gemm_rejects_malformed_inputs(quant_type: int) -> None:
    """Public validation must reject bad geometry before decoding."""
    from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

    block_size, block_bytes = GEOMETRY[quant_type]
    w = torch.zeros((2, block_bytes), dtype=torch.uint8)

    with pytest.raises(ValueError, match="hidden size"):
        ggml_mul_mat_a8_triton(
            w,
            torch.zeros((1, block_size + block_size), dtype=torch.float32),
            quant_type,
            2,
        )

    with pytest.raises(ValueError, match="row"):
        ggml_mul_mat_a8_triton(
            w, torch.zeros((1, block_size), dtype=torch.float32), quant_type, 5
        )

    with pytest.raises(ValueError, match="multiple"):
        ggml_mul_mat_a8_triton(
            torch.zeros((2, block_bytes + 1), dtype=torch.uint8),
            torch.zeros((1, block_size), dtype=torch.float32),
            quant_type,
            2,
        )


# ---------------------------------------------------------------------------
# Fused MoE
# ---------------------------------------------------------------------------

MOE_BLOCK_M = 4


def _align_tokens(
    topk_ids: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group routed slots by expert, padding each group to a block boundary.

    Mirrors moe_align_block_size. sorted_ids holds flat slot indices, not token
    indices: the kernel recovers the token as ``slot // top_k``.
    """
    flat = topk_ids.reshape(-1)
    num_valid = flat.numel()
    sorted_ids: list[int] = []
    expert_ids: list[int] = []
    for expert in range(experts):
        slots = (flat == expert).nonzero(as_tuple=True)[0].tolist()
        for start in range(0, len(slots), MOE_BLOCK_M):
            chunk = slots[start : start + MOE_BLOCK_M]
            sorted_ids.extend(chunk + [num_valid] * (MOE_BLOCK_M - len(chunk)))
            expert_ids.append(expert)
    device = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids, dtype=torch.int32, device=device),
        torch.tensor(len(sorted_ids), dtype=torch.int32, device=device),
    )


def _pack_experts(
    quant_type: int, experts: int, rows: int, blocks: int
) -> torch.Tensor:
    """Stack per-expert packed weights into ``[experts, rows, blocks*bytes]``."""
    return torch.stack(
        [
            _pack_random(quant_type, rows, blocks, seed=e * 7)
            for e in range(experts)
        ]
    )


@pytest.mark.parametrize("quant_type", FP4_TYPES)
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
def test_moe_matches_oracle(quant_type: int, dtype: torch.dtype) -> None:
    """The MoE kernel must match a dense matmul on oracle-decoded weights.

    Reduced precision is covered explicitly: tl.dot rejects mismatched operand
    dtypes, so a kernel leaving decoded weights in float32 compiles only for
    float32 activations and fails the moment a real bf16 model runs.
    """
    from tests.numerics import assert_gemm_close
    from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton

    if not torch.cuda.is_available():
        pytest.skip("fused MoE requires a GPU")

    block_size, _ = GEOMETRY[quant_type]
    experts, tokens, top_k = 4, 8, 2
    blocks = 2
    hidden = blocks * block_size

    packed = _pack_experts(quant_type, experts, hidden, blocks).cuda()
    reference = torch.stack(
        [_oracle_decode(quant_type, packed[e].cpu()) for e in range(experts)]
    ).cuda()

    generator = torch.Generator().manual_seed(quant_type)
    x = torch.randn(tokens, hidden, generator=generator).cuda().to(dtype)
    topk_ids = torch.randint(0, experts, (tokens, top_k), device="cuda")
    sorted_ids, expert_ids, padded = _align_tokens(topk_ids, experts)

    actual = ggml_moe_a8_triton(
        x, packed, sorted_ids, expert_ids, padded, quant_type, hidden, top_k, tokens
    )

    # Reproduce the kernel's slot ordering with dense matmuls.
    expected = torch.zeros_like(actual, dtype=torch.float32)
    for slot in range(tokens * top_k):
        token, k = slot // top_k, slot % top_k
        expert = int(topk_ids[token, k])
        expected[slot] = x[token].to(torch.float32) @ reference[expert].T

    assert_gemm_close(
        actual.to(torch.float32),
        expected,
        x,
        dense_weights=reference[0],
        label=f"type {quant_type} MoE {dtype}",
    )


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_moe_stays_within_weight_bounds(quant_type: int) -> None:
    """No FP4 MoE kernel may read past the end of its packed weight buffer.

    These decoders compute byte offsets into a packed buffer, so an addressing
    error reads neighbouring memory and still returns plausible numbers, which
    a comparison against a reference cannot see.
    """
    from tests.numerics import GuardedInput, assert_no_out_of_bounds_read
    from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton

    if not torch.cuda.is_available():
        pytest.skip("fused MoE requires a GPU")

    block_size, _ = GEOMETRY[quant_type]
    experts, tokens, top_k = 4, 8, 2
    blocks = 2
    hidden = blocks * block_size

    packed = _pack_experts(quant_type, experts, hidden, blocks).cuda()
    x = torch.randn(tokens, hidden, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, experts, (tokens, top_k), device="cuda")
    sorted_ids, expert_ids, padded = _align_tokens(topk_ids, experts)

    guarded = GuardedInput(packed)

    def invoke() -> torch.Tensor:
        return ggml_moe_a8_triton(
            x,
            guarded.tensor,
            sorted_ids,
            expert_ids,
            padded,
            quant_type,
            hidden,
            top_k,
            tokens,
        )

    assert_no_out_of_bounds_read(invoke, guarded, label=f"type {quant_type} MoE")


@pytest.mark.parametrize("quant_type", FP4_TYPES)
def test_gemm_stays_within_weight_bounds(quant_type: int) -> None:
    """No FP4 GEMM kernel may read past the end of its packed weight buffer."""
    from tests.numerics import GuardedInput, assert_no_out_of_bounds_read
    from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

    if not torch.cuda.is_available():
        pytest.skip("Triton GEMM requires a GPU")

    block_size, _ = GEOMETRY[quant_type]
    rows, blocks = 17, 3
    packed = _pack_random(quant_type, rows, blocks).cuda()
    x = torch.randn(4, blocks * block_size, dtype=torch.float32, device="cuda")

    guarded = GuardedInput(packed)

    def invoke() -> torch.Tensor:
        return ggml_mul_mat_a8_triton(guarded.tensor, x, quant_type, rows)

    assert_no_out_of_bounds_read(invoke, guarded, label=f"type {quant_type} GEMM")
