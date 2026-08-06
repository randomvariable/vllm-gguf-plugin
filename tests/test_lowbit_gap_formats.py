"""Contract tests for the low-bit GGML gap formats: Q1_0, Q2_0, TQ1_0, TQ2_0.

These four formats have native dequantisation but no Triton GEMM or fused MoE
dispatch. This module supplies *independent* ABI oracles transcribed directly
from llama.cpp's ``ggml-common.h`` block structs and ``ggml-quants.c`` scalar
decode paths, then asserts the plugin's dispatch surfaces agree.

The oracles deliberately import nothing from the production decoders: a shared
helper cannot expose a shared misreading of the ABI.

ABI summary (llama.cpp):

* ``Q1_0``  (id 41, QK=128, 18B) -- ``fp16 d`` then ``qs[16]``.
  ``y[j] = ((qs[j/8] >> (j%8)) & 1) ? d : -d``
* ``Q2_0``  (id 42, QK=64,  18B) -- ``fp16 d`` then ``qs[16]``.
  ``y[j] = (((qs[j/4] >> ((j%4)*2)) & 3) - 1) * d``
* ``TQ2_0`` (id 35, QK=256, 66B) -- ``qs[64]`` then ``fp16 d``. Plane-major.
* ``TQ1_0`` (id 34, QK=256, 54B) -- ``qs[48]``, ``qh[4]``, ``fp16 d``. Base-3.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vllm_gguf_plugin  # noqa: F401  (registers the patched GGML enum)

GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q2_0 = 42

GAP_TYPES = (GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0)

# (block_size, block_bytes) straight from the llama.cpp static_asserts.
GEOMETRY: dict[int, tuple[int, int]] = {
    GGML_TYPE_TQ1_0: (256, 54),
    GGML_TYPE_TQ2_0: (256, 66),
    GGML_TYPE_Q1_0: (128, 18),
    GGML_TYPE_Q2_0: (64, 18),
}


# ---------------------------------------------------------------------------
# Independent oracles: pack + decode, transcribed from llama.cpp
# ---------------------------------------------------------------------------


def _fp16_bytes(value: float) -> tuple[int, int]:
    """Little-endian byte pair for an fp16 scale."""
    raw = np.float16(value).view(np.uint16).item()
    return raw & 0xFF, (raw >> 8) & 0xFF


def _fp16_from_bytes(lo: int, hi: int) -> float:
    return np.uint16((hi << 8) | lo).view(np.float16).astype(np.float32).item()


def oracle_pack_q1_0(codes: list[int], scale: float) -> bytes:
    """Pack 128 one-bit codes. Layout: fp16 d, then 16 payload bytes."""
    assert len(codes) == 128
    lo, hi = _fp16_bytes(scale)
    qs = bytearray(16)
    for j, bit in enumerate(codes):
        assert bit in (0, 1)
        qs[j // 8] |= bit << (j % 8)
    return bytes([lo, hi]) + bytes(qs)


def oracle_decode_q1_0(block: bytes) -> list[float]:
    d = _fp16_from_bytes(block[0], block[1])
    qs = block[2:18]
    out = []
    for j in range(128):
        bit = (qs[j // 8] >> (j % 8)) & 1
        out.append(d if bit else -d)
    return out


def oracle_pack_q2_0(codes: list[int], scale: float) -> bytes:
    """Pack 64 two-bit codes. Layout: fp16 d, then 16 payload bytes."""
    assert len(codes) == 64
    lo, hi = _fp16_bytes(scale)
    qs = bytearray(16)
    for j, code in enumerate(codes):
        assert 0 <= code <= 3
        qs[j // 4] |= code << ((j % 4) * 2)
    return bytes([lo, hi]) + bytes(qs)


def oracle_decode_q2_0(block: bytes) -> list[float]:
    d = _fp16_from_bytes(block[0], block[1])
    qs = block[2:18]
    out = []
    for j in range(64):
        code = (qs[j // 4] >> ((j % 4) * 2)) & 0x03
        out.append((code - 1) * d)
    return out


def oracle_pack_tq2_0(codes: list[int], scale: float) -> bytes:
    """Pack 256 two-bit codes plane-major. Layout: qs[64] then fp16 d.

    Decode order (llama.cpp): for j in (0, 32): for l in 0..3: for m in 0..31:
        q = (qs[j + m] >> (l * 2)) & 3
    So output index o maps to j = (o // 128) * 32, l = (o % 128) // 32,
    m = o % 32.
    """
    assert len(codes) == 256
    qs = bytearray(64)
    for o, code in enumerate(codes):
        assert 0 <= code <= 3
        j = (o // 128) * 32
        plane = (o % 128) // 32
        m = o % 32
        qs[j + m] |= code << (plane * 2)
    lo, hi = _fp16_bytes(scale)
    return bytes(qs) + bytes([lo, hi])


def oracle_decode_tq2_0(block: bytes) -> list[float]:
    qs = block[0:64]
    d = _fp16_from_bytes(block[64], block[65])
    out = []
    for j in (0, 32):
        for plane in range(4):
            for m in range(32):
                q = (qs[j + m] >> (plane * 2)) & 3
                out.append((q - 1) * d)
    return out


# TQ1_0 base-3 packing. Five ternary values share a byte via powers of three.
_POW3 = (1, 3, 9, 27, 81, 243)

_TQ1_0_QS_BYTES = 48
_TQ1_0_QH_BYTES = 4
# llama.cpp splits qs into a 32-byte stretch and a 16-byte tail:
#   sizeof(qs) - sizeof(qs) % 32 == 32
_TQ1_0_SPLIT = _TQ1_0_QS_BYTES - _TQ1_0_QS_BYTES % 32


def oracle_pack_tq1_0(codes: list[int], scale: float) -> bytes:
    """Pack 256 ternary codes (0, 1, 2). Layout: qs[48], qh[4], fp16 d.

    Transcribed from ``quantize_row_tq1_0_ref``. Two details are easy to miss
    and both change every byte:

    * after accumulating ``q = q*3 + xi`` the value is rescaled with a ceiling
      division ``(q * 256 + 242) / 243``, spreading 0..242 over 0..255 so the
      decoder's ``(q * 3) >> 8`` recovers each trit exactly;
    * the 4-trit ``qh`` tail gets an extra ``q *= 3`` so its first value still
      lands on the most significant trit.
    """
    assert len(codes) == 256
    qs = bytearray(_TQ1_0_QS_BYTES)
    qh = bytearray(_TQ1_0_QH_BYTES)
    off = 0

    # 5 elements per byte, along 32 bytes.
    for j in range(0, _TQ1_0_SPLIT, 32):
        for m in range(32):
            q = 0
            for n in range(5):
                q = q * 3 + codes[off + m + n * 32]
            qs[j + m] = (q * 256 + 242) // 243
        off += 5 * 32

    # 5 elements per byte, along the 16-byte tail.
    for j in range(_TQ1_0_SPLIT, _TQ1_0_QS_BYTES, 16):
        for m in range(16):
            q = 0
            for n in range(5):
                q = q * 3 + codes[off + m + n * 16]
            qs[j + m] = (q * 256 + 242) // 243
        off += 5 * 16

    # 4 elements per byte.
    for j in range(_TQ1_0_QH_BYTES):
        q = 0
        for m in range(4):
            q = q * 3 + codes[off + j + m * _TQ1_0_QH_BYTES]
        q *= 3  # shift the first value onto the most significant trit
        qh[j] = (q * 256 + 242) // 243

    lo, hi = _fp16_bytes(scale)
    return bytes(qs) + bytes(qh) + bytes([lo, hi])


def oracle_decode_tq1_0(block: bytes) -> list[float]:
    """Decode a TQ1_0 block, transcribed from ``dequantize_row_tq1_0``.

    The ``q * pow3[n]`` product is deliberately truncated to uint8, matching
    the C code's ``uint8_t q`` -- the wraparound is what isolates each trit.
    """
    qs = block[0:_TQ1_0_QS_BYTES]
    qh = block[_TQ1_0_QS_BYTES : _TQ1_0_QS_BYTES + _TQ1_0_QH_BYTES]
    d = _fp16_from_bytes(block[52], block[53])
    out: list[float] = []

    for j in range(0, _TQ1_0_SPLIT, 32):
        for n in range(5):
            for m in range(32):
                q = (qs[j + m] * _POW3[n]) & 0xFF
                xi = (q * 3) >> 8
                out.append(float(xi - 1) * d)

    for j in range(_TQ1_0_SPLIT, _TQ1_0_QS_BYTES, 16):
        for n in range(5):
            for m in range(16):
                q = (qs[j + m] * _POW3[n]) & 0xFF
                xi = (q * 3) >> 8
                out.append(float(xi - 1) * d)

    for n in range(4):
        for j in range(_TQ1_0_QH_BYTES):
            q = (qh[j] * _POW3[n]) & 0xFF
            xi = (q * 3) >> 8
            out.append(float(xi - 1) * d)

    assert len(out) == 256
    return out


ORACLES = {
    GGML_TYPE_Q1_0: (oracle_pack_q1_0, oracle_decode_q1_0, 2),
    GGML_TYPE_Q2_0: (oracle_pack_q2_0, oracle_decode_q2_0, 4),
    GGML_TYPE_TQ2_0: (oracle_pack_tq2_0, oracle_decode_tq2_0, 4),
    GGML_TYPE_TQ1_0: (oracle_pack_tq1_0, oracle_decode_tq1_0, 3),
}


# ---------------------------------------------------------------------------
# Oracle self-consistency: pack -> decode must round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_oracle_roundtrips(quant_type: int) -> None:
    """The oracle's packer and decoder must agree with each other."""
    pack, decode, num_codes = ORACLES[quant_type]
    block_size, block_bytes = GEOMETRY[quant_type]
    rng = np.random.default_rng(1234 + quant_type)
    codes = [int(c) for c in rng.integers(0, num_codes, size=block_size)]
    scale = 0.125

    blob = pack(codes, scale)
    assert len(blob) == block_bytes, (
        f"type {quant_type}: packed {len(blob)} bytes, ABI says {block_bytes}"
    )

    decoded = decode(blob)
    assert len(decoded) == block_size

    # Reconstruct the code from the decoded value and compare.
    for o, (code, value) in enumerate(zip(codes, decoded, strict=True)):
        if quant_type == GGML_TYPE_Q1_0:
            expected = scale if code else -scale
        elif quant_type == GGML_TYPE_Q2_0:
            expected = (code - 1) * scale
        else:  # ternary formats decode code-1
            expected = (code - 1) * scale
        assert value == pytest.approx(expected, abs=1e-6), (
            f"type {quant_type} index {o}: code={code} -> {value}, want {expected}"
        )


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_oracle_geometry_matches_gguf(quant_type: int) -> None:
    """Oracle geometry must match gguf-py's registered quant sizes."""
    import gguf

    block_size, block_bytes = GEOMETRY[quant_type]
    qtype = gguf.GGMLQuantizationType(quant_type)
    gguf_qk, gguf_bytes = gguf.GGML_QUANT_SIZES[qtype]
    assert (gguf_qk, gguf_bytes) == (block_size, block_bytes)


# ---------------------------------------------------------------------------
# Dispatch contracts (RED until the kernels land)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_triton_gemm_dispatch_registered(quant_type: int) -> None:
    """Each gap format must resolve to a dedicated Triton GEMM kernel."""
    from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

    block_size, block_bytes = GEOMETRY[quant_type]
    w = torch.zeros((2, block_bytes), dtype=torch.uint8)
    x = torch.zeros((1, block_size), dtype=torch.float32)

    # CPU tensors: the kernel must reject on device, *not* on unknown type.
    with pytest.raises((ValueError, TypeError)) as exc:
        ggml_mul_mat_a8_triton(w, x, quant_type, 2)
    assert "Unsupported Triton quant type" not in str(exc.value), (
        f"type {quant_type} has no dedicated Triton GEMM kernel"
    )


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_triton_moe_dispatch_registered(quant_type: int) -> None:
    """Each gap format must appear in the fused-MoE dispatch table."""
    from vllm_gguf_plugin.triton.fused_moe.interface import TRITON_MOE_DISPATCH

    assert quant_type in TRITON_MOE_DISPATCH, (
        f"type {quant_type} missing from TRITON_MOE_DISPATCH"
    )


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_moe_registration_tables_agree(quant_type: int) -> None:
    """All five MoE registration tables must carry the type coherently."""
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

    assert BLOCK_BYTES_BY_TYPE.get(quant_type) == block_bytes
    assert BLOCK_QK_BY_TYPE.get(quant_type) == block_size
    assert quant_type in TRITON_FUSED_MOE_SUPPORTED_TYPES
    assert quant_type in TRITON_MOE_DISPATCH
    assert TRITON_MOE_BLOCK_M_BY_TYPE.get(quant_type) is not None, (
        f"type {quant_type} would silently inherit the default BLOCK_M"
    )


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_native_dequant_still_registered(quant_type: int) -> None:
    """Scope guard: adding GEMM/MoE must not disturb native dequant support."""
    import vllm_gguf_plugin.ops as ops

    supported = (
        ops._CUDA_DEQUANT_ONLY_TYPES
        | ops._CUDA_GEMM_QUANT_TYPES
        | ops._CUDA_GEMV_QUANT_TYPES
    )
    assert quant_type in supported


def test_rocmfpx_moe_types_unchanged() -> None:
    """Scope guard: the gap formats are not ROCmFPX and must stay out of its set."""
    from vllm_gguf_plugin.triton.fused_moe.utils import ROCMFPX_MOE_TYPES

    for quant_type in GAP_TYPES:
        assert quant_type not in ROCMFPX_MOE_TYPES


# ---------------------------------------------------------------------------
# Public routing: registering a kernel is useless if dispatch never reaches it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_dense_linear_routes_to_gemm(quant_type: int) -> None:
    """Large-batch dense linear must route through ``ops.ggml_mul_mat_a8``.

    ``MMQ_QUANT_TYPES`` selects the routing branch, not native capability --
    ``Q8_1`` is already a member with no native GEMM and falls through to the
    Triton kernel. Without membership these formats silently take the
    dequantise-plus-dense path and never reach their dedicated kernel.
    """
    from vllm_gguf_plugin.quantization.utils import MMQ_QUANT_TYPES

    assert quant_type in MMQ_QUANT_TYPES, (
        f"type {quant_type} never reaches its Triton GEMM kernel from "
        "_fused_mul_mat_gguf"
    )


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_small_batch_routes_to_gemv(quant_type: int) -> None:
    """Small-batch dense linear must route through ``ops.ggml_mul_mat_vec_a8``.

    There is no dedicated native GEMV kernel for these formats; the vector op
    falls through to the Triton GEMM kernel, which handles ``M=1`` correctly.
    Membership is still required or small batches take the dequantise path.
    """
    from vllm_gguf_plugin.quantization.utils import MMVQ_QUANT_TYPES

    assert quant_type in MMVQ_QUANT_TYPES, (
        f"type {quant_type} takes the dequantise path for small batches"
    )


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_native_gemv_capability_not_overclaimed(quant_type: int) -> None:
    """Routing membership must not be mistaken for native kernel capability.

    ``_CUDA_GEMV_QUANT_TYPES`` and ``_CUDA_GEMM_QUANT_TYPES`` mean "a native
    ``_C_gguf`` kernel exists". No native GEMV/GEMM was written for these
    formats, so both sets must stay fail-closed.
    """
    import vllm_gguf_plugin.ops as ops

    assert quant_type not in ops._CUDA_GEMV_QUANT_TYPES
    assert quant_type not in ops._CUDA_GEMM_QUANT_TYPES


@pytest.mark.parametrize("quant_type", GAP_TYPES)
def test_moe_reaches_triton_via_mmq_or_mmvq(quant_type: int) -> None:
    """Fused MoE must reach a Triton kernel rather than the slow Python loop.

    ``_fused_moe_gguf`` selects the MMQ branch above 64 tokens and the MMVQ
    branch otherwise; both ``ops.ggml_moe_a8`` and ``ops.ggml_moe_a8_vec``
    fall through to ``ggml_moe_a8_triton`` when no native kernel exists.
    """
    from vllm_gguf_plugin.quantization.utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES

    assert quant_type in MMQ_QUANT_TYPES
    assert quant_type in MMVQ_QUANT_TYPES


def test_block_m_coherent_across_gap_types() -> None:
    """Mixed-quant MoE layers share one ``moe_align_block_size`` result.

    ``_fused_moe_gguf`` raises if the two sides disagree on ``BLOCK_M``, so the
    gap formats must agree with the ROCmFPX types they can be paired with.
    """
    from vllm_gguf_plugin.triton.fused_moe.utils import (
        ROCMFPX_MOE_TYPES,
        TRITON_MOE_BLOCK_M_BY_TYPE,
    )

    gap_block_ms = {TRITON_MOE_BLOCK_M_BY_TYPE[t] for t in GAP_TYPES}
    rocmfpx_block_ms = {TRITON_MOE_BLOCK_M_BY_TYPE[t] for t in ROCMFPX_MOE_TYPES}
    assert gap_block_ms == {4}
    assert gap_block_ms == rocmfpx_block_ms
