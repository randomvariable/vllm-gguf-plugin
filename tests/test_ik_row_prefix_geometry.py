# SPDX-License-Identifier: Apache-2.0
"""Geometry/shape tests for row-prefix-aware ik_llama IQ KS/KSS/KL storage.

Authoritative ik_llama.cpp ABI for these types:

    row_size = row_meta_size + blocks_per_row * payload_size

GGML_QUANT_SIZES cannot express this layout (it assumes
``bytes = blocks * type_size``), so the legacy
``qweight.shape[1] // type_size * block_size`` shape formula undercounts
blocks for any row with more than one super-block. These tests pin the
explicit row-prefix geometry and the loader/materialization shape helpers
that consume it.
"""

from __future__ import annotations

import pytest
import torch

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin import ik_types

# Authoritative (row_meta, payload, QK) triples from ik_llama.cpp.
_AUTHORITATIVE_ABI: dict[int, tuple[int, int, int]] = {
    ik_types.GGML_TYPE_IQ4_KS: (4, 136, 256),
    ik_types.GGML_TYPE_IQ2_KS: (2, 70, 256),
    ik_types.GGML_TYPE_IQ3_KS: (2, 102, 256),
    ik_types.GGML_TYPE_IQ5_KS: (4, 168, 256),
    ik_types.GGML_TYPE_IQ4_KSS: (4, 128, 256),
    ik_types.GGML_TYPE_IQ2_KL: (2, 86, 256),
}

_ROW_PREFIX_TYPE_IDS = sorted(_AUTHORITATIVE_ABI)


def _row_bytes(row_meta: int, payload: int, blocks_per_row: int) -> int:
    return row_meta + blocks_per_row * payload


def gguf_std(name: str) -> int:
    """Resolve a standard gguf quant type id via the patched enum."""
    import gguf

    return int(gguf.GGMLQuantizationType[name])


def test_row_prefix_constants_match_authoritative_abi():
    meta = ik_types.ROW_PREFIX_GGUF_TYPES
    for type_id, (row_meta, payload, qk) in _AUTHORITATIVE_ABI.items():
        assert meta[type_id] == (row_meta, payload, qk), (
            f"type {type_id}: expected {(row_meta, payload, qk)}, "
            f"got {meta.get(type_id)}"
        )


@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_row_prefix_block_bytes_equals_prefix_plus_payload(type_id):
    """Public *_BLOCK_BYTES keep the historical prefix+payload value."""
    import gguf

    row_meta, payload, _ = _AUTHORITATIVE_ABI[type_id]
    name = gguf.GGMLQuantizationType(type_id).name
    block_const = {
        "IQ4_KS": ik_types.IQ4_KS_BLOCK_BYTES,
        "IQ2_KS": ik_types.IQ2_KS_BLOCK_BYTES,
        "IQ3_KS": ik_types.IQ3_KS_BLOCK_BYTES,
        "IQ5_KS": ik_types.IQ5_KS_BLOCK_BYTES,
        "IQ4_KSS": ik_types.IQ4_KSS_BLOCK_BYTES,
        "IQ2_KL": ik_types.IQ2_KL_BLOCK_BYTES,
    }[name]
    assert block_const == row_meta + payload


@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_is_row_prefix_gguf_type_true_for_row_prefix(type_id):
    assert ik_types.is_row_prefix_gguf_type(type_id)


@pytest.mark.parametrize(
    "type_id",
    [
        # Standard / K-quant / iquant types without a row prefix.
        gguf_std("Q4_0"),
        gguf_std("Q8_0"),
        ik_types.GGML_TYPE_IQ2_K,
        ik_types.GGML_TYPE_IQ4_K,
        ik_types.GGML_TYPE_IQ5_K,
        ik_types.GGML_TYPE_IQ6_K,
        # BitNet rows store prefixes separately and are intentionally excluded.
        ik_types.GGML_TYPE_IQ1_BN,
        ik_types.GGML_TYPE_IQ2_BN,
    ],
)
def test_is_row_prefix_gguf_type_false_for_others(type_id):
    assert not ik_types.is_row_prefix_gguf_type(type_id)


def _row_bytes_legacy_removed():  # pragma: no cover - placeholder removed
    ...


@pytest.mark.parametrize(
    ("type_id", "blocks_per_row"),
    [(t, b) for t in _ROW_PREFIX_TYPE_IDS for b in (1, 2, 16, 64)],
)
def test_gguf_qweight_dequant_shape_row_prefix(type_id, blocks_per_row):
    row_meta, payload, qk = _AUTHORITATIVE_ABI[type_id]
    num_rows = 7
    col_bytes = _row_bytes(row_meta, payload, blocks_per_row)
    shape = ik_types.gguf_qweight_dequant_shape(num_rows, col_bytes, type_id)
    expected_n = blocks_per_row * qk
    assert shape == (num_rows, expected_n)


@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_legacy_formula_undercounts_blocks_for_multi_block_rows(type_id):
    """Document the regression: old ``// type_size * block_size`` is wrong.

    With 16 super-blocks per row the legacy formula divides the row byte
    count (which includes a single per-row prefix) by ``prefix + payload``,
    so it drops almost one whole super-block.
    """
    row_meta, payload, qk = _AUTHORITATIVE_ABI[type_id]
    blocks_per_row = 16
    col_bytes = _row_bytes(row_meta, payload, blocks_per_row)
    type_size = row_meta + payload  # legacy registered type_size
    legacy_n = col_bytes // type_size * qk
    correct_n = blocks_per_row * qk
    assert legacy_n != correct_n, "legacy formula unexpectedly matched"
    assert correct_n == 16 * qk
    assert legacy_n < correct_n


@pytest.mark.parametrize(
    "type_id",
    [
        gguf_std("Q4_0"),
        gguf_std("Q8_0"),
        ik_types.GGML_TYPE_IQ2_K,
        ik_types.GGML_TYPE_IQ4_K,
    ],
)
def test_gguf_qweight_dequant_shape_standard_falls_through(type_id):
    """Non-row-prefix types still resolve via GGML_QUANT_SIZES."""
    import gguf

    block_size, type_size = gguf.GGML_QUANT_SIZES[type_id]
    num_rows = 5
    blocks_per_row = 12
    col_bytes = blocks_per_row * type_size
    shape = ik_types.gguf_qweight_dequant_shape(num_rows, col_bytes, type_id)
    assert shape == (num_rows, col_bytes // type_size * block_size)


# ---------------------------------------------------------------------------
# Integration: the four loader/materialization shape-helper sites must route
# row-prefix types through the new geometry. We monkeypatch
# ``ops.ggml_dequantize`` to capture the (m, n) it is asked to produce.
# ---------------------------------------------------------------------------


class _FakeDequant:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def __call__(self, w, quant_type, m, n, dtype):
        self.calls.append((int(quant_type), int(m), int(n)))
        return torch.zeros((int(m), int(n)), dtype=dtype, device=w.device)


@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_dequant_gemm_gguf_uses_row_prefix_shape(monkeypatch, type_id):
    import torch

    from vllm_gguf_plugin.quantization import diffusion_config

    row_meta, payload, qk = _AUTHORITATIVE_ABI[type_id]
    blocks_per_row = 16
    num_rows = 3
    col_bytes = _row_bytes(row_meta, payload, blocks_per_row)
    qweight = torch.zeros((num_rows, col_bytes), dtype=torch.uint8)
    x = torch.zeros((2, blocks_per_row * qk), dtype=torch.float32)

    fake = _FakeDequant()
    monkeypatch.setattr(ops, "ggml_dequantize", fake)
    # diffusion_config looks up `ops` via a module-level import alias.
    monkeypatch.setattr(diffusion_config.ops, "ggml_dequantize", fake)

    diffusion_config.dequant_gemm_gguf(x, qweight, type_id)

    assert fake.calls, "ggml_dequantize was not called"
    assert fake.calls[-1] == (type_id, num_rows, blocks_per_row * qk)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="row-prefix loader path is the ops.ggml_dequantize (CUDA) branch; "
    "CPU falls to gguf.dequantize which lacks row-prefix types",
)
@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_dense_weight_from_gguf_qweight_uses_row_prefix_shape(
    monkeypatch, type_id
):
    import torch

    from vllm_gguf_plugin.weights_adapter.diffusion import loader as diff_loader

    row_meta, payload, qk = _AUTHORITATIVE_ABI[type_id]
    blocks_per_row = 16
    num_rows = 3
    col_bytes = _row_bytes(row_meta, payload, blocks_per_row)
    qweight = torch.zeros((num_rows, col_bytes), dtype=torch.uint8)

    fake = _FakeDequant()
    monkeypatch.setattr(ops, "ggml_dequantize", fake)
    monkeypatch.setattr(diff_loader.ops, "ggml_dequantize", fake)

    diff_loader._dense_weight_from_gguf_qweight(qweight, type_id)

    assert fake.calls, "ggml_dequantize was not called"
    assert fake.calls[-1] == (type_id, num_rows, blocks_per_row * qk)


@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_fused_mul_mat_gguf_dequant_branch_uses_row_prefix_shape(
    monkeypatch, type_id
):
    import torch

    from vllm_gguf_plugin.quantization import linear as gguf_linear

    row_meta, payload, qk = _AUTHORITATIVE_ABI[type_id]
    blocks_per_row = 16
    num_rows = 3
    col_bytes = _row_bytes(row_meta, payload, blocks_per_row)
    qweight = torch.zeros((num_rows, col_bytes), dtype=torch.uint8)
    x = torch.zeros((2, blocks_per_row * qk), dtype=torch.float32)

    fake = _FakeDequant()
    monkeypatch.setattr(ops, "ggml_dequantize", fake)
    monkeypatch.setattr(gguf_linear.ops, "ggml_dequantize", fake)
    # Force the DEQUANT_TYPES fallback branch (not MMVQ/MMQ).
    monkeypatch.setattr(
        gguf_linear, "MMVQ_QUANT_TYPES", set(), raising=False
    )
    monkeypatch.setattr(
        gguf_linear, "MMQ_QUANT_TYPES", set(), raising=False
    )

    gguf_linear._fused_mul_mat_gguf(x, qweight, type_id)

    assert fake.calls, "ggml_dequantize was not called"
    assert fake.calls[-1] == (type_id, num_rows, blocks_per_row * qk)


@pytest.mark.parametrize("type_id", _ROW_PREFIX_TYPE_IDS)
def test_apply_gguf_embedding_uses_row_prefix_shape(monkeypatch, type_id):
    import torch

    from vllm_gguf_plugin.quantization import vocal_embeds

    row_meta, payload, qk = _AUTHORITATIVE_ABI[type_id]
    blocks_per_row = 16
    num_rows = 10
    col_bytes = _row_bytes(row_meta, payload, blocks_per_row)
    qweight = torch.zeros((num_rows, col_bytes), dtype=torch.uint8)
    x = torch.tensor([0, 1, 2], dtype=torch.long)
    hidden_size = blocks_per_row * qk

    fake = _FakeDequant()
    monkeypatch.setattr(ops, "ggml_dequantize", fake)
    monkeypatch.setattr(vocal_embeds.ops, "ggml_dequantize", fake)

    vocal_embeds._apply_gguf_embedding(
        x, qweight, type_id, hidden_size, dtype=torch.float32
    )

    assert fake.calls, "ggml_dequantize was not called"
    # _apply_gguf_embedding passes (hidden_size, num_indexed) as (m, n).
    assert fake.calls[-1] == (type_id, hidden_size, int(x.flatten().shape[0]))
