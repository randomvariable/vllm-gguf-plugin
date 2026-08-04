"""Regression coverage for the ROCmFPX MoE dispatch guard.

Type 101 (Q4_0_ROCMFP4_FAST) sits in ``MMVQ_QUANT_TYPES`` because that set is
reused for *linear* GEMV routing. Before the guard landed, ``_fused_moe_gguf``
also used it for MoE and sent type 101 into ``ops.ggml_moe_a8_vec``, which has
no ROCmFPX implementation and crashed. These tests pin the contract: every
ROCmFPX type routes through the fallback lane until a fused Triton MoE kernel
registers it, and the native MMQ/MMVQ ops are never called for them.

The contract under test is the *routing decision*, not fallback execution
(which needs vLLM native ops unavailable in a CPU-only environment). Native
ops are spied; any exception from the fallback lane is caught so the spy
record is the only assertion that matters.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import fused_moe
from vllm_gguf_plugin.quantization.utils import ROCMFPX_TYPES


def _make_inputs(quant_type: int, block_bytes: int):
    E, HIDDEN, INTER, TOPK, TOKENS = 2, 32, 32, 1, 4
    blocks = HIDDEN // 32
    w1 = torch.zeros((E, 2 * INTER, blocks * block_bytes), dtype=torch.uint8)
    w2 = torch.zeros((E, HIDDEN, blocks * block_bytes), dtype=torch.uint8)
    x = torch.zeros((TOKENS, HIDDEN), dtype=torch.float32)
    topk_weights = torch.ones((TOKENS, TOPK), dtype=torch.float32)
    topk_ids = torch.zeros((TOKENS, TOPK), dtype=torch.int32)
    return x, w1, w2, topk_weights, topk_ids


def _spy_on_native_moe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any call into the native MMQ/MMVQ MoE ops, return zeros."""
    calls: list[str] = []

    def record(name: str):
        def _spy(*args: object, **_: object) -> torch.Tensor:
            calls.append(name)
            # Return a rank-2 zero tensor shaped to keep act() happy enough.
            x = args[0] if args else None
            if isinstance(x, torch.Tensor):
                return torch.zeros((x.shape[0], 1), dtype=x.dtype)
            return torch.zeros(1)

        return _spy

    monkeypatch.setattr(ops, "ggml_moe_a8", record("mmq"))
    monkeypatch.setattr(ops, "ggml_moe_a8_vec", record("mmvq"))
    return calls


def _stub_fallback_matmul(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stub the fallback lane's per-expert matmul.

    ``fused_mul_mat_gguf_op`` is a registered vLLM custom op rebound inside the
    function body; the lookup target depends on registration state. We patch it
    wherever the name resolves from ``fused_moe``'s import, and no-op if the
    binding is not patchable in this environment -- the contract under test is
    the *routing* decision, not fallback execution.
    """
    calls: list[int] = []

    def _stub(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int):
        calls.append(qweight_type)
        return torch.zeros((*x.shape[:-1], qweight.shape[0]), dtype=x.dtype)

    with contextlib.suppress(AttributeError, TypeError):
        monkeypatch.setattr(fused_moe, "fused_mul_mat_gguf_op", _stub)
    return calls


# (quant type, block bytes per 32-weight block)
ROCmFPX_CASES = [
    pytest.param(100, 18, id="q4_0_rocmfp4"),
    pytest.param(101, 17, id="q4_0_rocmfp4_fast"),
    pytest.param(102, 26, id="q6_0_rocmfpx"),
    pytest.param(103, 33, id="q8_0_rocmfpx"),
    pytest.param(104, 14, id="q3_0_rocmfpx"),
    pytest.param(107, 10, id="q2_0_rocmfpx"),
]


@pytest.mark.parametrize(("quant_type", "block_bytes"), ROCmFPX_CASES)
def test_rocmfpx_moe_never_calls_native_mmq_or_mmvq(
    monkeypatch: pytest.MonkeyPatch, quant_type: int, block_bytes: int
) -> None:
    """ROCmFPX types must not reach the native MMQ/MMVQ MoE ops.

    Before the guard, type 101 crashed through ``ops.ggml_moe_a8_vec`` because
    ``MMVQ_QUANT_TYPES`` is reused for linear GEMV routing. The contract under
    test is the *routing decision*: the native ops are spied, and any exception
    from the fallback lane is swallowed because fallback execution needs vLLM
    native ops (``silu_and_mul``) unavailable in a CPU-only environment.
    """
    native_calls = _spy_on_native_moe(monkeypatch)

    x, w1, w2, topk_weights, topk_ids = _make_inputs(quant_type, block_bytes)
    with contextlib.suppress(Exception):
        fused_moe._fused_moe_gguf(
            x, w1, w2, topk_weights, topk_ids, quant_type, quant_type, "silu"
        )

    assert native_calls == [], (
        f"type {quant_type} reached native MoE ops: {native_calls}"
    )


def test_rocmfpx_guard_leaves_standard_types_on_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standard ggml type must still reach the native MMVQ op (guard is additive)."""
    native_calls = _spy_on_native_moe(monkeypatch)

    E, HIDDEN, INTER, TOPK, TOKENS = 2, 32, 32, 1, 4
    Q4_0_BLOCK_BYTES = 18
    w1 = torch.zeros((E, 2 * INTER, Q4_0_BLOCK_BYTES), dtype=torch.uint8)
    w2 = torch.zeros((E, HIDDEN, Q4_0_BLOCK_BYTES), dtype=torch.uint8)
    x = torch.zeros((TOKENS, HIDDEN), dtype=torch.float32)
    topk_weights = torch.ones((TOKENS, TOPK), dtype=torch.float32)
    topk_ids = torch.zeros((TOKENS, TOPK), dtype=torch.int32)

    # Token count <= 64 selects the MMVQ branch for standard types.
    with contextlib.suppress(Exception):
        fused_moe._fused_moe_gguf(x, w1, w2, topk_weights, topk_ids, 2, 2, "silu")

    assert "mmvq" in native_calls, (
        "standard type 2 lost its native MMVQ path after the ROCmFPX guard"
    )


def test_every_rocmfpx_type_is_covered_by_guard() -> None:
    """Coverage guard: a new ROCmFPX format cannot land without a dispatch test."""
    covered = {param.values[0] for param in ROCmFPX_CASES}
    assert covered == set(ROCMFPX_TYPES), (
        f"ROCmFPX_TYPES has {set(ROCMFPX_TYPES) ^ covered} untested"
    )
