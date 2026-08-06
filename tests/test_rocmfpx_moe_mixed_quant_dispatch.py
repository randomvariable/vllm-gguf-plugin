"""Mixed-quant fused MoE dispatch contracts.

Real ROCmFPX GGUF exports mix quant types across the expert tensors. Observed
in the wild (GGUF headers read directly from HuggingFace):

    Qwen3.6-14B-A3B-ROCmFPX-STRIX_LEAN   gate/up/down all 101
    Qwen-AgentWorld-35B-A3B Q6_0_ROCMFPX gate/up 102, down {102, 103}
    Qwen-AgentWorld-35B-A3B Q4_0_ROCMFP4 up 100, gate {13, 100}, down {13, 14}

So a layer can legitimately present w13 and w2 with *different* quant types,
including one ROCmFPX type paired with a K-quant. Requiring both sides to be
ROCmFPX sends such layers to the slow per-token per-expert loop even though
both tensors have working Triton MoE kernels.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import fused_moe as fm
from vllm_gguf_plugin.triton.fused_moe.interface import TRITON_MOE_DISPATCH
from vllm_gguf_plugin.triton.fused_moe.utils import ROCMFPX_MOE_TYPES

GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q4_0_ROCMFP4 = 100
GGML_TYPE_Q4_0_ROCMFP4_FAST = 101
GGML_TYPE_Q6_0_ROCMFPX = 102
GGML_TYPE_Q8_0_ROCMFPX = 103
GGML_TYPE_Q3_0_ROCMFPX = 104
GGML_TYPE_Q2_0_ROCMFPX = 107

ALL_ROCMFPX = (100, 101, 102, 103, 104, 107)


@pytest.fixture(autouse=True)
def _stub_moe_align(monkeypatch: pytest.MonkeyPatch) -> None:
    """moe_align_block_size is a native _moe_C op, absent in CPU-only envs.

    Stub it so these dispatch tests exercise routing rather than dying in the
    alignment helper. Each expert group is padded to a block_m boundary, which
    is the contract the kernels rely on.
    """
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_fused_moe

    def fake_align(topk_ids, block_size, num_experts):
        tokens, top_k = topk_ids.shape
        by_expert: dict[int, list[int]] = {}
        for t in range(tokens):
            for k in range(top_k):
                by_expert.setdefault(int(topk_ids[t, k]), []).append(t * top_k + k)
        sorted_ids: list[int] = []
        expert_ids: list[int] = []
        for expert in sorted(by_expert):
            group = by_expert[expert]
            padded = group + [-1] * ((-len(group)) % block_size)
            sorted_ids.extend(padded)
            expert_ids.extend([expert] * (len(padded) // block_size))
        return (
            torch.tensor(sorted_ids, dtype=torch.int32),
            torch.tensor(expert_ids, dtype=torch.int32),
            torch.tensor([len(sorted_ids)], dtype=torch.int32),
        )

    monkeypatch.setattr(vllm_fused_moe, "moe_align_block_size", fake_align)

    # apply_moe_activation lowers to the native _C.silu_and_mul op, also absent
    # on CPU. Routing is what these tests assert, so a shape-correct stand-in
    # is enough to let dispatch reach the second (w2) call.
    def fake_activation(_enum, out, inp):
        out.copy_(inp[..., : out.shape[-1]])

    monkeypatch.setattr(fm, "apply_moe_activation", fake_activation)

    # moe_sum is likewise a native _moe_C op.
    def fake_moe_sum(inp, out):
        out.copy_(inp.sum(dim=1))

    monkeypatch.setattr(ops, "moe_sum", fake_moe_sum, raising=False)


@pytest.mark.parametrize("quant_type", ALL_ROCMFPX)
def test_every_rocmfpx_type_has_a_fused_moe_kernel(quant_type: int) -> None:
    """All six ROCmFPX formats must reach a fused MoE kernel.

    Kernels exist for every format; a format missing from the dispatch table
    silently degrades to the per-token per-expert fallback.
    """
    assert quant_type in TRITON_MOE_DISPATCH
    assert quant_type in ROCMFPX_MOE_TYPES


# (w13 type, w2 type, description) pairs taken from real model headers.
MIXED_PAIRS = [
    pytest.param(101, 101, id="strix_lean_uniform_101"),
    pytest.param(102, 103, id="agentworld_q6_mixed_rocmfpx"),
    pytest.param(100, GGML_TYPE_Q6_K, id="agentworld_q4_rocmfpx_plus_q6k"),
    pytest.param(100, GGML_TYPE_Q5_K, id="rocmfpx_plus_q5k"),
    pytest.param(GGML_TYPE_Q5_K, 100, id="q5k_plus_rocmfpx"),
]


@pytest.mark.parametrize(("w13_type", "w2_type"), MIXED_PAIRS)
def test_mixed_quant_layers_reach_the_fused_triton_path(
    monkeypatch: pytest.MonkeyPatch, w13_type: int, w2_type: int
) -> None:
    """A layer whose tensors both have Triton kernels must use them.

    The per-tensor kernel choice happens inside ggml_moe_a8_triton, so the
    dispatch decision only needs both sides to be dispatchable -- not to be
    the same type, and not both ROCmFPX.
    """
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda *_: True)

    calls: list[int] = []

    def spy_triton(x, w, *args, **kwargs):
        # signature: (x, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
        #             quant_type, row, top_k, num_tokens)
        quant_type = args[3] if len(args) > 3 else kwargs["quant_type"]
        rows = args[4] if len(args) > 4 else kwargs["row"]
        call_top_k = args[5] if len(args) > 5 else kwargs["top_k"]
        call_tokens = args[6] if len(args) > 6 else kwargs["num_tokens"]
        calls.append(quant_type)
        return torch.zeros((call_tokens * call_top_k, rows), dtype=x.dtype)

    def unexpected(*_: object, **__: object) -> torch.Tensor:
        pytest.fail("mixed-quant layer fell out of the fused Triton path")

    monkeypatch.setattr(fm, "ggml_moe_a8_triton", spy_triton)
    monkeypatch.setattr(fm, "fused_mul_mat_gguf_op", unexpected, raising=False)

    tokens, hidden, inter, experts, top_k = 4, 64, 64, 2, 2
    x = torch.zeros((tokens, hidden), dtype=torch.float32)
    w1 = torch.zeros((experts, 2 * inter, 36), dtype=torch.uint8)
    w2 = torch.zeros((experts, hidden, 36), dtype=torch.uint8)
    topk_weights = torch.full((tokens, top_k), 0.5, dtype=torch.float32)
    topk_ids = torch.zeros((tokens, top_k), dtype=torch.int32)

    fm._fused_moe_gguf(x, w1, w2, topk_weights, topk_ids, w13_type, w2_type, "silu")

    assert calls == [w13_type, w2_type], (
        f"expected per-tensor dispatch [{w13_type}, {w2_type}], got {calls}"
    )


def test_unsupported_type_still_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A type with no Triton kernel must not be forced into the fused path."""
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda *_: True)

    # IQ1_BN: decoders exist but no fused-MoE kernel. MXFP4 (39) previously
    # stood in here and no longer can, now that it has one.
    unsupported = 134
    assert unsupported not in TRITON_MOE_DISPATCH

    def unexpected(*_: object, **__: object) -> torch.Tensor:
        pytest.fail("unsupported quant type reached the fused Triton path")

    monkeypatch.setattr(fm, "ggml_moe_a8_triton", unexpected)

    tokens, hidden, inter, experts, top_k = 4, 64, 64, 2, 2
    x = torch.zeros((tokens, hidden), dtype=torch.float32)
    w1 = torch.zeros((experts, 2 * inter, 36), dtype=torch.uint8)
    w2 = torch.zeros((experts, hidden, 36), dtype=torch.uint8)
    topk_weights = torch.full((tokens, top_k), 0.5, dtype=torch.float32)
    topk_ids = torch.zeros((tokens, top_k), dtype=torch.int32)

    # Must not raise from the fused path; the fallback owns this case.
    with pytest.raises(Exception):  # noqa: B017 - fallback needs real weights
        fm._fused_moe_gguf(x, w1, w2, topk_weights, topk_ids, 100, unsupported, "silu")


def test_non_rocmfpx_layers_keep_existing_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure K-quant layers must not be pulled onto the ROCmFPX Triton path.

    Those already have native MMQ/MMVQ routing; rerouting them would be an
    unmeasured performance change, not a fix.
    """
    monkeypatch.setattr(ops, "_rocmfpx_gfx115x_available", lambda *_: True)

    def unexpected(*_: object, **__: object) -> torch.Tensor:
        pytest.fail("pure K-quant layer was rerouted onto the ROCmFPX path")

    monkeypatch.setattr(fm, "ggml_moe_a8_triton", unexpected)

    native_calls: list[int] = []

    def spy_mmq(x, w, *args, **kwargs):
        # (x, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
        #  quant_type, row, top_k, num_tokens)
        native_calls.append(args[3] if len(args) > 3 else kwargs["quant_type"])
        rows = args[4] if len(args) > 4 else kwargs["row"]
        call_top_k = args[5] if len(args) > 5 else kwargs["top_k"]
        call_tokens = args[6] if len(args) > 6 else kwargs["num_tokens"]
        return torch.zeros((call_tokens * call_top_k, rows), dtype=x.dtype)

    def spy_mmvq(x, w, *args, **kwargs):
        # (x, w, topk_ids, top_k, quant_type, row, num_tokens)
        native_calls.append(args[2] if len(args) > 2 else kwargs["quant_type"])
        rows = args[3] if len(args) > 3 else kwargs["row"]
        call_top_k = args[1] if len(args) > 1 else kwargs["top_k"]
        call_tokens = args[4] if len(args) > 4 else kwargs["num_tokens"]
        return torch.zeros((call_tokens * call_top_k, rows), dtype=x.dtype)

    monkeypatch.setattr(ops, "ggml_moe_a8", spy_mmq)
    monkeypatch.setattr(ops, "ggml_moe_a8_vec", spy_mmvq)

    tokens, hidden, inter, experts, top_k = 4, 64, 64, 2, 2
    x = torch.zeros((tokens, hidden), dtype=torch.float32)
    w1 = torch.zeros((experts, 2 * inter, 36), dtype=torch.uint8)
    w2 = torch.zeros((experts, hidden, 36), dtype=torch.uint8)
    topk_weights = torch.full((tokens, top_k), 0.5, dtype=torch.float32)
    topk_ids = torch.zeros((tokens, top_k), dtype=torch.int32)

    fm._fused_moe_gguf(
        x, w1, w2, topk_weights, topk_ids, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, "silu"
    )
    assert native_calls, "K-quant layer should still use the native MoE ops"
