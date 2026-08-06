"""Tests for undoing llama.cpp's V-head tiling for qwen35moe linear attention.

llama.cpp's converter (``conversion/qwen.py::_reorder_v_heads``) permutes V
heads from HF *grouped* order into ggml *tiled* order so ``ggml_repeat`` can
broadcast cheaply::

    HF:   [G0_v0, G0_v1, G1_v0, G1_v1, ...]   grouped by K head
    GGUF: [G0_v0, G1_v0, ..., G0_v1, G1_v1]   tiled

vLLM's ``QwenGatedDeltaNetAttention`` expects HF grouped order, so the plugin
must invert the permutation on load.  It is a pure permutation, so a wrong
answer keeps every magnitude healthy while scrambling semantics.
"""

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter

NUM_K_HEADS = 16
NUM_V_HEADS = 32
NUM_V_PER_K = NUM_V_HEADS // NUM_K_HEADS
HEAD_K_DIM = 128
HEAD_V_DIM = 128


def _converter_reorder(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_per_k: int,
    head_dim: int,
) -> torch.Tensor:
    """Verbatim port of llama.cpp ``_reorder_v_heads`` (the forward direction).

    Independent oracle: transcribed from ``conversion/qwen.py:452-461``, not
    from the plugin implementation under test.
    """
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


class _Cfg:
    """Minimal stand-in for a qwen3_5_moe_text HF config."""

    model_type = "qwen3_5_moe_text"
    linear_num_key_heads = NUM_K_HEADS
    linear_num_value_heads = NUM_V_HEADS
    linear_key_head_dim = HEAD_K_DIM
    linear_value_head_dim = HEAD_V_DIM

    def get_text_config(self):
        return self


class _OtherCfg(_Cfg):
    model_type = "qwen3_moe"


@pytest.fixture
def adapter():
    return GGUFWeightsAdapter(_Cfg())


def _name(suffix: str) -> str:
    return f"model.layers.0.linear_attn.{suffix}"


class TestInversePermutation:
    """The undo must be the exact inverse, not a re-application of forward."""

    def test_roundtrip_is_identity(self, adapter):
        rows = NUM_V_HEADS * HEAD_V_DIM
        hf = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        gguf = _converter_reorder(hf, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_V_DIM)
        assert not torch.equal(gguf, hf), "fixture must actually permute"

        restored = adapter.transform_weight(_name("in_proj_z.weight"), gguf)
        assert torch.equal(restored, hf)

    def test_reapplying_forward_would_be_wrong(self):
        """Guards against 'permutation is its own inverse' (only true if K == VPK)."""
        rows = NUM_V_HEADS * HEAD_V_DIM
        hf = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
        gguf = _converter_reorder(hf, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_V_DIM)
        naive = _converter_reorder(gguf, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_V_DIM)
        assert not torch.equal(naive, hf)


class TestQkvOnlyPermutesVRows:
    def test_qk_rows_untouched_v_rows_restored(self, adapter):
        q_dim = k_dim = HEAD_K_DIM * NUM_K_HEADS
        v_dim = NUM_V_HEADS * HEAD_V_DIM
        hf = torch.arange((q_dim + k_dim + v_dim) * 2, dtype=torch.float32).reshape(
            q_dim + k_dim + v_dim, 2
        )
        q, k, v = hf[:q_dim], hf[q_dim : q_dim + k_dim], hf[q_dim + k_dim :]
        gguf = torch.cat(
            [q, k, _converter_reorder(v, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_V_DIM)],
            dim=0,
        )

        restored = adapter.transform_weight(_name("in_proj_qkv.weight"), gguf)
        assert torch.equal(restored, hf)
        assert torch.equal(restored[: q_dim + k_dim], hf[: q_dim + k_dim])


class TestPackedQuantizedTensors:
    """Row permutations are safe on packed bytes; rows are independent."""

    @pytest.mark.parametrize("block_bytes", [17, 18])
    def test_packed_rows_permute_bytewise(self, adapter, block_bytes):
        rows = NUM_V_HEADS * HEAD_V_DIM
        packed = torch.arange(rows * block_bytes, dtype=torch.uint8).reshape(
            rows, block_bytes
        )
        gguf = _converter_reorder(packed, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_V_DIM)
        restored = adapter.transform_weight(_name("in_proj_z.qweight"), gguf)
        assert torch.equal(restored, packed)

    def test_out_proj_permutes_packed_columns_block_aligned(self, adapter):
        """out_proj permutes the *input* dim, which lives inside packed blocks.

        head_v_dim=128 spans exactly 4 blocks of 32, so the permutation is
        block-aligned and can be done on raw bytes.
        """
        block_bytes = 17
        blocks_per_head = HEAD_V_DIM // 32
        out_dim = 8
        total_bytes = NUM_V_HEADS * blocks_per_head * block_bytes
        packed = torch.arange(out_dim * total_bytes, dtype=torch.uint8).reshape(
            out_dim, total_bytes
        )
        # Forward: permute head-sized byte groups along dim=1.
        gguf = _converter_reorder(
            packed, 1, NUM_K_HEADS, NUM_V_PER_K, blocks_per_head * block_bytes
        )
        assert not torch.equal(gguf, packed)

        restored = adapter.transform_weight(_name("out_proj.qweight"), gguf)
        assert torch.equal(restored, packed)


class TestPerHeadScalars:
    """A_log / dt_bias / b / a carry one element per V head (head_dim == 1)."""

    def test_dt_bias_1d(self, adapter):
        hf = torch.arange(NUM_V_HEADS, dtype=torch.float32)
        gguf = _converter_reorder(
            hf.unsqueeze(-1), 0, NUM_K_HEADS, NUM_V_PER_K, 1
        ).squeeze(-1)
        restored = adapter.transform_weight(_name("dt_bias"), gguf)
        assert torch.equal(restored, hf)

    def test_a_log_permuted_before_log_transform(self, adapter):
        """A_log needs both the V-head undo *and* the log(-x) inversion."""
        hf_a = -torch.arange(1, NUM_V_HEADS + 1, dtype=torch.float32)
        gguf = _converter_reorder(
            hf_a.unsqueeze(-1), 0, NUM_K_HEADS, NUM_V_PER_K, 1
        ).squeeze(-1)
        restored = adapter.transform_weight(_name("A_log"), gguf)
        assert torch.allclose(restored, torch.log(-hf_a), atol=1e-6)

    @pytest.mark.parametrize("suffix", ["in_proj_b.qweight", "in_proj_a.qweight"])
    def test_beta_alpha_rows(self, adapter, suffix):
        packed = torch.arange(NUM_V_HEADS * 17, dtype=torch.uint8).reshape(
            NUM_V_HEADS, 17
        )
        gguf = _converter_reorder(packed, 0, NUM_K_HEADS, NUM_V_PER_K, 1)
        restored = adapter.transform_weight(_name(suffix), gguf)
        assert torch.equal(restored, packed)


class TestConv1d:
    def test_only_v_channels_permuted_then_unsqueezed(self, adapter):
        qk_channels = HEAD_K_DIM * NUM_K_HEADS * 2
        v_channels = NUM_V_HEADS * HEAD_V_DIM
        kernel = 4
        hf = torch.arange(
            (qk_channels + v_channels) * kernel, dtype=torch.float32
        ).reshape(qk_channels + v_channels, kernel)
        qk_part, v_part = hf[:qk_channels], hf[qk_channels:]
        gguf = torch.cat(
            [
                qk_part,
                _converter_reorder(v_part, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_V_DIM),
            ],
            dim=0,
        )

        restored = adapter.transform_weight(_name("conv1d.weight"), gguf)
        # vLLM wants [channels, 1, kernel]
        assert restored.shape == (qk_channels + v_channels, 1, kernel)
        assert torch.equal(restored.squeeze(1), hf)


class TestSidecarTensors:
    """Quantised params also yield a scalar ``.qweight_type`` sidecar."""

    @pytest.mark.parametrize(
        "suffix",
        [
            "in_proj_qkv.qweight_type",
            "in_proj_z.qweight_type",
            "in_proj_b.qweight_type",
            "out_proj.qweight_type",
        ],
    )
    def test_scalar_weight_type_passes_through(self, adapter, suffix):
        scalar = torch.tensor(101)
        out = adapter.transform_weight(_name(suffix), scalar)
        assert torch.equal(out, scalar)


class TestNormPlusOne:
    """llama.cpp's converter bakes ``+1`` into norm weights.

    ``conversion/qwen.py::Qwen3NextModel.modify_tensors`` adds 1 to every
    tensor ending in ``norm.weight`` except ``linear_attn.norm.weight``.

    vLLM maps these norms to ``GemmaRMSNorm``, which itself computes
    ``x * (1 + w)``, so the stored ``+1`` must be removed on load or every
    norm is applied at roughly double strength.  ``linear_attn.norm`` is
    excluded by the converter and feeds ``RMSNormGated`` (plain ``x * w``),
    so it must pass through untouched.
    """

    @pytest.mark.parametrize(
        "suffix",
        [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ],
    )
    def test_plus_one_removed(self, adapter, suffix):
        hf = torch.tensor([-0.08, 0.33, 0.5, -0.25])
        gguf = hf + 1  # what the converter wrote
        out = adapter.transform_weight(f"model.layers.0.{suffix}", gguf)
        assert torch.allclose(out, hf, atol=1e-6)

    def test_final_norm_plus_one_removed(self, adapter):
        hf = torch.tensor([0.1, -0.2, 0.3])
        out = adapter.transform_weight("model.norm.weight", hf + 1)
        assert torch.allclose(out, hf, atol=1e-6)

    def test_linear_attn_norm_untouched(self, adapter):
        """Excluded by the converter; feeds RMSNormGated (plain x * w)."""
        w = torch.tensor([0.88, 0.95, 0.52])
        out = adapter.transform_weight(_name("norm.weight"), w)
        assert torch.equal(out, w)

    def test_non_norm_weights_untouched(self, adapter):
        w = torch.tensor([1.0, 2.0, 3.0])
        name = "model.layers.0.mlp.shared_expert.gate_proj.weight"
        assert torch.equal(adapter.transform_weight(name, w), w)

    def test_non_qwen35moe_norms_untouched(self):
        adapter = GGUFWeightsAdapter(_OtherCfg())
        w = torch.tensor([1.0, 0.9])
        name = "model.layers.0.input_layernorm.weight"
        assert torch.equal(adapter.transform_weight(name, w), w)


class TestScopeGuards:
    def test_norm_is_not_permuted(self, adapter):
        w = torch.arange(HEAD_V_DIM, dtype=torch.float32)
        assert torch.equal(adapter.transform_weight(_name("norm.weight"), w), w)

    def test_non_qwen35moe_model_untouched(self):
        adapter = GGUFWeightsAdapter(_OtherCfg())
        rows = NUM_V_HEADS * HEAD_V_DIM
        w = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
        assert torch.equal(adapter.transform_weight(_name("in_proj_z.weight"), w), w)

    def test_unrelated_names_pass_through(self, adapter):
        w = torch.arange(64, dtype=torch.float32)
        name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert torch.equal(adapter.transform_weight(name, w), w)

    def test_equal_head_counts_skip_permutation(self):
        """When num_k_heads == num_v_heads the converter does nothing."""

        class _Equal(_Cfg):
            linear_num_value_heads = NUM_K_HEADS

        adapter = GGUFWeightsAdapter(_Equal())
        rows = NUM_K_HEADS * HEAD_V_DIM
        w = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
        assert torch.equal(adapter.transform_weight(_name("in_proj_z.weight"), w), w)
