# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5 MoE linear-attention GGUF→HF 1:1 weight remapping.

With vLLM's GDN split-projection path (``create_in_proj_qkvz=False``,
``create_in_proj_ba=False``), GGUF's separate qkv/z/b/a tensors map 1:1
to vLLM params — no fusion or dequantization.  Linear-attn layers stay
quantized (llama.cpp memory parity).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter

# ---------------------------------------------------------------------------
# Fake config helpers (mirror tests/test_weights_adapter.py pattern)
# ---------------------------------------------------------------------------


class _FakeHFConfig:
    """Minimal HF config stand-in exposing qwen35moe linear-attn fields."""

    def __init__(
        self,
        num_hidden_layers: int = 4,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        model_type: str = "qwen3_5_moe_text",
    ) -> None:
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        if layer_types is None:
            # Layers 0,1,2 linear-attn; layer 3 full-attention.
            layer_types = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
        self.layer_types = layer_types
        self.full_attention_interval = full_attention_interval
        self.vision_config = None

    def get_text_config(self) -> _FakeHFConfig:
        return self


class _FakeModelConfig:
    def __init__(self, **kwargs: object) -> None:
        self.hf_config = _FakeHFConfig(**kwargs)
        self.trust_remote_code = False


def _build_adapter(model_config: _FakeModelConfig) -> GGUFWeightsAdapter:
    """Build name map with mocked AutoModel / gguf name resolver."""
    # Bypass __init__ (which requires a PretrainedConfig) — only build_name_map
    # is exercised, mirroring tests/test_weights_adapter.py.
    adapter = object.__new__(GGUFWeightsAdapter)
    with patch(
        "vllm_gguf_plugin.weights_adapter.default.gguf.get_tensor_name_map"
    ) as mock_name_map, patch(
        "vllm_gguf_plugin.weights_adapter.default.AutoModelForCausalLM"
    ) as mock_automodel:
        # Empty name map + empty state_dict → only our explicit remaps run.
        mock_name_map.return_value = {}
        mock_automodel.from_config.return_value.state_dict.return_value = {}
        name_map = adapter.build_name_map(model_config)
    adapter._test_name_map = name_map
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwen35MoeLinearAttnOneToOneMap:
    """1:1 GGUF→HF name mapping for linear-attn layers."""

    def test_name_map_contains_all_linear_attn_entries(self) -> None:
        """build_name_map must register all 9 SSM GGUF→HF entries."""
        adapter = _build_adapter(_FakeModelConfig())
        name_map = adapter._test_name_map
        la = "model.layers.0.linear_attn"
        # Split-projection entries (quantized sources stay quantized).
        assert name_map["blk.0.attn_qkv.weight"] == f"{la}.in_proj_qkv.weight"
        assert name_map["blk.0.attn_gate.weight"] == f"{la}.in_proj_z.weight"
        assert name_map["blk.0.ssm_beta.weight"] == f"{la}.in_proj_b.weight"
        assert name_map["blk.0.ssm_alpha.weight"] == f"{la}.in_proj_a.weight"
        assert name_map["blk.0.ssm_out.weight"] == f"{la}.out_proj.weight"
        # F32 sources (1:1 renames).
        assert name_map["blk.0.ssm_dt.bias"] == f"{la}.dt_bias"
        assert name_map["blk.0.ssm_norm.weight"] == f"{la}.norm.weight"
        assert name_map["blk.0.ssm_conv1d.weight"] == f"{la}.conv1d.weight"
        assert name_map["blk.0.ssm_a"] == f"{la}.A_log"

    def test_name_map_uses_split_projection_not_merged(self) -> None:
        """Must use in_proj_qkv/z/b/a, NOT in_proj_qkvz/ba (merged path)."""
        adapter = _build_adapter(_FakeModelConfig())
        name_map = adapter._test_name_map
        values = set(name_map.values())
        assert "model.layers.0.linear_attn.in_proj_qkvz.weight" not in values
        assert "model.layers.0.linear_attn.in_proj_ba.weight" not in values
        assert "model.layers.0.linear_attn.in_proj_qkv.weight" in values
        assert "model.layers.0.linear_attn.in_proj_z.weight" in values
        assert "model.layers.0.linear_attn.in_proj_b.weight" in values
        assert "model.layers.0.linear_attn.in_proj_a.weight" in values

    def test_no_fusion_markers_in_name_map(self) -> None:
        """No __fuse__ markers — pure 1:1 mapping."""
        adapter = _build_adapter(_FakeModelConfig())
        name_map = adapter._test_name_map
        for gguf_name, hf_name in name_map.items():
            assert "__fuse__" not in gguf_name, f"marker in gguf: {gguf_name}"
            assert "__fuse__" not in hf_name, f"marker in hf: {hf_name}"

    def test_no_fusion_attrs_on_adapter(self) -> None:
        """Dense-fusion machinery must be gone."""
        adapter = _build_adapter(_FakeModelConfig())
        assert not hasattr(adapter, "_qwen35moe_fusion_spec")
        assert not hasattr(adapter, "_qwen35moe_unquantized_modules")
        assert not hasattr(adapter, "_qwen35moe_fusion_outputs")

    def test_config_flags_set_to_false(self) -> None:
        """build_name_map must set create_in_proj_qkvz/ba=False on config."""
        cfg = _FakeModelConfig()
        adapter = object.__new__(GGUFWeightsAdapter)
        with patch(
            "vllm_gguf_plugin.weights_adapter.default.gguf.get_tensor_name_map"
        ) as mock_name_map, patch(
            "vllm_gguf_plugin.weights_adapter.default.AutoModelForCausalLM"
        ) as mock_automodel:
            mock_name_map.return_value = {}
            mock_automodel.from_config.return_value.state_dict.return_value = {}
            adapter.build_name_map(cfg)
        assert cfg.hf_config.create_in_proj_qkvz is False
        assert cfg.hf_config.create_in_proj_ba is False

    def test_config_flags_set_on_text_config(self) -> None:
        """Flags must also be on text_config (where GDN reads them)."""
        cfg = _FakeModelConfig()
        adapter = object.__new__(GGUFWeightsAdapter)
        with patch(
            "vllm_gguf_plugin.weights_adapter.default.gguf.get_tensor_name_map"
        ) as mock_name_map, patch(
            "vllm_gguf_plugin.weights_adapter.default.AutoModelForCausalLM"
        ) as mock_automodel:
            mock_name_map.return_value = {}
            mock_automodel.from_config.return_value.state_dict.return_value = {}
            adapter.build_name_map(cfg)
        text_cfg = cfg.hf_config.get_text_config()
        assert text_cfg.create_in_proj_qkvz is False
        assert text_cfg.create_in_proj_ba is False


class TestQwen35MoeTransformWeight:
    """Per-tensor transforms for F32 sources (A_log, conv1d)."""

    def test_a_log_log_transform(self) -> None:
        """GGUF ssm_a stores -exp(A_log) (negative); vLLM needs raw A_log.

        llama.cpp multiplies ssm_a straight into the gate, so the stored
        value is already -exp(A_log). vLLM computes -A_log.exp() instead,
        so the inverse is A_log = log(-ssm_a). Values are negative in every
        real checkpoint (verified on STRIX_LEAN: min -72.33, max -0.0186).
        """
        adapter = object.__new__(GGUFWeightsAdapter)
        ssm_a = torch.tensor([-1.0, -2.5, -0.5])
        a_log = adapter.transform_weight(
            "model.layers.0.linear_attn.A_log", ssm_a
        )
        assert torch.allclose(a_log, torch.log(-ssm_a))
        # Round-trip: vLLM's -A_log.exp() must recover the stored ssm_a.
        assert torch.allclose(-a_log.exp(), ssm_a, atol=1e-6)

    def test_conv1d_transform(self) -> None:
        adapter = object.__new__(GGUFWeightsAdapter)
        # GGUF conv1d shape metadata is reversed: reports [kernel, channels]
        # but the materialized tensor data is [channels, kernel]. vLLM wants
        # [channels, 1, kernel], so only unsqueeze(1) is needed (no transpose).
        src = torch.randn(14, 4, dtype=torch.float32)
        out = adapter.transform_weight(
            "model.layers.0.linear_attn.conv1d.weight", src
        )
        # vLLM conv1d: [channels, 1, kernel]
        assert out.shape == (14, 1, 4)
        assert torch.equal(out.squeeze(1), src)

    def test_other_names_passthrough(self) -> None:
        adapter = object.__new__(GGUFWeightsAdapter)
        w = torch.randn(8, 14)
        out = adapter.transform_weight(
            "model.layers.0.linear_attn.in_proj_qkv.weight", w
        )
        assert out is w

    def test_non_linear_attn_passthrough(self) -> None:
        adapter = object.__new__(GGUFWeightsAdapter)
        w = torch.randn(8, 8)
        out = adapter.transform_weight("model.layers.0.mlp.gate.weight", w)
        assert out is w


class TestQwen35MoeMapWeightsPassthrough:
    """map_weights must pass tensors through with only transform_weight applied."""

    def test_quantized_tensor_passthrough(self) -> None:
        """Quantized in_proj_qkv bytes must pass through unchanged (no fusion)."""
        adapter = object.__new__(GGUFWeightsAdapter)
        qweight = torch.zeros((8, 36), dtype=torch.uint8)
        qweight_type = torch.tensor(100)
        inputs = [
            ("model.layers.0.linear_attn.in_proj_qkv.qweight", qweight),
            (
                "model.layers.0.linear_attn.in_proj_qkv.qweight_type",
                qweight_type,
            ),
        ]
        outputs = dict(adapter.map_weights(iter(inputs)))
        # Bytes pass through unchanged — no dequant, no fusion.
        assert (
            outputs["model.layers.0.linear_attn.in_proj_qkv.qweight"] is qweight
        )
        assert (
            outputs["model.layers.0.linear_attn.in_proj_qkv.qweight_type"]
            is qweight_type
        )


class TestQwen35MoeFullAttentionLayer:
    """Full-attention layers must NOT get SSM remapping."""

    def test_full_attention_layer_excluded(self) -> None:
        adapter = _build_adapter(_FakeModelConfig())
        name_map = adapter._test_name_map
        # Layer 3 is full_attention per default _FakeHFConfig.
        for tensor_name in (
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
            "ssm_a",
            "ssm_conv1d.weight",
            "ssm_dt.bias",
            "ssm_norm.weight",
            "ssm_out.weight",
        ):
            assert f"blk.3.{tensor_name}" not in name_map, (
                f"full-attention layer 3 should not map {tensor_name}"
            )

    def test_linear_attn_layers_mapped(self) -> None:
        """Layers 0,1,2 (linear-attn) must have entries."""
        adapter = _build_adapter(_FakeModelConfig())
        name_map = adapter._test_name_map
        for i in (0, 1, 2):
            assert f"blk.{i}.ssm_a" in name_map
            assert f"blk.{i}.attn_qkv.weight" in name_map

    def test_fallback_interval_detection(self) -> None:
        """Without layer_types, use full_attention_interval to detect full-attn."""
        cfg = _FakeModelConfig(
            num_hidden_layers=4,
            layer_types=None,
            full_attention_interval=4,
        )
        adapter = _build_adapter(cfg)
        name_map = adapter._test_name_map
        # Layer 3 = 4-1 → full-attention → no SSM entries.
        assert "blk.3.ssm_a" not in name_map
        # Layer 0 → linear-attn → has SSM entries.
        assert "blk.0.ssm_a" in name_map

    def test_fallback_interval_eight_layers(self) -> None:
        """Every 4th layer (3,7) is full-attention; rest are linear-attn."""
        cfg = _FakeModelConfig(
            num_hidden_layers=8,
            layer_types=None,
            full_attention_interval=4,
        )
        adapter = _build_adapter(cfg)
        name_map = adapter._test_name_map
        for full_attn_layer in (3, 7):
            assert f"blk.{full_attn_layer}.ssm_a" not in name_map
        for linear_layer in (0, 1, 2, 4, 5, 6):
            assert f"blk.{linear_layer}.ssm_a" in name_map


class TestNonQwen35ModelUnchanged:
    """Non-qwen35moe models must not get linear-attn remapping or config flags."""

    def test_no_remap_for_qwen3_moe(self) -> None:
        from tests.test_weights_adapter import (
            _FakeModelConfig as _UpstreamModelConfig,
        )

        cfg = _UpstreamModelConfig("qwen3_moe")
        adapter = object.__new__(GGUFWeightsAdapter)
        with patch(
            "vllm_gguf_plugin.weights_adapter.default.gguf.get_tensor_name_map"
        ) as mock_name_map, patch(
            "vllm_gguf_plugin.weights_adapter.default.AutoModelForCausalLM"
        ) as mock_automodel:
            mock_name_map.return_value = {}
            mock_automodel.from_config.return_value.state_dict.return_value = {}
            adapter.build_name_map(cfg)
        # No config flags set.
        assert not hasattr(cfg.hf_config, "create_in_proj_qkvz")
        assert not hasattr(cfg.hf_config, "create_in_proj_ba")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
