# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex
import torch
from transformers import AutoModelForCausalLM
from vllm.logger import init_logger

from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class GGUFWeightsAdapter(BaseGGUFWeightsAdapter):
    """Default adapter for GGUF models."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        del config
        return True

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []

        # Qwen3.5 MoE shares the Qwen MoE expert layout but adds
        # linear-attention (SSM) layers. Normalize the model type *before*
        # the if/elif chain below so the shared Qwen MoE expert mapping
        # still applies; a dedicated elif branch would short-circuit the
        # chain and silently drop every ffn_*_exps tensor.
        if model_type == "qwen3_5_moe_text":
            model_type = "qwen35moe"
            self._add_qwen35moe_linear_attn_remaps(config, gguf_to_hf_name_map)

        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        elif model_type in ("qwen2_moe", "qwen3_moe", "qwen35moe"):
            model_type = model_type.replace("_", "")
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "olmoe":
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.extend(
                    [
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        ),
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                        ),
                    ]
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        text_name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_name_map = gguf.get_tensor_name_map(
                mm_proj_arch, config.vision_config.num_hidden_layers
            )
        else:
            vision_name_map = None

        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]
                if is_multimodal:
                    hf_name = "model." + hf_name
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]
                    suffix = "weight"
            gguf_name = None
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None:
                return None
            return gguf_name + "." + suffix

        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x
                for x in unmapped_params
                if not any(regex.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        return gguf_to_hf_name_map

    # ------------------------------------------------------------------
    # Qwen3.5 MoE linear-attention (SSM) weight fusion
    # ------------------------------------------------------------------

    def _add_qwen35moe_linear_attn_remaps(
        self,
        config,
        gguf_to_hf_name_map: dict[str, str],
    ) -> None:
        """Register 1:1 GGUF→HF name mappings for qwen35moe linear-attn layers.

        GGUF stores each linear-attn projection as a separate tensor.
        With vLLM's GDN split-projection path (``create_in_proj_qkvz=False``,
        ``create_in_proj_ba=False``), each GGUF tensor maps 1:1 to an HF param:

            blk.{i}.attn_qkv.weight   → linear_attn.in_proj_qkv.weight
            blk.{i}.attn_gate.weight  → linear_attn.in_proj_z.weight
            blk.{i}.ssm_beta.weight   → linear_attn.in_proj_b.weight
            blk.{i}.ssm_alpha.weight  → linear_attn.in_proj_a.weight
            blk.{i}.ssm_out.weight    → linear_attn.out_proj.weight
            blk.{i}.ssm_dt.bias       → linear_attn.dt_bias
            blk.{i}.ssm_norm.weight   → linear_attn.norm.weight
            blk.{i}.ssm_conv1d.weight → linear_attn.conv1d.weight
            blk.{i}.ssm_a             → linear_attn.A_log

        No fusion or dequantization — linear-attn layers stay quantized
        (llama.cpp memory parity).  Only ``A_log`` (log) and ``conv1d.weight``
        (transpose+unsqueeze) need per-tensor transforms, handled in
        ``transform_weight``.
        """
        # Force vLLM's QwenGatedDeltaNetAttention to use the split-projection
        # path so GGUF's separate qkv/z/b/a tensors map 1:1 without fusion.
        # These flags are read by GDN.__init__; must be set before
        # AutoModelForCausalLM.from_config (called later in build_name_map).
        # Set on both the config and its text_config for robustness across
        # mono- and multi-modal layouts.
        text_config = config.get_text_config()
        for cfg in (config, text_config):
            cfg.create_in_proj_qkvz = False
            cfg.create_in_proj_ba = False

        layer_types = getattr(config, "layer_types", None)
        full_attn_interval = getattr(config, "full_attention_interval", None)
        num_layers = config.num_hidden_layers

        for i in range(num_layers):
            is_full_attn = False
            if layer_types is not None and i < len(layer_types):
                is_full_attn = layer_types[i] == "full_attention"
            elif full_attn_interval:
                is_full_attn = i % full_attn_interval == full_attn_interval - 1
            if is_full_attn:
                continue

            la = f"model.layers.{i}.linear_attn"
            # --- 1:1 GGUF→HF name mappings for split-projection path ---
            gguf_to_hf_name_map[f"blk.{i}.attn_qkv.weight"] = (
                f"{la}.in_proj_qkv.weight"
            )
            gguf_to_hf_name_map[f"blk.{i}.attn_gate.weight"] = (
                f"{la}.in_proj_z.weight"
            )
            gguf_to_hf_name_map[f"blk.{i}.ssm_beta.weight"] = (
                f"{la}.in_proj_b.weight"
            )
            gguf_to_hf_name_map[f"blk.{i}.ssm_alpha.weight"] = (
                f"{la}.in_proj_a.weight"
            )
            gguf_to_hf_name_map[f"blk.{i}.ssm_out.weight"] = (
                f"{la}.out_proj.weight"
            )
            gguf_to_hf_name_map[f"blk.{i}.ssm_dt.bias"] = f"{la}.dt_bias"
            gguf_to_hf_name_map[f"blk.{i}.ssm_norm.weight"] = (
                f"{la}.norm.weight"
            )
            # Single-source transforms handled in transform_weight.
            gguf_to_hf_name_map[f"blk.{i}.ssm_conv1d.weight"] = (
                f"{la}.conv1d.weight"
            )
            gguf_to_hf_name_map[f"blk.{i}.ssm_a"] = f"{la}.A_log"

    def transform_weight(
        self, hf_name: str, weight: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-tensor transforms for qwen35moe linear-attention params.

        - ``A_log``: GGUF ``ssm_a`` already stores ``-exp(A_log)`` (negative),
          matching llama.cpp which multiplies it straight into the gate.
          vLLM instead computes ``-A_log.exp()``, so it needs the raw
          ``A_log`` back: ``A_log = log(-ssm_a)``.
        - ``conv1d.weight``: GGUF [kernel, channels] → vLLM [channels, 1, kernel].

        All other names pass through unchanged.
        """
        if hf_name.endswith(".linear_attn.A_log"):
            return torch.log(torch.clamp(-weight, min=1e-4))
        # GGUF's shape metadata is reversed relative to the materialized
        # tensor: ssm_conv1d reports [kernel, conv_dim] but the data is laid
        # out [conv_dim, kernel]. vLLM wants [conv_dim, 1, kernel].
        if hf_name.endswith(".linear_attn.conv1d.weight"):
            return weight.unsqueeze(1).contiguous()
        # GGUF stores the shared-expert gate as a 1-D [hidden] vector, but
        # vLLM builds it as ReplicatedLinear(hidden, 1) -> [1, hidden].
        if hf_name.endswith(".shared_expert_gate.weight") and weight.ndim == 1:
            return weight.unsqueeze(0).contiguous()
        return weight

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for hf_name, weight in weights:
            weight = self.transform_weight(hf_name, weight)
            if weight.ndim == 3 and ".experts.0." in hf_name:
                for expert_id, expert_weight in enumerate(weight.unbind()):
                    expert_name = hf_name.replace(
                        ".experts.0.", f".experts.{expert_id}."
                    )
                    yield expert_name, expert_weight
            else:
                yield hf_name, weight
    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def update_tie_word_embeddings(
        self,
        model_path: str,
        hf_config: PretrainedConfig,
        gguf_to_hf_name_map: dict[str, str],
    ) -> None:
        if "lm_head.weight" not in gguf_to_hf_name_map.values():
            return

        all_extra_names = []
        for gguf_file in self._get_all_gguf_files(model_path):
            all_extra_names.extend(
                get_gguf_extra_tensor_names(gguf_file, gguf_to_hf_name_map)
            )
        hf_config.update({"tie_word_embeddings": "lm_head.weight" in all_extra_names})

    def get_weight_type_map(
        self,
        model_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = {}
        for gguf_file in self._get_all_gguf_files(model_path):
            weight_type_map.update(
                get_gguf_weight_type_map(gguf_file, gguf_to_hf_name_map)
            )
        return weight_type_map

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        return [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        gguf_to_hf_name_map = self.build_name_map(model_config)
        self.update_tie_word_embeddings(
            model_path, model_config.hf_config, gguf_to_hf_name_map
        )
        weight_type_map = self.get_weight_type_map(model_path, gguf_to_hf_name_map)
        self.load_spec = GGUFLoadSpec(
            weights_source=self._get_all_gguf_files(model_path),
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=self.get_unquantized_modules(weight_type_map),
        )
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            self.load_spec.gguf_to_hf_name_map,
        )
        yield from self.map_weights(weights)
