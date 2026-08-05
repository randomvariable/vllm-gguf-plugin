# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GGUF weights adapter model-type mapping."""
from unittest.mock import patch

import gguf

from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter


class _FakeHFConfig:
    def __init__(self, model_type: str, num_hidden_layers: int = 2):
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        self.vision_config = None

    def get_text_config(self):
        return self


class _FakeModelConfig:
    def __init__(self, model_type: str, num_hidden_layers: int = 2):
        self.hf_config = _FakeHFConfig(model_type, num_hidden_layers)
        self.trust_remote_code = False


def test_qwen3_5_moe_text_maps_to_qwen35moe_arch():
    """qwen3_5_moe_text HF configs must map to the qwen35moe GGUF arch.

    Regression guard: STRIX_LEAN / Qwen3.6-14B-A3B models ship with
    model_type="qwen3_5_moe_text" and GGUF architecture "qwen35moe".
    Without this mapping build_name_map raises "Unknown gguf model_type".
    """
    # build_name_map only needs model_config; bypass __init__ which builds a
    # dummy AutoModelForCausalLM from the HF config.
    adapter = object.__new__(GGUFWeightsAdapter)
    with patch(
        "vllm_gguf_plugin.weights_adapter.default.gguf.get_tensor_name_map"
    ) as mock_name_map, patch(
        "vllm_gguf_plugin.weights_adapter.default.AutoModelForCausalLM"
    ) as mock_automodel:
        # Avoid failing on unmapped SSM/linear_attn parameters; we only care
        # that the correct GGUF architecture is selected.
        mock_name_map.return_value = {}
        mock_automodel.from_config.return_value.state_dict.return_value = {}
        try:
            adapter.build_name_map(_FakeModelConfig("qwen3_5_moe_text"))
        except RuntimeError as exc:
            if "Unknown gguf model_type" in str(exc):
                raise AssertionError(
                    "qwen3_5_moe_text should not be treated as unknown"
                ) from exc
            # Other RuntimeErrors (unmapped params) are expected because we
            # mocked the name map to be empty.
    mock_name_map.assert_called_once()
    args, _ = mock_name_map.call_args
    assert args[0] == gguf.MODEL_ARCH.QWEN35MOE
