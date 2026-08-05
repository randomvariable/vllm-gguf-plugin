# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from ..triton.fused_moe.interface import TRITON_MOE_DISPATCH, ggml_moe_a8_triton
from ..triton.fused_moe.utils import ROCMFPX_MOE_TYPES, get_triton_moe_block_m
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
)
from .utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, ROCMFPX_TYPES, logger


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    # ROCmFPX fused Triton MoE: gated on gfx115x availability, falls back to the
    # per-token per-expert loop below when ineligible.
    #
    # Real ROCmFPX GGUF exports mix quant types across the expert tensors. Read
    # directly from published GGUF headers:
    #
    #   Qwen3.6-14B-A3B-ROCmFPX-STRIX_LEAN    gate/up/down all 101
    #   Qwen-AgentWorld-35B-A3B Q6_0_ROCMFPX  gate/up 102, down {102, 103}
    #   Qwen-AgentWorld-35B-A3B Q4_0_ROCMFP4  up 100, gate {13, 100}, down {13, 14}
    #
    # So w13 and w2 legitimately carry different quant types, including a ROCmFPX
    # type paired with a K-quant. Requiring both sides to be ROCmFPX sent those
    # layers to the slow loop even though both tensors have working Triton MoE
    # kernels.
    #
    # ggml_moe_a8_triton already selects the kernel per tensor, so this only needs
    # both sides dispatchable. Requiring at least one ROCmFPX side keeps pure
    # K-quant layers on their existing native MMQ/MMVQ routing rather than
    # silently rerouting them onto an unmeasured path.
    if (
        (qweight_type in ROCMFPX_MOE_TYPES or qweight_type2 in ROCMFPX_MOE_TYPES)
        and qweight_type in TRITON_MOE_DISPATCH
        and qweight_type2 in TRITON_MOE_DISPATCH
        and ops._rocmfpx_gfx115x_available(w1, x)
    ):
        # Both matmuls share one moe_align_block_size result, so they must agree
        # on BLOCK_M. They do today (all 4); assert so a future per-type override
        # fails loudly instead of silently misaligning the second matmul.
        block_m = get_triton_moe_block_m(qweight_type)
        block_m2 = get_triton_moe_block_m(qweight_type2)
        if block_m != block_m2:
            raise ValueError(
                f"fused MoE BLOCK_M mismatch: type {qweight_type} uses {block_m}, "
                f"type {qweight_type2} uses {block_m2}; a shared token alignment "
                f"cannot serve both"
            )
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(qweight_type)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ggml_moe_a8_triton(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ggml_moe_a8_triton(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
        return out_hidden_states

    # ROCmFPX types dispatch through their own lane, not the native MMQ/MMVQ
    # sets. MMVQ_QUANT_TYPES includes type 101 for *linear* GEMV routing,
    # but reusing it for MoE sent type 101 into ops.ggml_moe_a8_vec (which
    # has no ROCmFPX implementation) and crashed. Keep every ROCmFPX type
    # out of these branches until a fused Triton MoE kernel registers it.
    rocmfpx_dispatch = qweight_type in ROCMFPX_TYPES or qweight_type2 in ROCMFPX_TYPES
    if (
        not rocmfpx_dispatch
        and qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif (
        not rocmfpx_dispatch
        and qweight_type2 in MMVQ_QUANT_TYPES
        and qweight_type in MMVQ_QUANT_TYPES
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], qweight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], qweight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, qweight_type, qweight_type2, activation
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Materialize GGUF lazy params after checkpoint loading.

        Mirrors GGUFLinearMethod.process_weights_after_loading: the
        w13/w2 qweight and qweight_type params are registered as
        GGUFUninitializedWeight*Parameter so the checkpoint loader can
        populate them lazily, then materialized into concrete
        GGUFWeight*Parameter instances here before the first forward.
        Without this step, RoutedExperts.forward hits
        'ValueError: Attempted to use an uninitialized parameter'.
        """
        self._materialize_gguf_parameters(layer)
        self._materialize_gguf_parameters(layer)

        self._materialize_qweight(layer, "w13_qweight")
        self._materialize_qweight_type(layer, "w13_qweight_type")
        self._materialize_qweight(layer, "w2_qweight")
        self._materialize_qweight_type(layer, "w2_qweight_type")

    def _materialize_qweight(
        self, layer: torch.nn.Module, param_name: str
    ) -> None:
        _materialize_gguf_weight_parameter(layer, param_name)

    def _materialize_qweight_type(
        self, layer: torch.nn.Module, param_name: str
    ) -> None:
        _materialize_gguf_weight_type_parameter(layer, param_name)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
        )
