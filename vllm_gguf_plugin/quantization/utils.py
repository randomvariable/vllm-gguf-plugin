# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from types import MappingProxyType

from gguf import GGMLQuantizationType as WeightType
from vllm.logger import init_logger

from ..rocmfpx_types import GGML_TYPE_Q4_0_ROCMFP4_FAST

logger = init_logger(__name__)


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
):
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                shard_prefix in module_name for module_name in unquantized_modules
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# Low-bit formats with dedicated Triton GEMM and fused-MoE kernels but no
# native _C_gguf kernel. Routing membership means "dispatch through
# ops.ggml_mul_mat_a8 / ggml_moe_a8", not "a native kernel exists": those ops
# fall through to Triton when the native capability check fails. Q8_1 is the
# existing precedent -- it sits in MMQ_QUANT_TYPES with no native GEMM.
# Without membership these formats silently take the dequantise-plus-dense
# path and never reach their kernels.
LOWBIT_TRITON_TYPES = {
    WeightType.TQ1_0,
    WeightType.TQ2_0,
    WeightType.Q1_0,
    WeightType.Q2_0,
}

# The FP4 formats are in the same position: dedicated Triton kernels, no native
# _C_gguf kernel. Both share the E2M1 codebook and differ only in blocking and
# scale encoding -- MXFP4 32 weights with one trailing E8M0 byte, NVFP4 64 with
# four leading UE4M3 bytes.
FP4_TRITON_TYPES = {
    WeightType.MXFP4,
    WeightType.NVFP4,
}

DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = (
    STANDARD_QUANT_TYPES
    | KQUANT_TYPES
    | IMATRIX_QUANT_TYPES
    | LOWBIT_TRITON_TYPES
    | FP4_TRITON_TYPES
    | {GGML_TYPE_Q4_0_ROCMFP4_FAST}
)
MMQ_QUANT_TYPES = (
    STANDARD_QUANT_TYPES | KQUANT_TYPES | LOWBIT_TRITON_TYPES | FP4_TRITON_TYPES
)

# ROCmFPX custom quantization types (not in standard gguf enum)
from ..rocmfpx_types import (  # noqa: E402
    GGML_TYPE_Q2_0_ROCMFPX,
    GGML_TYPE_Q3_0_ROCMFPX,
    GGML_TYPE_Q4_0_ROCMFP4,
    GGML_TYPE_Q4_0_ROCMFP4_FAST,
    GGML_TYPE_Q6_0_ROCMFPX,
    GGML_TYPE_Q8_0_ROCMFPX,
)

ROCMFPX_TYPES = {
    GGML_TYPE_Q4_0_ROCMFP4,
    GGML_TYPE_Q4_0_ROCMFP4_FAST,
    GGML_TYPE_Q2_0_ROCMFPX,
    GGML_TYPE_Q3_0_ROCMFPX,
    GGML_TYPE_Q6_0_ROCMFPX,
    GGML_TYPE_Q8_0_ROCMFPX,
}
DEQUANT_TYPES = DEQUANT_TYPES | ROCMFPX_TYPES

# ik_llama.cpp K-variant i-quant types
from ..ik_types import (  # noqa: E402
    GGML_TYPE_I2_S,
    GGML_TYPE_IQ1_BN,
    GGML_TYPE_IQ1_KT,
    GGML_TYPE_IQ2_BN,
    GGML_TYPE_IQ2_K,
    GGML_TYPE_IQ2_KL,
    GGML_TYPE_IQ2_KS,
    GGML_TYPE_IQ2_KT,
    GGML_TYPE_IQ3_K,
    GGML_TYPE_IQ3_KS,
    GGML_TYPE_IQ3_KT,
    GGML_TYPE_IQ4_K,
    GGML_TYPE_IQ4_KS,
    GGML_TYPE_IQ4_KSS,
    GGML_TYPE_IQ4_KT,
    GGML_TYPE_IQ5_K,
    GGML_TYPE_IQ5_KS,
    GGML_TYPE_IQ6_K,
    GGML_TYPE_Q1_0_G128,
    GGML_TYPE_Q6_0,
)
from ..llama_types import (  # noqa: E402
    GGML_TYPE_Q2_0,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
)

IK_IQK_TYPES = {
    GGML_TYPE_IQ2_K,
    GGML_TYPE_IQ3_K,
    GGML_TYPE_IQ4_K,
    GGML_TYPE_IQ4_KS,
    GGML_TYPE_IQ5_K,
    GGML_TYPE_IQ6_K,
    GGML_TYPE_IQ2_KS,
    GGML_TYPE_IQ3_KS,
    GGML_TYPE_IQ5_KS,
    GGML_TYPE_IQ4_KSS,
    GGML_TYPE_IQ2_KL,
    GGML_TYPE_IQ1_KT,
    GGML_TYPE_IQ2_KT,
    GGML_TYPE_IQ3_KT,
    GGML_TYPE_IQ4_KT,
}
IK_BITNET_TYPES = {GGML_TYPE_IQ1_BN, GGML_TYPE_IQ2_BN}
IK_BASIC_TYPES = {GGML_TYPE_I2_S, GGML_TYPE_Q1_0_G128, GGML_TYPE_Q6_0}
LLAMA_EXTRA_TYPES = {GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q2_0}
DEQUANT_TYPES = DEQUANT_TYPES | IK_IQK_TYPES | IK_BITNET_TYPES | IK_BASIC_TYPES
DEQUANT_TYPES = DEQUANT_TYPES | LLAMA_EXTRA_TYPES
