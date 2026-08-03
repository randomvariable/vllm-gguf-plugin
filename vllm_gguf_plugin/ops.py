# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .ik_types import gguf_qweight_dequant_shape
from .llama_types import GGML_TYPE_Q2_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0
from .triton.dequantize.interface import ggml_dequantize_triton
from .triton.fused_moe.interface import ggml_moe_a8_triton
from .triton.fused_moe.utils import get_triton_moe_block_m
from .triton.gemm.interface import ggml_mul_mat_a8_triton
from .triton.gemm.utils import (
    GGML_TYPE_I2_S,
    GGML_TYPE_IQ1_BN,
    GGML_TYPE_IQ1_KT,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_BN,
    GGML_TYPE_IQ2_K,
    GGML_TYPE_IQ2_KL,
    GGML_TYPE_IQ2_KS,
    GGML_TYPE_IQ2_KT,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_K,
    GGML_TYPE_IQ3_KS,
    GGML_TYPE_IQ3_KT,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_K,
    GGML_TYPE_IQ4_KS,
    GGML_TYPE_IQ4_KSS,
    GGML_TYPE_IQ4_KT,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_IQ5_K,
    GGML_TYPE_IQ5_KS,
    GGML_TYPE_IQ6_K,
    GGML_TYPE_Q1_0_G128,
    GGML_TYPE_Q2_0_ROCMFPX,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_0_ROCMFPX,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_0_ROCMFP4,
    GGML_TYPE_Q4_0_ROCMFP4_FAST,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_0,
    GGML_TYPE_Q6_0_ROCMFPX,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_0_ROCMFPX,
)

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Backend selection: use CUDA kernels by default, unless explicitly disabled.
_USE_CUDA = os.environ.get("VLLM_GGUF_USE_CUDA", "1") == "1"

# Try importing CUDA extension
try:
    from . import _C_gguf  # noqa: F401

    _CUDA_AVAILABLE = True
except ImportError:
    _C_gguf = None
    _CUDA_AVAILABLE = False


# Effective CUDA usage: only when enabled AND available.
_CUDA_ENABLED = _USE_CUDA and _CUDA_AVAILABLE

_CUDA_GEMV_QUANT_TYPES = frozenset(
    {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ1_M,
    }
)
_CUDA_GEMM_QUANT_TYPES = frozenset(
    {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    }
)
_CUDA_DEQUANT_ONLY_TYPES = frozenset(
    {
        GGML_TYPE_IQ1_BN,
        GGML_TYPE_IQ2_BN,
        GGML_TYPE_I2_S,
        GGML_TYPE_Q1_0_G128,
        GGML_TYPE_Q6_0,
        GGML_TYPE_TQ1_0,
        GGML_TYPE_TQ2_0,
        GGML_TYPE_Q2_0,
        GGML_TYPE_IQ2_K,
        GGML_TYPE_IQ3_K,
        GGML_TYPE_IQ4_K,
        GGML_TYPE_IQ5_K,
        GGML_TYPE_IQ6_K,
        GGML_TYPE_IQ4_KS,
        GGML_TYPE_IQ2_KS,
        GGML_TYPE_IQ3_KS,
        GGML_TYPE_IQ5_KS,
        GGML_TYPE_IQ4_KSS,
        GGML_TYPE_IQ2_KL,
        GGML_TYPE_IQ1_KT,
        GGML_TYPE_IQ2_KT,
        GGML_TYPE_IQ3_KT,
        GGML_TYPE_IQ4_KT,
        GGML_TYPE_Q4_0_ROCMFP4,
        GGML_TYPE_Q4_0_ROCMFP4_FAST,
        GGML_TYPE_Q2_0_ROCMFPX,
        GGML_TYPE_Q3_0_ROCMFPX,
        GGML_TYPE_Q6_0_ROCMFPX,
        GGML_TYPE_Q8_0_ROCMFPX,
        GGML_TYPE_TQ1_0,
        GGML_TYPE_TQ2_0,
        GGML_TYPE_Q2_0,
    }
)


def _cuda_kernel_available(op_name: str, quant_type: int | None = None) -> bool:
    if not _CUDA_ENABLED:
        return False
    namespace = getattr(torch.ops, "_C_gguf", None)
    if namespace is None or not hasattr(namespace, op_name):
        return False
    if quant_type is None:
        return True
    quant_type = int(quant_type)
    if op_name == "ggml_mul_mat_vec_rocmfp4_fast":
        return quant_type == GGML_TYPE_Q4_0_ROCMFP4_FAST
    if op_name == "ggml_mul_mat_vec_rocmfpx":
        return quant_type in {
            GGML_TYPE_Q4_0_ROCMFP4,
            GGML_TYPE_Q2_0_ROCMFPX,
            GGML_TYPE_Q3_0_ROCMFPX,
            GGML_TYPE_Q6_0_ROCMFPX,
            GGML_TYPE_Q8_0_ROCMFPX,
        }
    if op_name == "ggml_dequantize":
        return quant_type in _CUDA_GEMV_QUANT_TYPES | _CUDA_DEQUANT_ONLY_TYPES
    return quant_type in _CUDA_GEMV_QUANT_TYPES - _CUDA_DEQUANT_ONLY_TYPES


def _cuda_gemm_kernel_available(op_name: str, quant_type: int) -> bool:
    return _cuda_kernel_available(op_name) and int(quant_type) in _CUDA_GEMM_QUANT_TYPES


_KT_QUANT_TYPES = frozenset(
    {GGML_TYPE_IQ1_KT, GGML_TYPE_IQ2_KT, GGML_TYPE_IQ3_KT, GGML_TYPE_IQ4_KT}
)


# --- Fake implementations for CUDA custom ops (needed for torch.compile) ---

if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_dequantize")
):

    @register_fake("_C_gguf::ggml_dequantize")
    def _ggml_dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    if hasattr(torch.ops._C_gguf, "ggml_mul_mat_vec_rocmfp4_fast"):

        @register_fake("_C_gguf::ggml_mul_mat_vec_rocmfp4_fast")
        def _ggml_mul_mat_vec_rocmfp4_fast_fake(
            W: torch.Tensor,
            X: torch.Tensor,
            quant_type: int,
            row: torch.SymInt,
        ) -> torch.Tensor:
            return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    if hasattr(torch.ops._C_gguf, "ggml_mul_mat_vec_rocmfpx"):

        @register_fake("_C_gguf::ggml_mul_mat_vec_rocmfpx")
        def _ggml_mul_mat_vec_rocmfpx_fake(
            W: torch.Tensor,
            X: torch.Tensor,
            quant_type: int,
            row: torch.SymInt,
        ) -> torch.Tensor:
            return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0), row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_moe_a8_vec")
):

    @register_fake("_C_gguf::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


# --- Public API ---


# ROCmFPX formats with a dedicated dense Triton GEMM kernel, mapped to their
# packed block size. Type 101 is excluded: it has its own native GEMV-first
# dispatch and a separate availability gate.
ROCMFPX_GEMM_BLOCK_BYTES = {
    GGML_TYPE_Q4_0_ROCMFP4: 18,
    GGML_TYPE_Q6_0_ROCMFPX: 26,
    GGML_TYPE_Q8_0_ROCMFPX: 33,
    GGML_TYPE_Q3_0_ROCMFPX: 14,
    GGML_TYPE_Q2_0_ROCMFPX: 10,
}


def _type101_triton_gemm_available(W: torch.Tensor, X: torch.Tensor) -> bool:
    if not W.is_cuda and X.is_cuda and torch.version.hip:
        return False
    try:
        arch = torch.cuda.get_device_properties(X.device).gcnArchName
    except (AttributeError, RuntimeError):
        return False
    return arch.startswith("gfx115")


def _rocmfpx_gfx115x_available(W: torch.Tensor, X: torch.Tensor) -> bool:
    """Shared eligibility gate for ROCmFPX Triton GEMM kernels.

    The kernels are wave32 and validated only on gfx115x targets, so every
    other device stays fail-closed on dequantize-plus-dense.
    """
    if not (W.is_cuda and X.is_cuda and torch.version.hip):
        return False
    try:
        arch = torch.cuda.get_device_properties(X.device).gcnArchName
    except (AttributeError, RuntimeError, AssertionError):
        return False
    return arch.startswith("gfx115")


def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    if int(quant_type) == GGML_TYPE_Q4_0_ROCMFP4 and (
        dtype is not None
        and dtype
        not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        )
    ):
        raise TypeError(
            "type-100 output dtype must be float32, float16, or bfloat16"
        )
    if int(quant_type) == GGML_TYPE_Q8_0_ROCMFPX:
        if W.ndim != 2 or W.dtype != torch.uint8:
            raise ValueError("type-103 weights must be a 2D uint8 packed tensor")
        if m != W.shape[0]:
            raise ValueError(
                f"type-103 row count {m} does not match weight rows {W.shape[0]}"
            )
        if m <= 0 or n <= 0:
            raise ValueError(
                "type-103 dequantization requires positive dimensions"
            )
        if W.shape[1] % 33:
            raise ValueError(
                "type-103 packed storage width must be divisible by 33 bytes"
            )
        hidden_size = W.shape[1] // 33 * 32
        if int(n) != hidden_size:
            raise ValueError("type-103 hidden size does not match packed weights")
    if W.is_cuda and _cuda_kernel_available("ggml_dequantize", quant_type):
        return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)
    return ggml_dequantize_triton(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_rocmfp4_fast(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    if not _cuda_kernel_available("ggml_mul_mat_vec_rocmfp4_fast", quant_type):
        raise RuntimeError("ROCmFP4_FAST GEMV native op is unavailable")
    return torch.ops._C_gguf.ggml_mul_mat_vec_rocmfp4_fast(W, X, quant_type, row)


def ggml_mul_mat_vec_rocmfpx(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    if not _cuda_kernel_available("ggml_mul_mat_vec_rocmfpx", quant_type):
        raise RuntimeError(
            "ROCmFPX GEMV native op is unavailable for type "
            f"{quant_type}"
        )
    return torch.ops._C_gguf.ggml_mul_mat_vec_rocmfpx(W, X, quant_type, row)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if row != W.shape[0]:
        raise ValueError(
            f"GEMV row count {row} does not match weight rows {W.shape[0]}"
        )

    if (
        W.is_cuda
        and X.is_cuda
        and _cuda_kernel_available("ggml_mul_mat_vec_rocmfpx", quant_type)
        and quant_type
        in {
            GGML_TYPE_Q4_0_ROCMFP4,
            GGML_TYPE_Q2_0_ROCMFPX,
            GGML_TYPE_Q3_0_ROCMFPX,
            GGML_TYPE_Q6_0_ROCMFPX,
            GGML_TYPE_Q8_0_ROCMFPX,
        }
    ):
        return ggml_mul_mat_vec_rocmfpx(W, X, quant_type, row)
    if (
        W.is_cuda
        and X.is_cuda
        and _cuda_kernel_available("ggml_mul_mat_vec_rocmfp4_fast", quant_type)
    ):
        return ggml_mul_mat_vec_rocmfp4_fast(W, X, quant_type, row)
    if quant_type == GGML_TYPE_Q4_0_ROCMFP4_FAST:
        m, n = gguf_qweight_dequant_shape(W.shape[0], W.shape[1], quant_type)
        weight = ggml_dequantize(W, quant_type, m, n, X.dtype)
        return X @ weight.T
    if int(quant_type) in _KT_QUANT_TYPES:
        m, n = gguf_qweight_dequant_shape(W.shape[0], W.shape[1], quant_type)
        weight = ggml_dequantize(W, quant_type, m, n, X.dtype)
        return X @ weight.T
    if _cuda_kernel_available("ggml_mul_mat_vec_a8", quant_type):
        return torch.ops._C_gguf.ggml_mul_mat_vec_a8(W, X, quant_type, row)
    return ggml_mul_mat_a8_triton(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if quant_type == GGML_TYPE_Q4_0_ROCMFP4_FAST:
        if W.ndim != 2:
            raise ValueError("type-101 weights must be 2D")
        if W.dtype != torch.uint8:
            raise TypeError("type-101 weights must use uint8")
        if X.ndim not in (2, 3):
            raise ValueError("type-101 activations must be 2D or 3D")
        if X.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "type-101 activations must use float16, bfloat16, or float32"
            )
        if W.device != X.device:
            raise ValueError(
                "type-101 weights and activations must use the same device"
            )
        if any(dim <= 0 for dim in W.shape) or any(dim <= 0 for dim in X.shape):
            raise ValueError(
                "type-101 weights and activations must have positive dimensions"
            )
        if W.shape[1] % 17:
            raise ValueError("type-101 packed width must be divisible by 17")
        hidden_size = W.shape[1] // 17 * 32
        if X.shape[-1] != hidden_size:
            raise ValueError("type-101 hidden size does not match packed weights")
        if row != W.shape[0]:
            raise ValueError(
                f"GEMV row count {row} does not match weight rows {W.shape[0]}"
            )
        if _type101_triton_gemm_available(W, X):
            return ggml_mul_mat_a8_triton(W, X, quant_type, row)
        m, n = gguf_qweight_dequant_shape(W.shape[0], W.shape[1], quant_type)
        weight = ggml_dequantize(W, quant_type, m, n, X.dtype)
        return X @ weight.T
    if int(quant_type) in ROCMFPX_GEMM_BLOCK_BYTES:
        label = f"type-{int(quant_type)}"
        block_bytes = ROCMFPX_GEMM_BLOCK_BYTES[int(quant_type)]
        if W.ndim != 2:
            raise ValueError(f"{label} weights must be 2D")
        if W.dtype != torch.uint8:
            raise TypeError(f"{label} weights must use uint8")
        if X.ndim not in (2, 3):
            raise ValueError(f"{label} activations must be 2D or 3D")
        if X.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"{label} activations must use float16, bfloat16, or float32"
            )
        if W.device != X.device:
            raise ValueError(
                f"{label} weights and activations must use the same device"
            )
        if any(dim <= 0 for dim in W.shape) or any(dim <= 0 for dim in X.shape):
            raise ValueError(
                f"{label} weights and activations must have positive dimensions"
            )
        if W.shape[1] % block_bytes:
            raise ValueError(
                f"{label} packed width must be divisible by {block_bytes} bytes"
            )
        hidden_size = W.shape[1] // block_bytes * 32
        if X.shape[-1] != hidden_size:
            raise ValueError(f"{label} hidden size does not match packed weights")
        if row != W.shape[0]:
            raise ValueError(
                f"GEMM row count {row} does not match weight rows {W.shape[0]}"
            )
        if _rocmfpx_gfx115x_available(W, X):
            return ggml_mul_mat_a8_triton(W, X, quant_type, row)
        m, n = gguf_qweight_dequant_shape(W.shape[0], W.shape[1], quant_type)
        weight = ggml_dequantize(W, quant_type, m, n, X.dtype)
        return X @ weight.T
    if int(quant_type) in _KT_QUANT_TYPES:
        m, n = gguf_qweight_dequant_shape(W.shape[0], W.shape[1], quant_type)
        weight = ggml_dequantize(W, quant_type, m, n, X.dtype)
        return X @ weight.T
    if _cuda_gemm_kernel_available("ggml_mul_mat_a8", quant_type):
        return torch.ops._C_gguf.ggml_mul_mat_a8(W, X, quant_type, row)
    return ggml_mul_mat_a8_triton(W, X, quant_type, row)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    if _cuda_gemm_kernel_available("ggml_moe_a8", quant_type):
        return torch.ops._C_gguf.ggml_moe_a8(
            X,
            W,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            quant_type,
            row,
            top_k,
            tokens,
        )
    return ggml_moe_a8_triton(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    if _cuda_kernel_available("ggml_moe_a8_vec", quant_type):
        return torch.ops._C_gguf.ggml_moe_a8_vec(
            X, W, topk_ids, top_k, quant_type, row, tokens
        )
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    E = W.shape[0]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, get_triton_moe_block_m(quant_type), E
    )
    return ggml_moe_a8_triton(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    if _cuda_gemm_kernel_available("ggml_moe_get_block_size", quant_type):
        return torch.ops._C_gguf.ggml_moe_get_block_size(quant_type)
    return get_triton_moe_block_m(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
