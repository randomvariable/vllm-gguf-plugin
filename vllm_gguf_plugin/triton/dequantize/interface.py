import torch

from ...llama_types import (
    GGML_TYPE_Q1_0_G128,
    GGML_TYPE_Q2_0,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
)
from ..gemm.utils import (
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
)
from .ik_bn_reference import REFERENCE_DECODERS as BN_REFERENCE_DECODERS
from .ik_k56_reference import REFERENCE_DECODERS as K56_REFERENCE_DECODERS
from .ik_k234_reference import REFERENCE_DECODERS as K234_REFERENCE_DECODERS
from .ik_ks_reference import REFERENCE_DECODERS as KS_REFERENCE_DECODERS
from .ik_kss_kl_reference import REFERENCE_DECODERS as KSS_KL_REFERENCE_DECODERS
from .ik_kt_reference import REFERENCE_DECODERS as KT_REFERENCE_DECODERS
from .ik_reference import REFERENCE_DECODERS as IK_REFERENCE_DECODERS
from .iq_quant import (
    ggml_dequantize_iq1_m_triton,
    ggml_dequantize_iq1_s_triton,
    ggml_dequantize_iq2_s_triton,
    ggml_dequantize_iq2_xs_triton,
    ggml_dequantize_iq2_xxs_triton,
    ggml_dequantize_iq3_s_triton,
    ggml_dequantize_iq3_xxs_triton,
    ggml_dequantize_iq4_nl_triton,
    ggml_dequantize_iq4_xs_triton,
)
from .k_quant import (
    ggml_dequantize_q2_k_triton,
    ggml_dequantize_q3_k_triton,
    ggml_dequantize_q4_k_triton,
    ggml_dequantize_q5_k_triton,
    ggml_dequantize_q6_k_triton,
)
from .standard_quant import (
    ggml_dequantize_q4_0_triton,
    ggml_dequantize_q4_1_triton,
    ggml_dequantize_q5_0_triton,
    ggml_dequantize_q5_1_triton,
    ggml_dequantize_q8_0_triton,
    ggml_dequantize_q8_1_triton,
)

REFERENCE_DECODERS = {
    **IK_REFERENCE_DECODERS,
    **BN_REFERENCE_DECODERS,
    **K234_REFERENCE_DECODERS,
    **K56_REFERENCE_DECODERS,
    **KS_REFERENCE_DECODERS,
    **KSS_KL_REFERENCE_DECODERS,
    **KT_REFERENCE_DECODERS,
}


def _rocmfpx_scale(values: torch.Tensor) -> torch.Tensor:
    exponent = (values >> 3) & 0x0F
    mantissa = values & 0x07
    scale = torch.where(
        exponent == 0,
        mantissa.to(torch.float32) / 1024.0,
        (1.0 + mantissa.to(torch.float32) / 8.0)
        * torch.pow(2.0, exponent.to(torch.float32) - 8.0),
    )
    return torch.where(values <= 0x7E, scale, torch.zeros_like(scale))


def _dequantize_rocmfpx_reference(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None,
    quant_type: int,
) -> torch.Tensor:
    """Decode ROCmFPX blocks without requiring the native extension or Triton."""
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if not W.is_contiguous():
        W = W.contiguous()
    if m <= 0 or n <= 0:
        raise ValueError(
            f"GGUF type {quant_type} shape must have positive dimensions, got {m} x {n}"
        )
    total = int(m) * int(n)
    block_bytes = {
        100: 18,
        101: 17,
        102: 26,
        103: 33,
        104: 14,
        107: 10,
    }[int(quant_type)]
    if n % 32:
        raise ValueError(f"Dequantized element count {total} must be divisible by 32")
    expected_bytes = total // 32 * block_bytes
    if W.numel() != expected_bytes:
        raise ValueError(
            f"Quantized weights have {W.numel()} bytes, but type {quant_type} requires "
            f"{expected_bytes} bytes for shape ({m}, {n})"
        )

    raw = W.reshape(-1)[:expected_bytes].reshape(-1, block_bytes).to(torch.int32)
    if int(quant_type) in (100, 101):
        qs = raw[:, :16]
        codebook = torch.tensor(
            [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10],
            device=W.device,
            dtype=torch.float32,
        )
        low = codebook[qs & 0x0F]
        high = codebook[qs >> 4]
        if int(quant_type) == 100:
            # Type 100 stores contiguous low/high output halves.
            scale = _rocmfpx_scale(raw[:, 16:18])
            output = torch.cat((low * scale[:, 0:1], high * scale[:, 1:2]), dim=1)
        else:
            values = torch.cat((low, high), dim=1)
            scale = _rocmfpx_scale(raw[:, 16:17])
            output = values * scale
    elif int(quant_type) == 107:
        codes = torch.stack([((raw[:, :8] >> (2 * i)) & 3) for i in range(4)], dim=2)
        codebook = torch.tensor([-4, -1, 1, 4], device=W.device, dtype=torch.float32)
        output = codebook[codes].reshape(-1, 32)
        output *= _rocmfpx_scale(raw[:, 8:10]).repeat_interleave(16, dim=1)
    elif int(quant_type) == 104:
        codes = []
        for group in range(4):
            src = raw[:, group * 3 : group * 3 + 3]
            codes.append(
                torch.stack(
                    (
                        src[:, 0] & 7,
                        (src[:, 0] >> 3) & 7,
                        ((src[:, 0] >> 6) | (src[:, 1] << 2)) & 7,
                        (src[:, 1] >> 1) & 7,
                        (src[:, 1] >> 4) & 7,
                        ((src[:, 1] >> 7) | (src[:, 2] << 1)) & 7,
                        (src[:, 2] >> 2) & 7,
                        (src[:, 2] >> 5) & 7,
                    ),
                    dim=1,
                )
            )
        codebook = torch.tensor(
            [0, 1, 2, 4, 0, -1, -2, -4],
            device=W.device,
            dtype=torch.float32,
        )
        output = codebook[torch.cat(codes, dim=1)]
        output *= _rocmfpx_scale(raw[:, 12:14]).repeat_interleave(16, dim=1)
    elif int(quant_type) == 102:
        codes = []
        for group in range(8):
            src = raw[:, group * 3 : group * 3 + 3]
            codes.append(
                torch.stack(
                    (
                        src[:, 0] & 0x3F,
                        ((src[:, 0] >> 6) | (src[:, 1] << 2)) & 0x3F,
                        ((src[:, 1] >> 4) | (src[:, 2] << 4)) & 0x3F,
                        (src[:, 2] >> 2) & 0x3F,
                    ),
                    dim=1,
                )
            )
        codes = torch.cat(codes, dim=1)
        magnitude = codes & 31
        output = torch.where(
            (codes & 32) != 0,
            -(magnitude.where(magnitude != 0, 32)),
            magnitude,
        )
        output = output.to(torch.float32) * _rocmfpx_scale(
            raw[:, 24:26]
        ).repeat_interleave(16, dim=1)
    else:
        signed = raw[:, :32].where(raw[:, :32] < 128, raw[:, :32] - 256)
        output = signed.to(torch.float32) * _rocmfpx_scale(raw[:, 32:33])
    return output.reshape(m, n).to(dtype or torch.float16)


def _dequantize_llama_reference(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """Decode llama extras blocks without Triton or native extensions."""
    layouts = {
        GGML_TYPE_TQ1_0: (256, 54),
        GGML_TYPE_TQ2_0: (256, 66),
        GGML_TYPE_Q1_0_G128: (128, 18),
        GGML_TYPE_Q2_0: (64, 18),
    }
    qk, block_bytes = layouts[quant_type]
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if m <= 0 or n <= 0 or n % qk:
        raise ValueError(f"Invalid reference dequant shape ({m}, {n})")
    blocks = m * n // qk
    expected_bytes = blocks * block_bytes
    if W.numel() != expected_bytes:
        raise ValueError(
            f"Quantized weights have {W.numel()} bytes, but shape ({m}, {n}) "
            f"requires {expected_bytes} bytes"
        )

    raw = W.contiguous().reshape(blocks, block_bytes)
    if quant_type in (GGML_TYPE_Q1_0_G128, GGML_TYPE_Q2_0):
        scale = raw[:, :2].contiguous().view(torch.float16).to(torch.float32)
        qs = raw[:, 2:].to(torch.int32)
        positions = torch.arange(qk, device=W.device)
        if quant_type == GGML_TYPE_Q1_0_G128:
            values = torch.where(
                ((qs[:, positions // 8] >> (positions % 8)) & 1) != 0,
                1.0,
                -1.0,
            )
        else:
            values = ((qs[:, positions // 4] >> (2 * (positions % 4))) & 3) - 1
        return (
            (values.float() * scale[:, None]).reshape(m, n).to(dtype or torch.float16)
        )

    scale = raw[:, -2:].contiguous().view(torch.float16).to(torch.float32)
    if quant_type == GGML_TYPE_TQ2_0:
        qs = raw[:, :64].to(torch.int32)
        values = torch.cat(
            [
                ((qs[:, j : j + 32] >> (2 * shift)) & 3) - 1
                for j in (0, 32)
                for shift in range(4)
            ],
            dim=1,
        )
    else:
        powers = (1, 3, 9, 27, 81)
        qs = raw[:, :48].to(torch.int32)
        values = torch.cat(
            [
                (((qs[:, start : start + count] * power) & 0xFF) * 3 >> 8) - 1
                for start, count in ((0, 32), (32, 16))
                for power in powers
            ],
            dim=1,
        )
        qh = raw[:, 48:52].to(torch.int32)
        values = torch.cat(
            [
                values,
                torch.cat(
                    [(((qh * power) & 0xFF) * 3 >> 8) - 1 for power in powers[:4]],
                    dim=1,
                ),
            ],
            dim=1,
        )
    return (values.float() * scale[:, None]).reshape(m, n).to(dtype or torch.float16)


def ggml_dequantize_triton(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if int(quant_type) in (100, 101, 102, 103, 104, 107):
        return _dequantize_rocmfpx_reference(W, m, n, dtype, quant_type)
    if int(quant_type) in {
        GGML_TYPE_TQ1_0,
        GGML_TYPE_TQ2_0,
        GGML_TYPE_Q1_0_G128,
        GGML_TYPE_Q2_0,
    }:
        return _dequantize_llama_reference(W, int(quant_type), m, n, dtype)
    reference = REFERENCE_DECODERS.get(int(quant_type))
    if reference is not None:
        return reference(W, m, n, dtype)
    kernel = {
        GGML_TYPE_IQ1_M: ggml_dequantize_iq1_m_triton,
        GGML_TYPE_IQ1_S: ggml_dequantize_iq1_s_triton,
        GGML_TYPE_IQ2_S: ggml_dequantize_iq2_s_triton,
        GGML_TYPE_IQ2_XXS: ggml_dequantize_iq2_xxs_triton,
        GGML_TYPE_IQ2_XS: ggml_dequantize_iq2_xs_triton,
        GGML_TYPE_IQ3_S: ggml_dequantize_iq3_s_triton,
        GGML_TYPE_IQ3_XXS: ggml_dequantize_iq3_xxs_triton,
        GGML_TYPE_IQ4_NL: ggml_dequantize_iq4_nl_triton,
        GGML_TYPE_IQ4_XS: ggml_dequantize_iq4_xs_triton,
        GGML_TYPE_Q2_K: ggml_dequantize_q2_k_triton,
        GGML_TYPE_Q3_K: ggml_dequantize_q3_k_triton,
        GGML_TYPE_Q4_0: ggml_dequantize_q4_0_triton,
        GGML_TYPE_Q4_1: ggml_dequantize_q4_1_triton,
        GGML_TYPE_Q4_K: ggml_dequantize_q4_k_triton,
        GGML_TYPE_Q5_0: ggml_dequantize_q5_0_triton,
        GGML_TYPE_Q5_1: ggml_dequantize_q5_1_triton,
        GGML_TYPE_Q5_K: ggml_dequantize_q5_k_triton,
        GGML_TYPE_Q6_K: ggml_dequantize_q6_k_triton,
        GGML_TYPE_Q8_0: ggml_dequantize_q8_0_triton,
        GGML_TYPE_Q8_1: ggml_dequantize_q8_1_triton,
    }.get(int(quant_type))
    if kernel is None:
        raise ValueError(f"Unsupported Triton dequant quant type: {quant_type}")
    return kernel(W, m, n, dtype)
