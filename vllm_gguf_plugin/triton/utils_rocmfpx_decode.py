"""Shared Triton decode primitives for ROCmFPX formats.

Both the dense GEMM kernels (``triton/gemm/standard_quant/``) and the fused
MoE kernels (``triton/fused_moe/standard_quant/``) decode the same packed
weights. Inlining decode in both places is how ABI drift ships: a fix in one
lane never reaches the other. These helpers are the single source of truth.

Kept as ``@triton.jit`` functions so they inline into kernel bodies on both
the GEMM and MoE sides.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def decode_ue4m3_scale(scale_byte):
    """Decode one UE4M3 half-scale byte the ROCmFPX way.

    Normal: ``exp == 0`` is subnormal (``mantissa / 1024``), otherwise
    ``(1 + mantissa/8) * 2**(exp-8)``. Reserved bytes ``0x7F..0xFF`` decode to
    zero. This matches the native CUDA/HIP path (dequantize.cuh) and the
    Python reference (triton/dequantize/interface.py), which were validated
    bit-exactly on gfx1151 and GB10.
    """
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    return tl.where(
        scale_byte > 0x7E,
        0.0,
        tl.where(
            exponent == 0,
            mantissa.to(tl.float32) / 1024.0,
            (1.0 + mantissa.to(tl.float32) / 8.0)
            * tl.exp2(exponent.to(tl.float32) - 8.0),
        ),
    )
