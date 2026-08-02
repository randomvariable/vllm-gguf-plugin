# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCmFPX custom GGML type registration.

The ROCmFPX project (a llama.cpp fork) defines custom GGML tensor types
in the 100-106 range. The standard ``gguf`` Python package does not know
about these types, so we patch the enum and quant-size tables at import
time to allow the GGUFReader to parse tensors with these types.

Tensor type 100 = GGML_TYPE_Q4_0_ROCMFP4 (block_rocmfp4, 18 bytes/block,
32 weights/block). Type 101 = GGML_TYPE_Q4_0_ROCMFP4_FAST (block_rocmfp4_fast,
17 bytes/block, 32 weights/block).
"""

import gguf
from gguf.quants import GGMLQuantizationType

# ROCmFPX custom tensor type IDs
GGML_TYPE_Q4_0_ROCMFP4 = 100
GGML_TYPE_Q4_0_ROCMFP4_FAST = 101
GGML_TYPE_Q6_0_ROCMFPX = 102
GGML_TYPE_Q8_0_ROCMFPX = 103
GGML_TYPE_Q3_0_ROCMFPX = 104
GGML_TYPE_Q2_0_ROCMFPX = 107

# Block layout constants
Q4_0_ROCMFP4_QK = 32  # weights per block
Q4_0_ROCMFP4_BLOCK_BYTES = 18  # 16 qs + 2 scale bytes
Q4_0_ROCMFP4_FAST_BLOCK_BYTES = 17  # 16 qs + 1 scale byte

Q2_0_ROCMFPX_QK = 32
Q2_0_ROCMFPX_BLOCK_BYTES = 10  # 8 qs + 2 scale bytes

Q3_0_ROCMFPX_QK = 32
Q3_0_ROCMFPX_BLOCK_BYTES = 14  # 12 qs + 2 scale bytes

Q6_0_ROCMFPX_QK = 32
Q6_0_ROCMFPX_BLOCK_BYTES = 26  # 24 qs + 2 scale bytes

Q8_0_ROCMFPX_QK = 32
Q8_0_ROCMFPX_BLOCK_BYTES = 33  # 32 qs + 1 scale byte


def _patch_gguf_enum():
    """Add ROCmFPX types to the gguf GGMLQuantizationType enum.

    Python IntEnum doesn't support adding members after creation, so we
    rebuild the enum with the extra members using a dynamic subclass.
    """
    # Check if already patched
    if hasattr(GGMLQuantizationType, "Q4_0_ROCMFP4"):
        return

    # Build a new enum class with the extra members
    existing = {m.name: m.value for m in GGMLQuantizationType}
    existing["Q4_0_ROCMFP4"] = GGML_TYPE_Q4_0_ROCMFP4
    existing["Q4_0_ROCMFP4_FAST"] = GGML_TYPE_Q4_0_ROCMFP4_FAST
    existing["Q6_0_ROCMFPX"] = GGML_TYPE_Q6_0_ROCMFPX
    existing["Q8_0_ROCMFPX"] = GGML_TYPE_Q8_0_ROCMFPX
    existing["Q3_0_ROCMFPX"] = GGML_TYPE_Q3_0_ROCMFPX
    existing["Q2_0_ROCMFPX"] = GGML_TYPE_Q2_0_ROCMFPX

    # Replace the enum class in the gguf.quants module
    import enum

    new_enum = enum.IntEnum(
        "GGMLQuantizationType",
        existing,
    )

    # Patch the module references in all gguf submodules that imported
    # GGMLQuantizationType (handles `from gguf.quants import ...` copies)
    gguf.quants.GGMLQuantizationType = new_enum
    gguf.GGMLQuantizationType = new_enum
    # Patch any submodule that has a local reference
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if (
            mod_name
            and mod_name.startswith("gguf")
            and mod is not None
            and getattr(mod, "GGMLQuantizationType", None) is GGMLQuantizationType
        ):
            mod.GGMLQuantizationType = new_enum

    # Patch GGML_QUANT_SIZES with block sizes for new types
    # Format: (block_qk, type_size_bytes)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q4_0_ROCMFP4] = (
        Q4_0_ROCMFP4_QK,
        Q4_0_ROCMFP4_BLOCK_BYTES,
    )
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q4_0_ROCMFP4_FAST] = (
        Q4_0_ROCMFP4_QK,
        Q4_0_ROCMFP4_FAST_BLOCK_BYTES,
    )
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q2_0_ROCMFPX] = (
        Q2_0_ROCMFPX_QK,
        Q2_0_ROCMFPX_BLOCK_BYTES,
    )
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q3_0_ROCMFPX] = (
        Q3_0_ROCMFPX_QK,
        Q3_0_ROCMFPX_BLOCK_BYTES,
    )
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q6_0_ROCMFPX] = (
        Q6_0_ROCMFPX_QK,
        Q6_0_ROCMFPX_BLOCK_BYTES,
    )
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q8_0_ROCMFPX] = (
        Q8_0_ROCMFPX_QK,
        Q8_0_ROCMFPX_BLOCK_BYTES,
    )


_patch_gguf_enum()
