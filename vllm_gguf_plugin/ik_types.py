# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ik_llama.cpp K-variant i-quant type registration.

The ik_llama.cpp project (an ikawrakow/ik_llama.cpp fork of llama.cpp)
defines additional GGML tensor types in the 137-144 range for its
K-variant i-quant formats. These are SOTA low-bit quants used by the
community (e.g. IQ4_KS is the 35B-in-96GB sweet spot).

Type IDs:
  137 = GGML_TYPE_IQ2_K  (block_iq2_k,  76 bytes/256 weights, 2.375 bpw)
   138 = GGML_TYPE_IQ3_K  (block_iq3_k, 110 bytes/256 weights, 3.44 bpw)
   139 = GGML_TYPE_IQ4_K  (block_iq4_k, 144 bytes/256 weights, 4.5 bpw)
   140 = GGML_TYPE_IQ5_K  (block_iq5_k, 182 bytes/256 weights, 5.69 bpw)
   141 = GGML_TYPE_IQ6_K  (block_iq6_k, 214 bytes/256 weights, 6.69 bpw)
   144 = GGML_TYPE_IQ4_KS (block_iq4_ks, 136 bytes/256 weights + 4B row-prefix FP32 = 140 on-disk bytes/row, 4.25 bpw)

All use QK_K=256 super-blocks with software dequant and a dual-codebook
lookup via the `extra` field. IQ4_KS has a row-prefix FP32 scale.
"""

import gguf
from gguf.quants import GGMLQuantizationType

# ik_llama.cpp K-variant tensor type IDs
GGML_TYPE_IQ2_K = 137
GGML_TYPE_IQ3_K = 138
GGML_TYPE_IQ4_K = 139
GGML_TYPE_IQ5_K = 140
GGML_TYPE_IQ6_K = 141
GGML_TYPE_IQ4_KS = 144
GGML_TYPE_IQ2_KS = 145
GGML_TYPE_IQ5_KS = 152
GGML_TYPE_IQ3_KS = 156

# Block layout constants (per 256-weight super-block)
QK_IQ2_K = 256
IQ2_K_BLOCK_BYTES = 76  # half(2) + uint16(2) + scales[8] + qs[64] = 76

QK_IQ3_K = 256
IQ3_K_BLOCK_BYTES = 110  # half(2) + uint16(2) + uint16(2) + scales_l[8] + qs[64] + qh[32] = 110

QK_IQ4_K = 256
IQ4_K_BLOCK_BYTES = 144  # half(2) + uint16(2) + scales_h[4] + scales_l[8] + qs[128] = 144

QK_IQ5_K = 256
IQ5_K_BLOCK_BYTES = 182

QK_IQ6_K = 256
IQ6_K_BLOCK_BYTES = 214

QK_IQ4_KS = 256
# Effective on-disk row stride: 4-byte FP32 prefix + 136-byte block = 140.
# This is NOT sizeof(block_iq4_ks) — it includes the per-row FP32 scale
# that ggml stores as row_meta_size (ggml.c:4790-4792). The gguf package
# uses this value for tensor byte-size computation, so it must reflect
# the actual on-disk layout including the prefix.
IQ4_KS_BLOCK_BYTES = 140  # sizeof(float) + sizeof(block_iq4_ks) = 4 + 136

QK_IQ2_KS = 256
IQ2_KS_BLOCK_BYTES = 72  # half row prefix + 70-byte block

QK_IQ3_KS = 256
IQ3_KS_BLOCK_BYTES = 104  # half row prefix + 102-byte block

QK_IQ5_KS = 256
IQ5_KS_BLOCK_BYTES = 172  # float row prefix + 168-byte block


def _patch_gguf_enum():
    """Add ik K-variant types to the gguf GGMLQuantizationType enum."""
    if all(
        hasattr(GGMLQuantizationType, name)
        for name in (
            "IQ2_K",
            "IQ3_K",
            "IQ4_K",
            "IQ5_K",
            "IQ6_K",
            "IQ4_KS",
            "IQ2_KS",
            "IQ3_KS",
            "IQ5_KS",
        )
    ):
        return

    import enum

    existing = {m.name: m.value for m in GGMLQuantizationType}
    existing["IQ2_K"] = GGML_TYPE_IQ2_K
    existing["IQ3_K"] = GGML_TYPE_IQ3_K
    existing["IQ4_K"] = GGML_TYPE_IQ4_K
    existing["IQ5_K"] = GGML_TYPE_IQ5_K
    existing["IQ6_K"] = GGML_TYPE_IQ6_K
    existing["IQ4_KS"] = GGML_TYPE_IQ4_KS
    existing["IQ2_KS"] = GGML_TYPE_IQ2_KS
    existing["IQ3_KS"] = GGML_TYPE_IQ3_KS
    existing["IQ5_KS"] = GGML_TYPE_IQ5_KS

    new_enum = enum.IntEnum("GGMLQuantizationType", existing)

    gguf.quants.GGMLQuantizationType = new_enum
    gguf.GGMLQuantizationType = new_enum

    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod_name and mod_name.startswith("gguf") and mod is not None:
            if getattr(mod, "GGMLQuantizationType", None) is GGMLQuantizationType:
                mod.GGMLQuantizationType = new_enum

    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ2_K] = (QK_IQ2_K, IQ2_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ3_K] = (QK_IQ3_K, IQ3_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ4_K] = (QK_IQ4_K, IQ4_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ5_K] = (QK_IQ5_K, IQ5_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ6_K] = (QK_IQ6_K, IQ6_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ4_KS] = (QK_IQ4_KS, IQ4_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ2_KS] = (QK_IQ2_KS, IQ2_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ3_KS] = (QK_IQ3_KS, IQ3_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ5_KS] = (QK_IQ5_KS, IQ5_KS_BLOCK_BYTES)


_patch_gguf_enum()
