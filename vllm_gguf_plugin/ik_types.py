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
   140 = GGML_TYPE_IQ5_K  (block_iq5_k, 176 bytes/256 weights, 5.5 bpw)
   141 = GGML_TYPE_IQ6_K  (block_iq6_k, 212 bytes/256 weights, 6.625 bpw)
   144 = GGML_TYPE_IQ4_KS (block_iq4_ks, 136 bytes/256 weights + 4B row-prefix
         FP32 = 140 on-disk bytes/row, 4.25 bpw)

All use QK_K=256 super-blocks with software dequant and a dual-codebook
lookup via the `extra` field. IQ4_KS has a row-prefix FP32 scale.
"""

import gguf
from gguf.quants import GGMLQuantizationType

# ik_llama.cpp K-variant tensor type IDs
GGML_TYPE_IQ1_BN = 134
GGML_TYPE_IQ2_BN = 135
GGML_TYPE_I2_S = 36
GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q1_0_G128 = 41
GGML_TYPE_Q6_0 = 133
GGML_TYPE_IQ2_K = 137
GGML_TYPE_IQ3_K = 138
GGML_TYPE_IQ4_K = 139
GGML_TYPE_IQ5_K = 140
GGML_TYPE_IQ6_K = 141
GGML_TYPE_IQ4_KS = 144
GGML_TYPE_IQ2_KS = 145
GGML_TYPE_IQ4_KSS = 146
GGML_TYPE_IQ5_KS = 152
GGML_TYPE_IQ2_KT = 153
GGML_TYPE_IQ3_KT = 154
GGML_TYPE_IQ4_KT = 155
GGML_TYPE_IQ3_KS = 156
GGML_TYPE_IQ2_KL = 157
GGML_TYPE_IQ1_KT = 158

# BitNet block layouts. Row prefixes are stored separately from block payloads.
QK_IQ1BN = 64
IQ1_BN_BLOCK_BYTES = 13
QK_IQ2BN = 64
IQ2_BN_BLOCK_BYTES = 16
QK_I2S = 128
# Compatibility size for a 128-weight row; FP32 scale follows all packed row data.
I2_S_BLOCK_BYTES = 36
QK_Q1_0_G128 = 128
Q1_0_G128_BLOCK_BYTES = 18
QK_Q6_0 = 32
Q6_0_BLOCK_BYTES = 26

# Block layout constants (per 256-weight super-block)
QK_IQ2_K = 256
IQ2_K_BLOCK_BYTES = 76  # half(2) + uint16(2) + scales[8] + qs[64] = 76

QK_IQ3_K = 256
IQ3_K_BLOCK_BYTES = 110  # half(2) + uint16(2) + uint16(2) + scales_l[8]
# + qs[64] + qh[32] = 110

QK_IQ4_K = 256
IQ4_K_BLOCK_BYTES = 144  # half(2) + uint16(2) + scales_h[4] + scales_l[8]
# + qs[128] = 144

QK_IQ5_K = 256
IQ5_K_BLOCK_BYTES = 176

QK_IQ6_K = 256
IQ6_K_BLOCK_BYTES = 212

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

QK_IQ4_KSS = 256
IQ4_KSS_BLOCK_BYTES = 132  # float row prefix + 128-byte block

QK_IQ2_KL = 256
IQ2_KL_BLOCK_BYTES = 88  # half row prefix + 86-byte block

# Row-prefix storage geometry for ik_llama K-variant types.
#
# These types pack a per-row metadata field (FP32 or FP16 scale) before the
# super-block payloads, so the on-disk row stride is nonlinear in the block
# count:
#
#     row_size = row_meta_size + blocks_per_row * payload_size
#
# GGML_QUANT_SIZES cannot express this layout (it assumes
# ``bytes = blocks * type_size``), and the legacy shape formula
# ``qweight.shape[1] // type_size * block_size`` undercounts blocks whenever a
# row carries more than one super-block. The constants below expose the
# authoritative ik_llama.cpp ABI so loaders/materializers can compute the
# dequantized element count directly.
IQ4_KS_ROW_META_BYTES = 4  # float row prefix
IQ4_KS_PAYLOAD_BYTES = 136
IQ2_KS_ROW_META_BYTES = 2  # half row prefix
IQ2_KS_PAYLOAD_BYTES = 70
IQ3_KS_ROW_META_BYTES = 2  # half row prefix
IQ3_KS_PAYLOAD_BYTES = 102
IQ5_KS_ROW_META_BYTES = 4  # float row prefix
IQ5_KS_PAYLOAD_BYTES = 168
IQ4_KSS_ROW_META_BYTES = 4  # float row prefix
IQ4_KSS_PAYLOAD_BYTES = 128
IQ2_KL_ROW_META_BYTES = 2  # half row prefix
IQ2_KL_PAYLOAD_BYTES = 86

# type_id -> (row_meta_size, payload_size, super-block size).
ROW_PREFIX_GGUF_TYPES: dict[int, tuple[int, int, int]] = {
    GGML_TYPE_IQ4_KS: (IQ4_KS_ROW_META_BYTES, IQ4_KS_PAYLOAD_BYTES, QK_IQ4_KS),
    GGML_TYPE_IQ2_KS: (IQ2_KS_ROW_META_BYTES, IQ2_KS_PAYLOAD_BYTES, QK_IQ2_KS),
    GGML_TYPE_IQ3_KS: (IQ3_KS_ROW_META_BYTES, IQ3_KS_PAYLOAD_BYTES, QK_IQ3_KS),
    GGML_TYPE_IQ5_KS: (IQ5_KS_ROW_META_BYTES, IQ5_KS_PAYLOAD_BYTES, QK_IQ5_KS),
    GGML_TYPE_IQ4_KSS: (
        IQ4_KSS_ROW_META_BYTES,
        IQ4_KSS_PAYLOAD_BYTES,
        QK_IQ4_KSS,
    ),
    GGML_TYPE_IQ2_KL: (IQ2_KL_ROW_META_BYTES, IQ2_KL_PAYLOAD_BYTES, QK_IQ2_KL),
}

QK_IQ1_KT = 256
IQ1_KT_PAYLOAD_BYTES = 56
IQ1_KT_BLOCK_BYTES = 60

QK_IQ2_KT = 256
IQ2_KT_PAYLOAD_BYTES = 68
IQ2_KT_BLOCK_BYTES = 72

QK_IQ3_KT = 256
IQ3_KT_PAYLOAD_BYTES = 100
IQ3_KT_BLOCK_BYTES = 104

QK_IQ4_KT = 256
IQ4_KT_PAYLOAD_BYTES = 128
IQ4_KT_BLOCK_BYTES = 132

ROW_PREFIX_GGUF_TYPES.update(
    {
        GGML_TYPE_IQ1_KT: (4, IQ1_KT_PAYLOAD_BYTES, QK_IQ1_KT),
        GGML_TYPE_IQ2_KT: (4, IQ2_KT_PAYLOAD_BYTES, QK_IQ2_KT),
        GGML_TYPE_IQ3_KT: (4, IQ3_KT_PAYLOAD_BYTES, QK_IQ3_KT),
        GGML_TYPE_IQ4_KT: (4, IQ4_KT_PAYLOAD_BYTES, QK_IQ4_KT),
    }
)


def is_row_prefix_gguf_type(quant_type: int) -> bool:
    """True if *quant_type* uses an ik_llama row-prefix storage layout."""
    return quant_type in ROW_PREFIX_GGUF_TYPES


def gguf_qweight_dequant_shape(
    num_rows: int, row_bytes: int, quant_type: int
) -> tuple[int, int]:
    """Return the dequantized ``(rows, n)`` shape of a 2D GGUF qweight buffer.

    For ik_llama row-prefix types the per-row byte stride is
    ``row_meta_size + blocks_per_row * payload_size``; the standard
    ``GGML_QUANT_SIZES`` formula divides the prefix into every block and
    silently drops super-blocks, so dispatch on the row-prefix registry first.
    All other types resolve through ``GGML_QUANT_SIZES`` as before.
    """
    geometry = ROW_PREFIX_GGUF_TYPES.get(quant_type)
    if geometry is not None:
        row_meta, payload, qk = geometry
        blocks_per_row = (row_bytes - row_meta) // payload
        return num_rows, blocks_per_row * qk
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    return num_rows, row_bytes // type_size * block_size


def _patch_gguf_enum():
    """Add ik K-variant types to the gguf GGMLQuantizationType enum."""
    import enum

    q1_0_size = (QK_Q1_0_G128, Q1_0_G128_BLOCK_BYTES)
    existing_q1_0 = next(
        (member for member in GGMLQuantizationType if member.value == GGML_TYPE_Q1_0),
        None,
    )
    if existing_q1_0 is not None and existing_q1_0.name not in {
        "Q1_0",
        "Q1_0_G128",
    }:
        raise RuntimeError(
            f"gguf type ID 41 is already registered as {existing_q1_0.name}"
        )
    existing_q1_0_size = gguf.GGML_QUANT_SIZES.get(GGML_TYPE_Q1_0)
    if existing_q1_0_size is not None and existing_q1_0_size != q1_0_size:
        raise RuntimeError(
            f"gguf type ID 41 has incompatible geometry {existing_q1_0_size}; "
            f"expected {q1_0_size}"
        )

    ik_types = (
        ("IQ1_BN", GGML_TYPE_IQ1_BN),
        ("IQ2_BN", GGML_TYPE_IQ2_BN),
        ("I2_S", GGML_TYPE_I2_S),
        ("Q1_0", GGML_TYPE_Q1_0),
        ("Q1_0_G128", GGML_TYPE_Q1_0_G128),
        ("Q6_0", GGML_TYPE_Q6_0),
        ("IQ2_K", GGML_TYPE_IQ2_K),
        ("IQ3_K", GGML_TYPE_IQ3_K),
        ("IQ4_K", GGML_TYPE_IQ4_K),
        ("IQ5_K", GGML_TYPE_IQ5_K),
        ("IQ6_K", GGML_TYPE_IQ6_K),
        ("IQ4_KS", GGML_TYPE_IQ4_KS),
        ("IQ2_KS", GGML_TYPE_IQ2_KS),
        ("IQ3_KS", GGML_TYPE_IQ3_KS),
        ("IQ5_KS", GGML_TYPE_IQ5_KS),
        ("IQ4_KSS", GGML_TYPE_IQ4_KSS),
        ("IQ2_KL", GGML_TYPE_IQ2_KL),
        ("IQ1_KT", GGML_TYPE_IQ1_KT),
        ("IQ2_KT", GGML_TYPE_IQ2_KT),
        ("IQ3_KT", GGML_TYPE_IQ3_KT),
        ("IQ4_KT", GGML_TYPE_IQ4_KT),
    )
    for name, value in ik_types:
        existing_member = next(
            (m for m in GGMLQuantizationType if m.value == value and m.name != name),
            None,
        )
        if existing_member and existing_member.name != name and value != GGML_TYPE_Q1_0:
            raise RuntimeError(
                f"ik type ID {value} is already registered as {existing_member.name}"
            )

    if all(hasattr(GGMLQuantizationType, name) for name, _ in ik_types):
        gguf.GGML_QUANT_SIZES[GGML_TYPE_Q1_0] = q1_0_size
        gguf.GGML_QUANT_SIZES.pop(GGML_TYPE_IQ1_BN, None)
        gguf.GGML_QUANT_SIZES.pop(GGML_TYPE_IQ2_BN, None)
        return

    existing = {m.name: m.value for m in GGMLQuantizationType}
    existing.update(ik_types)

    new_enum = enum.IntEnum("GGMLQuantizationType", existing)

    gguf.quants.GGMLQuantizationType = new_enum
    gguf.GGMLQuantizationType = new_enum

    import sys

    for mod_name, mod in list(sys.modules.items()):
        if (
            mod_name
            and mod_name.startswith("gguf")
            and mod is not None
            and getattr(mod, "GGMLQuantizationType", None) is GGMLQuantizationType
        ):
            mod.GGMLQuantizationType = new_enum

    # BN rows have a prefix plus one payload per 64 weights.  GGUF's fixed
    # block geometry cannot express that layout, so leave these types out of
    # GGML_QUANT_SIZES rather than making tensor-size calculation silently
    # omit the prefix (the payload constants remain public above).
    gguf.GGML_QUANT_SIZES.pop(GGML_TYPE_IQ1_BN, None)
    gguf.GGML_QUANT_SIZES.pop(GGML_TYPE_IQ2_BN, None)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_I2_S] = (QK_I2S, I2_S_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q1_0] = q1_0_size
    gguf.GGML_QUANT_SIZES[GGML_TYPE_Q6_0] = (QK_Q6_0, Q6_0_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ2_K] = (QK_IQ2_K, IQ2_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ3_K] = (QK_IQ3_K, IQ3_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ4_K] = (QK_IQ4_K, IQ4_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ5_K] = (QK_IQ5_K, IQ5_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ6_K] = (QK_IQ6_K, IQ6_K_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ4_KS] = (QK_IQ4_KS, IQ4_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ2_KS] = (QK_IQ2_KS, IQ2_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ3_KS] = (QK_IQ3_KS, IQ3_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ5_KS] = (QK_IQ5_KS, IQ5_KS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ4_KSS] = (QK_IQ4_KSS, IQ4_KSS_BLOCK_BYTES)
    gguf.GGML_QUANT_SIZES[GGML_TYPE_IQ2_KL] = (QK_IQ2_KL, IQ2_KL_BLOCK_BYTES)


_patch_gguf_enum()
