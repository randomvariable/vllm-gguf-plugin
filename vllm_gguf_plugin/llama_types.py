# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""llama.cpp additional quant type registration.

Type IDs:
  34 = GGML_TYPE_TQ1_0 (block_tq1_0, 54 bytes/256 weights, 1.6875 bpw)
  35 = GGML_TYPE_TQ2_0 (block_tq2_0, 66 bytes/256 weights, 2.0625 bpw)
  42 = GGML_TYPE_Q2_0  (block_q2_0, 18 bytes/64 weights, 2.25 bpw)
"""

import enum
import sys

import gguf
from gguf.quants import GGMLQuantizationType

GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q1_0_G128 = 41
GGML_TYPE_Q2_0 = 42

GGML_QUANT_SIZES = {
    GGML_TYPE_TQ1_0: (256, 54),
    GGML_TYPE_TQ2_0: (256, 66),
    GGML_TYPE_Q1_0: (128, 18),
    GGML_TYPE_Q2_0: (64, 18),
}

Q1_0_GEOMETRY = GGML_QUANT_SIZES[GGML_TYPE_Q1_0]


def _patch_gguf_enum() -> None:
    llama_types = (
        ("TQ1_0", GGML_TYPE_TQ1_0),
        ("TQ2_0", GGML_TYPE_TQ2_0),
        ("Q1_0", GGML_TYPE_Q1_0),
        ("Q1_0_G128", GGML_TYPE_Q1_0_G128),
        ("Q2_0", GGML_TYPE_Q2_0),
    )
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
    if existing_q1_0_size is not None and existing_q1_0_size != Q1_0_GEOMETRY:
        raise RuntimeError(
            f"gguf type ID 41 has incompatible geometry {existing_q1_0_size}; "
            f"expected {Q1_0_GEOMETRY}"
        )
    if all(hasattr(GGMLQuantizationType, name) for name, _ in llama_types):
        # Native gguf-py already defines these; verify they match CUDA switches.
        assert GGMLQuantizationType.TQ1_0 == GGML_TYPE_TQ1_0
        assert GGMLQuantizationType.TQ2_0 == GGML_TYPE_TQ2_0
        assert GGMLQuantizationType.Q1_0 == GGML_TYPE_Q1_0
        assert GGMLQuantizationType.Q1_0_G128 == GGML_TYPE_Q1_0_G128
        assert GGMLQuantizationType.Q2_0 == GGML_TYPE_Q2_0
        gguf.GGML_QUANT_SIZES.update(GGML_QUANT_SIZES)
        return

    # Iterate __members__ so aliases (e.g. Q1_0_G128 aliasing Q1_0) survive the
    # rebuild; plain enum iteration yields canonical members only.
    existing = {
        name: member.value for name, member in GGMLQuantizationType.__members__.items()
    }
    existing.update(llama_types)
    new_enum = enum.IntEnum("GGMLQuantizationType", existing)

    gguf.quants.GGMLQuantizationType = new_enum
    gguf.GGMLQuantizationType = new_enum
    for mod_name, mod in list(sys.modules.items()):
        if (
            mod_name
            and mod_name.startswith("gguf")
            and mod is not None
            and getattr(mod, "GGMLQuantizationType", None) is GGMLQuantizationType
        ):
            mod.GGMLQuantizationType = new_enum

    gguf.GGML_QUANT_SIZES.update(GGML_QUANT_SIZES)


_patch_gguf_enum()
