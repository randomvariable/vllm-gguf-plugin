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
GGML_TYPE_Q2_0 = 42

GGML_QUANT_SIZES = {
    GGML_TYPE_TQ1_0: (256, 54),
    GGML_TYPE_TQ2_0: (256, 66),
    GGML_TYPE_Q2_0: (64, 18),
}


def _patch_gguf_enum() -> None:
    llama_types = (
        ("TQ1_0", GGML_TYPE_TQ1_0),
        ("TQ2_0", GGML_TYPE_TQ2_0),
        ("Q2_0", GGML_TYPE_Q2_0),
    )
    if all(hasattr(GGMLQuantizationType, name) for name, _ in llama_types):
        # Native gguf-py already defines these; verify they match CUDA switches.
        assert GGMLQuantizationType.TQ1_0 == GGML_TYPE_TQ1_0
        assert GGMLQuantizationType.TQ2_0 == GGML_TYPE_TQ2_0
        assert GGMLQuantizationType.Q2_0 == GGML_TYPE_Q2_0
        gguf.GGML_QUANT_SIZES.update(GGML_QUANT_SIZES)
        return

    existing = {member.name: member.value for member in GGMLQuantizationType}
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
