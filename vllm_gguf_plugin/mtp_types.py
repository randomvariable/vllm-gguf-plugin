"""Runtime registration of GGUF multi-token-prediction (MTP) tensor mappings.

llama.cpp declares ``nextn.*`` tensors -- the GGUF encoding of a multi-token
prediction head -- for a growing list of architectures. The published ``gguf``
Python package lags that list, so an MTP export produced by a recent llama.cpp
build carries tensors the installed package cannot name.

The gap is narrow. ``gguf.TensorNameMap.block_mappings_cfg`` already contains
every ``nextn.*`` pattern; ``TensorNameMap.__init__`` simply skips any tensor
that is not listed in ``MODEL_TENSORS[arch]``. So for an architecture the
installed package already knows, adding the nextn tensors to that list is
enough to make the mapping resolve -- no new patterns, no new name formats.

Without this, ``weights_adapter/default.py`` silently fails to resolve MTP
weights and the loader reports them as unmapped parameters.

This mirrors :func:`vllm_gguf_plugin.ik_types._patch_gguf_enum`, which patches
ik_llama quantization types into the same package for the same reason.

Architectures that the installed ``gguf`` release does not know at all cannot
be reached this way; :func:`unsupported_mtp_architectures` reports those so the
gap stays visible instead of looking like missing model support.
"""

from __future__ import annotations

import gguf
from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES, MODEL_TENSOR, MODEL_TENSORS

# Architectures that declare nextn tensors in llama.cpp master
# (gguf-py/gguf/constants.py, MODEL_TENSORS). Kept as GGUF architecture strings
# rather than MODEL_ARCH members because several are absent from released gguf
# packages, so the enum members may not exist locally.
NEXTN_ARCH_NAMES: tuple[str, ...] = (
    "bailingmoe2",
    "cohere2moe",
    "deepseek2",
    "deepseek32",
    "deepseek4",
    "exaone-moe",
    "exaone4",
    "gemma4-assistant",
    "glm-dsa",
    "glm4",
    "glm4moe",
    "hy_v3",
    "mimo2",
    "qwen35",
    "qwen35moe",
    "qwen3next",
    "step35",
)


def _nextn_tensors() -> list[MODEL_TENSOR]:
    """Nextn tensors the installed gguf package knows about.

    Released packages carry the six block-level tensors. Newer llama.cpp also
    defines non-block projections; those are included automatically when the
    installed package defines them.
    """
    return [tensor for tensor in MODEL_TENSOR if tensor.name.startswith("NEXTN_")]


def _arch_by_name(name: str) -> MODEL_ARCH | None:
    for arch in MODEL_ARCH:
        if MODEL_ARCH_NAMES[arch] == name:
            return arch
    return None


def unsupported_mtp_architectures() -> tuple[str, ...]:
    """Architectures with upstream MTP support that this gguf release cannot express.

    These have no ``MODEL_ARCH`` member locally, so no runtime patch can add
    their tensors; loading one of their MTP exports requires a newer ``gguf``.
    """
    return tuple(name for name in NEXTN_ARCH_NAMES if _arch_by_name(name) is None)


def patch_gguf_nextn_architectures() -> None:
    """Register nextn tensors for architectures the installed gguf omits.

    Idempotent, and additive only: an architecture that already declares the
    tensors is left untouched, and architectures without an MTP head are never
    given one.
    """
    nextn = _nextn_tensors()
    if not nextn:
        # No nextn tensors at all: the installed package predates MTP support
        # entirely, and adding names it cannot format would break the mapping.
        return

    for name in NEXTN_ARCH_NAMES:
        arch = _arch_by_name(name)
        if arch is None:
            continue
        tensors = MODEL_TENSORS.get(arch)
        if tensors is None:
            continue
        missing = [tensor for tensor in nextn if tensor not in tensors]
        if missing:
            MODEL_TENSORS[arch] = list(tensors) + missing


patch_gguf_nextn_architectures()

# gguf.MODEL_TENSORS is re-exported from gguf.constants; keep the alias in sync
# so callers reaching it through either path observe the patch.
gguf.MODEL_TENSORS = MODEL_TENSORS
