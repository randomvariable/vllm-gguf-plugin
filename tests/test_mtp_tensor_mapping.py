"""Contract coverage for GGUF multi-token-prediction (MTP) tensor mapping.

vLLM's native MTP reuses the target model's quantization: the MTP head is built
from ordinary ``LinearBase``/``RoutedExperts`` layers, so it loads through the
same ``GGUFLinearMethod``/``GGUFMoEMethod`` as the main model. There is no
MTP-specific quantization path.

What can still fail is *loading*. An MTP checkpoint only works if its tensor
names map between vLLM parameter names and GGUF tensor names, and that mapping
is per architecture. ``gguf.TensorNameMap`` already knows every ``nextn.*``
pattern, but only emits them for architectures listed in ``MODEL_TENSORS``.
The published ``gguf`` package lags llama.cpp master, so recent MTP exports
(DeepSeek, Qwen3-Next, Qwen3.5, ...) would otherwise fail to resolve.

``vllm_gguf_plugin.mtp_types`` patches the missing membership in, mirroring
``ik_types._patch_gguf_enum``. These tests pin that the patched mapping matches
llama.cpp master rather than whatever the installed ``gguf`` release shipped.
"""

from __future__ import annotations

import gguf
import pytest
from gguf import MODEL_ARCH, TensorNameMap
from gguf.constants import MODEL_ARCH_NAMES, MODEL_TENSOR, MODEL_TENSORS

from vllm_gguf_plugin.mtp_types import (
    NEXTN_ARCH_NAMES,
    unsupported_mtp_architectures,
)

# vLLM MTP modules expose these submodules. See the per-model mtp.py under
# vllm/models/<model>/<vendor>/ (deepseek_v4, kimi_k3, minimax_m3, ...) and the
# older vllm/model_executor/models/*_mtp.py. Only eh_proj and shared_head.head
# are quantized linears; enorm/hnorm are RMSNorm and embed_tokens is embedding.
VLLM_MTP_SUBMODULES = (
    "enorm",
    "hnorm",
    "eh_proj",
    "shared_head.head",
    "shared_head.norm",
    "embed_tokens",
)


def _nextn_tensors() -> set[MODEL_TENSOR]:
    return {tensor for tensor in MODEL_TENSOR if "NEXTN" in tensor.name}


def _arch_by_name(name: str) -> MODEL_ARCH | None:
    for arch in MODEL_ARCH:
        if MODEL_ARCH_NAMES[arch] == name:
            return arch
    return None


def _locally_known_nextn_archs() -> list[str]:
    return [name for name in NEXTN_ARCH_NAMES if _arch_by_name(name) is not None]


def test_patch_covers_every_locally_known_nextn_architecture() -> None:
    """Every architecture we can patch must actually carry nextn tensors.

    Without the patch only the handful of architectures baked into the
    installed gguf release resolve, which silently breaks MTP loading for
    recent DeepSeek and Qwen exports.
    """
    nextn = _nextn_tensors()
    for name in _locally_known_nextn_archs():
        arch = _arch_by_name(name)
        assert arch is not None
        assert nextn & set(MODEL_TENSORS[arch]), (
            f"{name}: gguf declares no nextn tensors, so an MTP checkpoint for "
            f"this architecture cannot be loaded"
        )


@pytest.mark.parametrize("arch_name", _locally_known_nextn_archs())
@pytest.mark.parametrize("submodule", VLLM_MTP_SUBMODULES)
def test_vllm_mtp_names_resolve_to_gguf_nextn_tensors(
    arch_name: str, submodule: str
) -> None:
    """Every vLLM MTP submodule must resolve to a GGUF nextn tensor.

    The loader calls TensorNameMap.get_name with the vLLM parameter name minus
    its .weight/.bias suffix, which is what is exercised here.
    """
    arch = _arch_by_name(arch_name)
    assert arch is not None
    resolved = TensorNameMap(arch, 64).get_name(f"model.layers.46.{submodule}")
    assert resolved is not None, (
        f"{arch_name}: vLLM MTP submodule {submodule!r} does not map to a GGUF "
        f"tensor name; an MTP checkpoint would fail to load"
    )
    assert resolved == f"blk.46.nextn.{submodule.replace('.', '_')}"


def test_recent_mtp_architectures_are_covered() -> None:
    """Guard the architectures that motivated the patch.

    These ship MTP heads and are present in the installed gguf release, but
    that release predates their nextn declarations upstream.
    """
    for name in ("deepseek2", "qwen3next", "qwen35moe"):
        arch = _arch_by_name(name)
        assert arch is not None, f"gguf has no architecture named {name!r}"
        resolved = TensorNameMap(arch, 64).get_name("model.layers.46.eh_proj")
        assert resolved == "blk.46.nextn.eh_proj"


def test_architectures_without_nextn_do_not_resolve_mtp_names() -> None:
    """Architectures with no MTP head must not resolve MTP names.

    The patch is additive; it must not make unrelated architectures appear to
    support MTP.
    """
    for name in ("llama", "qwen2", "phi3"):
        arch = _arch_by_name(name)
        if arch is None:
            continue
        name_map = TensorNameMap(arch, 64)
        for submodule in ("eh_proj", "shared_head.head"):
            assert name_map.get_name(f"model.layers.46.{submodule}") is None


def test_unsupported_architectures_are_reported_not_silently_dropped() -> None:
    """Architectures absent from the installed gguf must be surfaced.

    Some upstream MTP architectures have no MODEL_ARCH entry in the installed
    release, so no runtime patch can reach them; they need a gguf upgrade. That
    gap must be inspectable rather than looking like unsupported hardware.
    """
    unsupported = unsupported_mtp_architectures()
    assert isinstance(unsupported, tuple)
    for name in unsupported:
        assert _arch_by_name(name) is None
        assert name in NEXTN_ARCH_NAMES


def test_patch_is_idempotent() -> None:
    """Re-running the patch must not duplicate tensor entries."""
    from vllm_gguf_plugin import mtp_types

    arch = _arch_by_name("deepseek2")
    assert arch is not None
    before = list(MODEL_TENSORS[arch])
    mtp_types.patch_gguf_nextn_architectures()
    assert list(MODEL_TENSORS[arch]) == before


def test_plugin_resolves_weights_through_gguf_tensor_name_map() -> None:
    """Guard the assumption that ties these tests to the loader.

    The mapping asserted here only reflects real loader behaviour while the
    weights adapter keeps resolving names through gguf's TensorNameMap.
    """
    from vllm_gguf_plugin.weights_adapter import default

    source = default.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "get_name(" in text, (
        "weights adapter no longer resolves names via TensorNameMap.get_name; "
        "the MTP mapping tests need to be re-pointed at the new mechanism"
    )
    assert hasattr(gguf, "TensorNameMap")
