"""Phase 0B.2 plugin ROCmFPX dense-baseline static path-map gates."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from validation.phase_0b.canonical import artifact_hash, validate_artifact

ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "validation" / "phase_0b" / "plugin_dense_baseline"
SOURCE_FILES = (
    "vllm_gguf_plugin/plugin.py",
    "vllm_gguf_plugin/quantization/diffusion_config.py",
    "vllm_gguf_plugin/quantization/fused_moe.py",
    "vllm_gguf_plugin/quantization/linear.py",
    "vllm_gguf_plugin/rocmfpx_types.py",
    "vllm_gguf_plugin/triton/dequantize/interface.py",
    "vllm_gguf_plugin/weights_adapter/diffusion/loader.py",
)


def _source_tree_sha256(
    source_bytes: dict[str, bytes] | None = None,
) -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_FILES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            (source_bytes or {}).get(relative_path, (ROOT / relative_path).read_bytes())
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_map(artifact_id: str) -> dict[str, Any]:
    path = MAP_DIR / f"{artifact_id}.json"
    assert path.is_file(), f"missing Phase 0B.2 plugin path map: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _locations(artifact: dict[str, Any]) -> set[tuple[str, int, int, str]]:
    locations: set[tuple[str, int, int, str]] = set()
    for entries in artifact["facts"].values():
        for entry in entries:
            location = entry["location"]
            locations.add(
                (
                    location["file"],
                    location["line"],
                    location["column"],
                    location["symbol_id"],
                )
            )
    location = artifact["arm"]["decision"]["selected_symbol_location"]
    locations.add(
        (location["file"], location["line"], location["column"], location["symbol_id"])
    )
    return locations


def _source_symbols(relative_path: str) -> dict[str, tuple[int, int]]:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    symbols: dict[str, tuple[int, int]] = {}

    def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified_name = f"{prefix}.{node.name}" if prefix else node.name
                symbols[qualified_name] = (node.lineno, node.end_lineno or node.lineno)
                visit(node.body, qualified_name)

    visit(tree.body)
    return symbols


def _map_sources(artifact: dict[str, Any]) -> dict[str, str]:
    return artifact["provenance"]["sources"]


def _semantic(artifact: dict[str, Any]) -> dict[str, Any]:
    return artifact["semantic"]


def _location(file: str, line: int, symbol_id: str) -> dict[str, Any]:
    return {"file": file, "line": line, "column": 1, "symbol_id": symbol_id}


def _source_hash(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _patch_speculator_passthrough_action() -> str:
    tree = ast.parse((ROOT / "vllm_gguf_plugin/plugin.py").read_text(encoding="utf-8"))
    patch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_speculator_probe"
    )
    wrapper = next(
        node
        for node in patch.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "maybe_override_with_speculators"
        )
    )
    branch = next(node for node in wrapper.body if isinstance(node, ast.If))
    passthrough = next(node for node in branch.body if isinstance(node, ast.Return))
    assert passthrough.value is not None
    return f"return {ast.unparse(passthrough.value)}"


@pytest.fixture(scope="module")
def maps() -> dict[str, dict[str, Any]]:
    artifact_ids = (
        "plugin-rocmfpx-dense-linear-baseline",
        "plugin-rocmfpx-diffusion-dense-baseline",
        "plugin-rocmfpx-moe-fallback-baseline",
        "plugin-rocmfpx-mtp-probe-passthrough-baseline",
    )
    return {artifact_id: _load_map(artifact_id) for artifact_id in artifact_ids}


def test_path_maps_have_exact_plugin_checkout_provenance(
    maps: dict[str, dict[str, Any]],
) -> None:
    """Artifacts must describe the source they were captured from.

    ``source_tree_sha256`` is the integrity gate: it is content-addressed over
    SOURCE_FILES, so any edit to a recorded authority invalidates the capture
    and forces a re-run.

    ``commit_sha`` is provenance context, not an integrity check. It is only
    required to name a real commit reachable from HEAD -- requiring equality
    with HEAD would be circular, since committing an artifact necessarily
    advances HEAD past the value recorded inside it.
    """
    exact = {
        "repo_url": _git_output("remote", "get-url", "origin"),
        "source_tree_sha256": _source_tree_sha256(),
        "extractor_version": "phase-0b-static/0.2",
    }
    for artifact in maps.values():
        validate_artifact(artifact)
        assert artifact["side"] == "plugin"
        assert artifact["comparator"]["kind"] == "rocmfpx_model_author_fork"
        for key, value in exact.items():
            assert artifact["provenance"][key] == value

        commit_sha = artifact["provenance"]["commit_sha"]
        assert re.fullmatch(r"[0-9a-f]{40}", commit_sha), commit_sha
        assert (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit_sha, "HEAD"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ), f"provenance commit {commit_sha} is not an ancestor of HEAD"


@pytest.mark.parametrize(
    "authority",
    (
        "vllm_gguf_plugin/rocmfpx_types.py",
        "vllm_gguf_plugin/triton/dequantize/interface.py",
    ),
)
def test_source_tree_hash_captures_rocmfpx_geometry_authority_mutations(
    maps: dict[str, dict[str, Any]], authority: str
) -> None:
    baseline = _source_tree_sha256()
    mutated_bytes = (ROOT / authority).read_bytes() + b"\n# phase-0b mutation\n"
    mutated_tree = _source_tree_sha256({authority: mutated_bytes})

    assert authority in SOURCE_FILES
    assert mutated_tree != baseline
    for artifact in maps.values():
        assert artifact["provenance"]["source_tree_sha256"] == baseline
        mutated_artifact = deepcopy(artifact)
        mutated_artifact["provenance"]["source_tree_sha256"] = mutated_tree
        with pytest.raises(ValueError, match="artifact_hash"):
            validate_artifact(mutated_artifact)


def test_diffusion_type_101_geometry_and_authorities(
    maps: dict[str, dict[str, Any]],
) -> None:
    artifact = maps["plugin-rocmfpx-diffusion-dense-baseline"]
    facts = artifact["facts"]
    sources = _map_sources(artifact)

    assert artifact["envelope"]["format"] == "Q4_0_ROCMFP4_FAST"
    assert {item["formula"] for item in facts["byte_formulas"]} == {
        "M * ceil(K / 32) * 17"
    }
    assert {item["bytes"] for item in facts["storage_reads"]} == {
        "M * ceil(K / 32) * 16",
        "M * ceil(K / 32) * 1",
    }
    assert "18" not in json.dumps(facts)
    assert "2 scale" not in json.dumps(facts).lower()
    for source in (
        "vllm_gguf_plugin/rocmfpx_types.py",
        "vllm_gguf_plugin/triton/dequantize/interface.py",
    ):
        assert sources[source] == _source_hash(source)


def test_diffusion_cpu_loader_materializes_persistent_dense_model_weight(
    maps: dict[str, dict[str, Any]],
) -> None:
    artifact = maps["plugin-rocmfpx-diffusion-dense-baseline"]
    semantic = _semantic(artifact)

    assert artifact["arm"]["decision"]["selected_symbol_location"] == _location(
        "vllm_gguf_plugin/weights_adapter/diffusion/loader.py",
        141,
        "_dense_weight_from_gguf_qweight",
    )
    assert semantic["route"] == {
        "quant_type": 101,
        "predicate": "not qweight.is_cuda",
        "callee": "ggml_dequantize_triton",
        "output_dtype": "float32",
    }
    assert semantic["type_100_fallback"] == {
        "quant_type": 100,
        "callee": "gguf.dequantize",
        "location": _location(
            "vllm_gguf_plugin/weights_adapter/diffusion/loader.py",
            142,
            "_dense_weight_from_gguf_qweight",
        ),
    }
    assert semantic["allocations"] == [
        {
            "location": _location(
                "vllm_gguf_plugin/weights_adapter/diffusion/loader.py",
                176,
                "_gguf_weights_for_loadable_names",
            ),
            "classification": "dense_materialization",
            "lifetime": "dense_model_weight",
        }
    ]
    assert artifact["facts"]["allocations"] == [
        {
            "bytes": "M * K * sizeof(dtype)",
            "created": True,
            "kind": "dense_materialization",
            "location": _location(
                "vllm_gguf_plugin/weights_adapter/diffusion/loader.py",
                176,
                "_gguf_weights_for_loadable_names",
            ),
        }
    ]


def test_moe_map_accounts_for_both_expert_dequantizations_and_intermediates(
    maps: dict[str, dict[str, Any]],
) -> None:
    artifact = maps["plugin-rocmfpx-moe-fallback-baseline"]
    semantic_allocations = _semantic(artifact)["allocations"]

    assert {
        item["location"]["line"]
        for item in artifact["facts"]["allocations"]
        if item["kind"] == "dense_weight_temporary" and item["created"]
    } == {183, 185}
    assert {
        (item["location"]["line"], item["classification"])
        for item in semantic_allocations
    } >= {
        (183, "dense_materialization"),
        (184, "activation"),
        (185, "dense_materialization"),
        (189, "output_accumulation"),
    }


def test_mtp_map_records_probe_passthrough_without_allocations(
    maps: dict[str, dict[str, Any]],
) -> None:
    artifact = maps["plugin-rocmfpx-mtp-probe-passthrough-baseline"]
    semantic = _semantic(artifact)
    passthrough = _patch_speculator_passthrough_action()

    assert artifact["artifact_id"] == "plugin-rocmfpx-mtp-probe-passthrough-baseline"
    assert artifact["arm"]["arm_id"] == (
        "plugin-rocmfpx-mtp-probe-passthrough-baseline-arm"
    )
    assert artifact["envelope"]["envelope_id"] == (
        "plugin-rocmfpx-mtp-probe-passthrough-baseline-envelope"
    )
    assert artifact["arm"]["decision"]["terminal"] == "selected"
    assert artifact["arm"]["decision"]["fail_closed"] is False
    assert semantic["route"] == {
        "predicate": "_is_gguf_reference(model)",
        "action": passthrough,
    }
    assert "unsupported_reason" not in semantic
    assert artifact["facts"]["allocations"] == []
    assert semantic["allocations"] == []


def test_maps_account_for_python_contiguous_and_cat_allocations(
    maps: dict[str, dict[str, Any]],
) -> None:
    expected_locations = {
        ("vllm_gguf_plugin/quantization/linear.py", 318, 1, "GGUFLinearMethod.apply"),
        ("vllm_gguf_plugin/quantization/linear.py", 321, 1, "GGUFLinearMethod.apply"),
        (
            "vllm_gguf_plugin/quantization/diffusion_config.py",
            69,
            1,
            "DiffusionGGUFLinearMethod.apply",
        ),
        (
            "vllm_gguf_plugin/quantization/diffusion_config.py",
            72,
            1,
            "DiffusionGGUFLinearMethod.apply",
        ),
    }
    found = set().union(*(_locations(artifact) for artifact in maps.values()))
    assert expected_locations <= found


def test_artifact_hashes_are_non_null_and_verify_canonical_content(
    maps: dict[str, dict[str, Any]],
) -> None:
    for artifact in maps.values():
        assert artifact["artifact_hash"] is not None
        assert artifact["artifact_hash"] == artifact_hash(artifact)


def test_provenance_hashes_every_referenced_source(
    maps: dict[str, dict[str, Any]],
) -> None:
    for artifact in maps.values():
        sources = _map_sources(artifact)
        referenced_files = {location[0] for location in _locations(artifact)}
        assert referenced_files <= sources.keys()
        for relative_path, expected_hash in sources.items():
            assert (ROOT / relative_path).is_file()
            assert expected_hash == _source_hash(relative_path)


def test_locations_resolve_to_real_symbols_and_filename_matches_artifact_id(
    maps: dict[str, dict[str, Any]],
) -> None:
    for artifact_id, artifact in maps.items():
        assert artifact["artifact_id"] == artifact_id
        for relative_path, line, _column, symbol_id in _locations(artifact):
            start, end = _source_symbols(relative_path)[symbol_id]
            assert start <= line <= end


def test_progress_captured_state_requires_corrected_path_maps(
    maps: dict[str, dict[str, Any]],
) -> None:
    progress = (ROOT / ".slim/deepwork/phase-0b-progress.md").read_text(
        encoding="utf-8"
    )
    diffusion = maps["plugin-rocmfpx-diffusion-dense-baseline"]
    mtp = maps["plugin-rocmfpx-mtp-probe-passthrough-baseline"]
    captured = "Status: CAPTURED" in progress
    corrected = (
        diffusion["arm"]["decision"]["selected_symbol_location"]
        == _location(
            "vllm_gguf_plugin/weights_adapter/diffusion/loader.py",
            141,
            "_dense_weight_from_gguf_qweight",
        )
        and mtp["arm"]["decision"]["terminal"] == "selected"
        and all(
            artifact["evidence"]["state"] == "captured" for artifact in maps.values()
        )
    )
    assert "## Phase 0B.2" in progress
    assert not captured or corrected
    for artifact in maps.values():
        assert artifact["artifact_id"] in progress
