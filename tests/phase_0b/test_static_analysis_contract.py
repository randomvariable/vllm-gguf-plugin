"""Phase 0B.1 static-analysis artifact contract tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from validation.phase_0b.canonical import (
    artifact_hash,
    canonical_json,
    validate_artifact,
)

ROOT = Path(__file__).parents[2]
SCHEMA = json.loads(
    (ROOT / "validation/phase_0b/static-analysis.schema.json").read_text()
)


def _location(symbol: str = "plugin.dispatch.rocmfpx") -> dict:
    return {"file": "src/dispatch.py", "line": 42, "column": 5, "symbol_id": symbol}


def _artifact(side: str = "plugin", kind: str = "rocmfpx_model_author_fork") -> dict:
    return {
        "schema_version": "0B.1",
        "artifact_id": f"{side}-rocmfpx-dense-linear",
        "artifact_hash": None,
        "side": side,
        "comparator": {"kind": kind, "artifact_id": f"{side}-comparator"},
        "provenance": {
            "repo_url": "https://github.com/example/repo",
            "commit_sha": "0123456789abcdef0123456789abcdef01234567",
            "source_tree_sha256": "a" * 64,
            "extractor_version": "phase-0b-static/0.1",
        },
        "envelope": {
            "envelope_id": f"{side}-envelope-rocmfpx-dense-linear",
            "side": side,
            "format": "Q4_0_ROCMFP4",
            "operation": "dense_linear",
            "dtype": "float16",
            "architecture": "gfx1151",
            "sharding": {"tensor_parallel": 1, "expert_parallel": 1},
            "moe": {"enabled": False, "experts": None},
            "mtp": {"enabled": False, "layers": None},
        },
        "arm": {
            "arm_id": f"{side}-rocmfpx-dense-linear",
            "envelope_id": f"{side}-envelope-rocmfpx-dense-linear",
            "side": side,
            "decision": {
                "selected_symbol_id": "plugin.dispatch.rocmfpx",
                "selected_symbol_location": _location(),
                "terminal": "selected",
                "fail_closed": True,
            },
        },
        "facts": {
            "dispatch": [{"decision": "select rocmfpx path", "location": _location()}],
            "storage_reads": [
                {
                    "kind": "packed_weight",
                    "bytes": "ceil(K / 32) * 16",
                    "location": _location("plugin.storage.weight"),
                },
                {
                    "kind": "companion_scale",
                    "bytes": "ceil(K / 16) * 1",
                    "location": _location("plugin.storage.scale"),
                },
            ],
            "scale_flow": [
                {
                    "operation": "apply UE4M3 scale once",
                    "location": _location("plugin.scale.apply"),
                }
            ],
            "byte_formulas": [
                {
                    "name": "packed_weight_bytes",
                    "formula": "M * ceil(K / 32) * 16",
                    "location": _location("plugin.bytes.weight"),
                }
            ],
            "allocations": [
                {
                    "kind": "dense_weight_temporary",
                    "created": False,
                    "bytes": None,
                    "location": _location("plugin.alloc.none"),
                }
            ],
            "mapping": [
                {
                    "kind": "tensor_parallel",
                    "mapping": "row shard maps packed blocks",
                    "location": _location("plugin.mapping.tp"),
                }
            ],
        },
        "compiler_resources": {
            "registers_per_thread": {
                "value": None,
                "reason": "not available from static extractor",
            },
            "shared_memory_bytes": {
                "value": None,
                "reason": "not available from static extractor",
            },
        },
        "evidence": {
            "state": "captured",
            "limitations": ["Static code path only"],
            "evidence": ["source inspection"],
        },
    }


@pytest.fixture(params=["rocmfpx", "nvfp4"])
def valid_artifact(request):
    artifact = _artifact(
        "plugin" if request.param == "rocmfpx" else "comparator",
        "rocmfpx_model_author_fork"
        if request.param == "rocmfpx"
        else "nvfp4_upstream_llama_cpp",
    )
    artifact["envelope"]["format"] = (
        "NVFP4" if request.param == "nvfp4" else "Q4_0_ROCMFP4"
    )
    return artifact


def test_valid_fixtures_match_strict_schema(valid_artifact):
    Draft202012Validator.check_schema(SCHEMA)
    errors = list(Draft202012Validator(SCHEMA).iter_errors(valid_artifact))
    assert errors == []
    validate_artifact(valid_artifact)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda a: a["provenance"].update(commit_sha="short"),
        lambda a: a.pop("envelope"),
        lambda a: a["facts"]["byte_formulas"][0].pop("formula"),
        lambda a: a["facts"]["dispatch"][0].pop("location"),
        lambda a: a["envelope"].update(
            side="comparator" if a["side"] == "plugin" else "plugin"
        ),
        lambda a: a["evidence"].update(state="passed"),
        lambda a: a.update(comparator={"kind": "radv_runtime"}),
        lambda a: a["compiler_resources"].update(unknown={"value": 1}),
    ],
)
def test_mutations_are_rejected(valid_artifact, mutator):
    mutant = copy.deepcopy(valid_artifact)
    mutator(mutant)
    with pytest.raises(ValueError):
        validate_artifact(mutant)


@pytest.mark.parametrize(
    "term",
    [
        "latency",
        "throughput",
        "bandwidth_gbps",
        "tokens_per_second",
        "benchmark",
        "rocprof",
        "nsys",
        "nsight",
        "supported",
        "optimized",
        "non_inferior",
    ],
)
def test_forbidden_runtime_and_status_terms_are_rejected(valid_artifact, term):
    mutant = copy.deepcopy(valid_artifact)
    mutant["evidence"]["limitations"] = [term]
    with pytest.raises(ValueError):
        validate_artifact(mutant)


def test_canonical_json_and_hash_are_deterministic(valid_artifact):
    first = canonical_json(valid_artifact)
    shuffled = json.loads(json.dumps(valid_artifact, sort_keys=True))
    second = canonical_json(shuffled)
    assert first == second
    assert artifact_hash(valid_artifact) == artifact_hash(shuffled)
    assert json.loads(first)["artifact_hash"] is None
