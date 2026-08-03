"""Canonical serialization and validation for Phase 0B.1 artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "phase_0b" / "static-analysis.schema.json"
FORBIDDEN = {
    "latency",
    "throughput",
    "bandwidth_gbps",
    "tokens_per_second",
    "benchmark",
    "rocprof",
    "nsys",
    "nsight",
    "passed",
    "supported",
    "optimized",
    "non_inferior",
}


def canonical_json(artifact: dict[str, Any]) -> str:
    value = dict(artifact)
    value["artifact_hash"] = None
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )


def artifact_hash(artifact: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_forbidden(k) or _contains_forbidden(v) for k, v in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden(v) for v in value)
    if isinstance(value, str):
        return any(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
                value,
                re.IGNORECASE,
            )
            for term in FORBIDDEN
        )
    return False


def validate_artifact(artifact: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    candidate = dict(artifact)
    # Phase 0B.1 fixtures validate before their hash is assigned; emitted
    # artifacts still require the non-null schema value below.
    prehash = candidate.get("artifact_hash") is None
    if prehash:
        candidate["artifact_hash"] = artifact_hash(candidate)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(candidate), key=lambda e: list(e.path)
    )
    if errors:
        raise ValueError("; ".join(error.message for error in errors))
    if _contains_forbidden(candidate):
        raise ValueError("artifact contains forbidden runtime or status term")
    if (
        artifact["arm"]["side"] != artifact["side"]
        or artifact["envelope"]["side"] != artifact["side"]
    ):
        raise ValueError("arm and envelope side must match artifact side")
    decision = artifact["arm"]["decision"]
    if (
        decision["terminal"] in {"blocked", "unsupported"}
        and not decision["fail_closed"]
    ):
        raise ValueError("blocked or unsupported artifacts must fail closed")
    if not prehash and artifact["artifact_hash"] != artifact_hash(artifact):
        raise ValueError("artifact_hash does not match canonical artifact content")
