"""Executable, arm-specific completion gates for Phase 0A.

Each blocked arm asserts the manifest faithfully reports the *absence* of
implementation/evidence: status stays ``blocked``, comparator and evidence
envelope are all null, blocking reason is concrete, and the fallback policy
is fail-closed.  A record that flips to ``passed`` without supplying real
comparator/evidence cannot pass these gates because the schema (and the
contract test module) rejects it; likewise a record that mislabels a
missing implementation as ``passed`` fails the assertions below.

Deferred arms assert the deferred contract: status ``deferred``, blocking
reason present, evidence envelope null, completion result state ``pending``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "validation" / "format-manifests"


def _load(arm_id: str) -> dict[str, Any]:
    path = MANIFESTS / f"{arm_id}.json"
    assert path.is_file(), f"missing manifest file for arm {arm_id}: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_ids() -> list[str]:
    return [
        path.stem
        for path in sorted(MANIFESTS.glob("*.json"))
        if path.name != "index.json"
    ]


ARM_IDS = _arm_ids()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "arm_id" in metafunc.fixturenames:
        metafunc.parametrize("arm_id", ARM_IDS, ids=ARM_IDS)


def test_arm_manifest_is_self_identifying(arm_id: str) -> None:
    record = _load(arm_id)
    execution = record["execution"]
    canonical = "-".join(
        (
            record["sample"]["sample_id"],
            execution["backend"],
            execution["architecture"],
            execution["operation"],
        )
    )
    assert record["arm_id"] == arm_id == canonical
    assert (MANIFESTS / f"{arm_id}.json").is_file()


def test_arm_enforces_blocked_or_deferred_contract(arm_id: str) -> None:
    record = _load(arm_id)
    status = record["status"]
    assert status in {"blocked", "deferred"}, (
        f"{arm_id}: Phase 0A currently has no passed arms; got status={status!r}"
    )

    assert record["blocking_reason"], f"{arm_id}: blocking_reason must be concrete"

    fallback = record["fallback_policy"]
    assert fallback["fail_closed"] is True, f"{arm_id}: fallback must be fail-closed"
    assert fallback["allowed_outside_envelope"] is False, (
        f"{arm_id}: blocked/deferred arms must not advertise open fallback"
    )
    if status == "blocked":
        assert fallback["mode"] == "blocked", (
            f"{arm_id}: blocked arms require fallback.mode == 'blocked'"
        )
    else:
        assert fallback["mode"] == "deferred", (
            f"{arm_id}: deferred arms require fallback.mode == 'deferred'"
        )

    # Phase 0A arms have no implementation evidence yet; every comparator and
    # every evidence sub-field (other than the completion_test identity) must
    # be null so a future author cannot mistake a null for "checked".
    for field, value in record["comparator"].items():
        assert value is None, f"{arm_id}: comparator.{field} must be null while unimplemented"

    evidence = record["evidence"]
    for field in (
        "dispatch_proof",
        "byte_accounting",
        "temporary_allocation",
        "occupancy_resources",
        "independent_reference",
        "architecture_selection",
    ):
        assert evidence[field] is None, (
            f"{arm_id}: evidence.{field} must be null while unimplemented"
        )

    completion = evidence["completion_test"]
    expected_state = "blocked" if status == "blocked" else "pending"
    assert completion["result_state"] == expected_state, (
        f"{arm_id}: result_state must agree with status ({expected_state})"
    )

    # Completion command must point at this exact parametrized test node.
    command = completion["command"]
    assert "tests/phase_0a/test_arm_records.py" in command, (
        f"{arm_id}: completion command must target test_arm_records.py"
    )
    expected_node = (
        "tests/phase_0a/test_arm_records.py::"
        f"test_arm_enforces_blocked_or_deferred_contract[{arm_id}]"
    )
    assert expected_node in command, (
        f"{arm_id}: completion command must select exact parametrized node"
    )

    # Assertion/expected_result must name the observable arm-specific outcome.
    assertion = completion["assertion"]
    assert arm_id in assertion, f"{arm_id}: assertion must name the arm"
    assert record["execution"]["operation"] in assertion, (
        f"{arm_id}: assertion must name the operation"
    )
    expected = completion["expected_result"].lower()
    if status == "blocked":
        assert "blocked" in expected and (
            "null" in expected or "no comparator" in expected
        ), (
            f"{arm_id}: expected_result must describe blocked/null evidence outcome"
        )
    else:
        assert "deferred" in expected and "pending" in expected, (
            f"{arm_id}: expected_result must describe deferred/pending outcome"
        )


def test_arm_recipe_and_layout_match_repository_policy(arm_id: str) -> None:
    record = _load(arm_id)
    sample_id = record["sample"]["sample_id"]
    backend = record["execution"]["backend"]
    recipe = record["recipe"]
    layout = record["quantization"]["layout"]

    if sample_id in {"nvfp4", "mxfp4"}:
        # Unknown ABI: every layout value and every numeric recipe field must be null.
        assert recipe["kind"] == "unknown", f"{arm_id}: unknown ABI kind expected"
        for field in (
            "values_per_block",
            "payload_bytes",
            "scale_bytes",
        ):
            assert recipe[field] is None, f"{arm_id}: recipe.{field} must be null"
        for field, value in layout.items():
            assert value is None, f"{arm_id}: layout.{field} must be null for unknown ABI"
        return

    if backend == "rocm_required":
        assert recipe["scale_bytes"] is not None and recipe["scale_bytes"] >= 0, (
            f"{arm_id}: ROCmFPX scale_bytes must be populated from rocmfpx_types.py"
        )
        assert layout["scale_bytes"] == recipe["scale_bytes"], (
            f"{arm_id}: layout.scale_bytes must agree with recipe.scale_bytes"
        )

    if record["quantization"]["ggml_type_id"] == 100:
        assert recipe["kind"] == "external_recipe", (
            f"{arm_id}: ggml_type_id 100 is a repository recipe, not a tensor ABI"
        )

    if sample_id in {"iq3-kt", "iq4-kt"}:
        # KT rows carry a row prefix; stride is shape-dependent, not a fixed constant.
        assert layout["row_prefix_bytes"] and layout["row_prefix_bytes"] > 0, (
            f"{arm_id}: KT row prefix must be non-zero"
        )
        assert layout["row_stride_bytes"] is None, (
            f"{arm_id}: KT row_stride_bytes must be null (shape-dependent)"
        )
        assert layout["row_stride_formula"] and "blocks_per_row" in layout["row_stride_formula"], (
            f"{arm_id}: KT row_stride_formula must encode multi-block semantics"
        )
