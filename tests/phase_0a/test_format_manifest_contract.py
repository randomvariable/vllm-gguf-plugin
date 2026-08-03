"""Phase 0A manifest contract gates for all sample-model acceptance arms.

These tests are the authoritative Phase 0A gate. They enforce:

* JSON Schema Draft 2020-12 validity for every arm record.
* Fail-closed fallback and status-dependent evidence semantics at the
  schema level (via mutation tests that prove the schema rejects open
  fallback states).
* Exact one-record-per-arm file linkage, canonical arm IDs, and
  ``claim.file == '{arm_id}.json'`` agreement with no missing or extra
  claims.
* Inventory label *and* source URL linkage against
  ``.slim/deepwork/tests.txt``.
* Mandatory ROCmFPX gfx1151 matrix (seven samples x dense_linear/moe/mtp).
* Mandatory Blackwell NVFP4 six-arm matrix (sm120, sm121 x
  dense_linear/moe/mtp). No scope escape hatch.
* ABI discipline: ROCmFPX scale bytes populated, type-100 recipe is
  ``external_recipe``, NVFP4/MXFP4 layouts all null, KT row stride is
  shape-dependent, no RADV support claims.
"""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "validation" / "format-manifests"
SCHEMA_PATH = REPO_ROOT / "validation" / "format-manifest.schema.json"
INDEX_PATH = MANIFEST_DIR / "index.json"
INVENTORY_PATH = REPO_ROOT / ".slim" / "deepwork" / "tests.txt"
ROCMFPX_SAMPLE_IDS = {
    "q4-0-rocmfp4-coherent",
    "q4-0-rocmfp4-fast",
    "q8-0-rocmfpx-agent",
    "q6-0-rocmfpx-strix-quality",
    "q8-0-rocmfpx-strix-agent",
    "q4-0-rocmfp4-strix-lean",
    "q3-0-rocmfpx",
}
OPERATIONS = {"dense_linear", "moe", "mtp"}
BLACKWELL_ARCHITECTURES = {"sm120", "sm121"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return _load_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.fixture(scope="module")
def index() -> dict[str, Any]:
    return _load_json(INDEX_PATH)


@pytest.fixture(scope="module")
def records() -> dict[str, dict[str, Any]]:
    return {
        path.stem: _load_json(path)
        for path in sorted(MANIFEST_DIR.glob("*.json"))
        if path.name != INDEX_PATH.name
    }


@pytest.fixture(scope="module")
def blocked_record(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = next(
        record
        for record in records.values()
        if record["status"] == "blocked"
        and record["execution"]["backend"] == "rocm_required"
    )
    return copy.deepcopy(candidate)


@pytest.fixture(scope="module")
def nvfp4_blocked_record(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = next(
        record
        for record in records.values()
        if record["sample"]["sample_id"] == "nvfp4"
        and record["status"] == "blocked"
    )
    return copy.deepcopy(candidate)


def _inventory() -> dict[int, tuple[str, str]]:
    entries: dict[int, tuple[str, str]] = {}
    for line_number, line in enumerate(
        INVENTORY_PATH.read_text(encoding="utf-8").splitlines(), 1
    ):
        name, source = line.split(": ", maxsplit=1)
        entries[line_number] = (name, source)
    return entries


def _canonical_inventory(index: dict[str, Any]) -> dict[int, dict[str, str]]:
    """Return explicit line -> label/sample mapping from the index artifact."""
    return {
        int(entry["inventory_line"]): entry
        for entry in index["inventory"]
    }


def _canonical_arm_id(record: dict[str, Any]) -> str:
    execution = record["execution"]
    return "-".join(
        (
            record["sample"]["sample_id"],
            execution["backend"],
            execution["architecture"],
            execution["operation"],
        )
    )


def _pytest_target(command: str) -> Path | None:
    parts = shlex.split(command)
    try:
        pytest_index = next(index for index, part in enumerate(parts) if part == "pytest")
    except StopIteration:
        return None
    for part in parts[pytest_index + 1 :]:
        if not part.startswith("-"):
            return REPO_ROOT / part.split("::", maxsplit=1)[0]
    return None


# ---------------------------------------------------------------------------
# JSON Schema validity
# ---------------------------------------------------------------------------


def test_all_arm_records_validate_against_current_json_schema(
    validator: Draft202012Validator, records: dict[str, dict[str, Any]]
) -> None:
    failures = {
        filename: sorted(error.message for error in validator.iter_errors(record))
        for filename, record in records.items()
    }
    assert not {filename: errors for filename, errors in failures.items() if errors}


# ---------------------------------------------------------------------------
# Fail-closed schema mutation tests
# ---------------------------------------------------------------------------


def test_schema_rejects_open_fallback_for_blocked_arm(
    validator: Draft202012Validator, blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(blocked_record)
    mutant["fallback_policy"]["fail_closed"] = False
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject fail_closed=false on a blocked arm"


def test_schema_rejects_envelope_escape_for_blocked_arm(
    validator: Draft202012Validator, blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(blocked_record)
    mutant["fallback_policy"]["allowed_outside_envelope"] = True
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject allowed_outside_envelope=true on a blocked arm"


def test_schema_rejects_non_blocked_fallback_mode_for_blocked_arm(
    validator: Draft202012Validator, blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(blocked_record)
    mutant["fallback_policy"]["mode"] = "cpu_reference"
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject fallback.mode != 'blocked' on a blocked arm"


def test_schema_rejects_pending_result_state_for_blocked_arm(
    validator: Draft202012Validator, blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(blocked_record)
    mutant["evidence"]["completion_test"]["result_state"] = "pending"
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject result_state=pending on a blocked arm"


def test_schema_rejects_passed_arm_with_open_fallback(
    validator: Draft202012Validator, blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(blocked_record)
    mutant["status"] = "passed"
    mutant["blocking_reason"] = None
    mutant["fallback_policy"]["mode"] = "blocked"
    mutant["evidence"]["completion_test"]["result_state"] = "pass"
    # Even with comparator/envelope populated, a passed arm may not keep a
    # blocked fallback mode.
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject passed arm with fallback.mode=blocked"


def test_schema_rejects_passed_arm_with_missing_evidence(
    validator: Draft202012Validator, blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(blocked_record)
    mutant["status"] = "passed"
    mutant["blocking_reason"] = None
    mutant["fallback_policy"]["mode"] = "cpu_reference"
    mutant["evidence"]["completion_test"]["result_state"] = "pass"
    # Comparator and evidence remain null: schema must reject.
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject passed arm with null comparator/evidence"


def test_schema_rejects_nvfp4_blocked_arm_with_deferred_fallback(
    validator: Draft202012Validator, nvfp4_blocked_record: dict[str, Any]
) -> None:
    mutant = copy.deepcopy(nvfp4_blocked_record)
    mutant["fallback_policy"]["mode"] = "deferred"
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject deferred fallback on a blocked NVFP4 arm"


def test_schema_rejects_open_fallback_for_deferred_arm(
    validator: Draft202012Validator, records: dict[str, dict[str, Any]]
) -> None:
    deferred = next(
        record for record in records.values() if record["status"] == "deferred"
    )
    mutant = copy.deepcopy(deferred)
    mutant["fallback_policy"]["fail_closed"] = False
    errors = list(validator.iter_errors(mutant))
    assert errors, "schema must reject fail_closed=false on a deferred arm"

def test_schema_rejects_wrong_fallback_mode_on_deferred_arm(
    validator: Draft202012Validator, records: dict[str, dict[str, Any]]
) -> None:
    deferred = next(record for record in records.values() if record["status"] == "deferred")
    mutant = copy.deepcopy(deferred)
    mutant["fallback_policy"]["mode"] = "blocked"
    assert list(validator.iter_errors(mutant)), "schema must reject blocked mode on deferred arm"


def test_schema_rejects_wrong_result_state_on_deferred_arm(
    validator: Draft202012Validator, records: dict[str, dict[str, Any]]
) -> None:
    deferred = next(record for record in records.values() if record["status"] == "deferred")
    mutant = copy.deepcopy(deferred)
    mutant["evidence"]["completion_test"]["result_state"] = "blocked"
    assert list(validator.iter_errors(mutant)), "schema must reject blocked result on deferred arm"


def test_schema_rejects_deferred_evidence_shape(
    validator: Draft202012Validator, records: dict[str, dict[str, Any]]
) -> None:
    deferred = next(record for record in records.values() if record["status"] == "deferred")
    mutant = copy.deepcopy(deferred)
    mutant["evidence"]["dispatch_proof"] = {"claimed": True}
    assert list(validator.iter_errors(mutant)), "schema must reject non-null deferred evidence"



# ---------------------------------------------------------------------------
# Index, file linkage, and canonical arm identity
# ---------------------------------------------------------------------------


def test_index_files_records_and_canonical_ids_are_exact(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    indexed = {claim["arm_id"]: claim for claim in index["claims"]}
    record_ids = {record["arm_id"] for record in records.values()}
    assert len(indexed) == len(index["claims"]), "index arm IDs must be unique"
    assert set(indexed) == record_ids, "index claims must match arm records exactly"
    assert len(records) == len(record_ids), "one arm record file per arm ID"

    for filename, record in records.items():
        arm_id = record["arm_id"]
        assert filename == arm_id, f"{arm_id} must be stored as {arm_id}.json"
        assert arm_id == _canonical_arm_id(record)
        claim = indexed[arm_id]
        assert claim["file"] == f"{arm_id}.json", (
            f"claim.file must equal '{arm_id}.json', got {claim.get('file')!r}"
        )
        assert (MANIFEST_DIR / claim["file"]).is_file(), (
            f"claim.file {claim['file']!r} does not exist on disk"
        )
        assert claim["sample_id"] == record["sample"]["sample_id"]
        assert claim["inventory_line"] == record["sample"]["inventory_line"]

    # No claim may name a file that does not correspond to an arm record.
    claimed_files = {claim["file"] for claim in index["claims"]}
    record_files = {f"{arm_id}.json" for arm_id in record_ids}
    assert claimed_files == record_files


def test_index_has_no_blackwell_scope_escape(index: dict[str, Any]) -> None:
    assert "blackwell_nvfp4_scope" not in index, (
        "Blackwell NVFP4 coverage is unconditional; no scope escape hatch permitted"
    )


def test_inventory_lines_and_sample_sources_are_fully_linked(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    inventory = _inventory()
    assert set(inventory) == set(range(1, 14))
    assert index["expected_inventory_lines"] == list(range(1, 14))

    canonical = _canonical_inventory(index)
    assert set(canonical) == set(range(1, 14))
    assert {entry["sample_id"] for entry in canonical.values()} == set(index["sample_ids"])

    for record in records.values():
        sample = record["sample"]
        inventory_label, source = inventory[sample["inventory_line"]]
        expected = canonical[sample["inventory_line"]]
        assert expected["label"] == inventory_label
        assert expected["source"] == source
        assert sample["sample_id"] == expected["sample_id"], (
            f"{record['arm_id']}: sample_id must match explicit index mapping"
        )
        # The manifest source URL must equal the inventory source URL exactly.
        assert sample["source"] == source, (
            f"{record['arm_id']}: sample.source does not match inventory line "
            f"{sample['inventory_line']} source {source!r}"
        )


def test_index_validator_rejects_wrong_file_link(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    """Regression: an index claim whose .file is wrong must trip the gate."""
    bad_claim = dict(index["claims"][0])
    arm_id = bad_claim["arm_id"]
    bad_claim["file"] = f"not-{arm_id}.json"
    bad_index = dict(index)
    bad_index = {**bad_index, "claims": [bad_claim, *index["claims"][1:]]}
    with pytest.raises(AssertionError):
        test_index_files_records_and_canonical_ids_are_exact(
            bad_index, records
        )


def test_index_validator_rejects_wrong_inventory_label(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    """Regression: a manifest whose sample_id drifts from inventory trips the gate."""
    target_arm = next(
        arm_id for arm_id in records
        if records[arm_id]["sample"]["inventory_line"] == 1
    )
    bad_records = {
        arm_id: copy.deepcopy(record) for arm_id, record in records.items()
    }
    bad_records[target_arm]["sample"]["sample_id"] = "wrong-sample-id"
    with pytest.raises(AssertionError):
        test_inventory_lines_and_sample_sources_are_fully_linked(index, bad_records)


def test_index_validator_rejects_tq1_sample_id_mutation(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    bad_index = copy.deepcopy(index)
    entry = next(item for item in bad_index["inventory"] if item["inventory_line"] == 8)
    entry["sample_id"] = "tq1-0-unrelated"
    with pytest.raises(AssertionError):
        test_inventory_lines_and_sample_sources_are_fully_linked(index=bad_index, records=records)


def test_index_validator_rejects_inventory_label_mutation(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    bad_index = copy.deepcopy(index)
    entry = next(item for item in bad_index["inventory"] if item["inventory_line"] == 8)
    entry["label"] = "NOT_TQ1"
    with pytest.raises(AssertionError):
        test_inventory_lines_and_sample_sources_are_fully_linked(index=bad_index, records=records)


def _exact_completion_target(command: str, arm_id: str) -> str:
    parts = shlex.split(command)
    target = (
        "tests/phase_0a/test_arm_records.py::"
        f"test_arm_enforces_blocked_or_deferred_contract[{arm_id}]"
    )
    expected_parts = [
        ".venv/bin/python",
        "-m",
        "pytest",
        target,
    ]
    assert parts == expected_parts, (
        f"{arm_id}: completion command must contain only the fixed launcher "
        f"and exact arm node {target!r}; got {parts!r}"
    )
    expected = (
        "tests/phase_0a/test_arm_records.py::"
        f"test_arm_enforces_blocked_or_deferred_contract[{arm_id}]"
    )
    assert target == expected, f"{arm_id}: expected exact pytest node {expected!r}, got {target!r}"
    assert (REPO_ROOT / target.split("::", maxsplit=1)[0]).is_file()
    return target


def test_completion_commands_are_exact_executable_arm_targets(
    records: dict[str, dict[str, Any]]
) -> None:
    for arm_id, record in records.items():
        target = _exact_completion_target(record["evidence"]["completion_test"]["command"], arm_id)
        result = subprocess.run(
            [".venv/bin/python", "-m", "pytest", "--collect-only", "-q", target],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{arm_id}: target collection failed: {result.stdout}{result.stderr}"
        assert target.split("::", maxsplit=1)[1] in result.stdout


def test_completion_parser_rejects_negative_arm_selection(
    records: dict[str, dict[str, Any]]
) -> None:
    arm_id = next(iter(records))
    with pytest.raises(AssertionError):
        _exact_completion_target(
            f".venv/bin/python -m pytest tests/phase_0a/test_arm_records.py -k 'not {arm_id}'",
            arm_id,
        )


def test_completion_parser_rejects_deselection(
    records: dict[str, dict[str, Any]]
) -> None:
    arm_id = next(iter(records))
    target = (
        "tests/phase_0a/test_arm_records.py::"
        f"test_arm_enforces_blocked_or_deferred_contract[{arm_id}]"
    )
    with pytest.raises(AssertionError):
        _exact_completion_target(
            f".venv/bin/python -m pytest {target} --deselect={target}", arm_id
        )


def test_completion_parser_rejects_extra_pytest_arguments(
    records: dict[str, dict[str, Any]]
) -> None:
    arm_id = next(iter(records))
    target = (
        "tests/phase_0a/test_arm_records.py::"
        f"test_arm_enforces_blocked_or_deferred_contract[{arm_id}]"
    )
    for extra in ("--collect-only", "--maxfail=1"):
        with pytest.raises(AssertionError):
            _exact_completion_target(
                f".venv/bin/python -m pytest {target} {extra}", arm_id
            )


# ---------------------------------------------------------------------------
# Mandatory matrices
# ---------------------------------------------------------------------------


def test_required_rocmfpx_matrix_covers_every_sample_and_operation(
    records: dict[str, dict[str, Any]]
) -> None:
    actual = {
        (record["sample"]["sample_id"], record["execution"]["operation"])
        for record in records.values()
        if record["execution"]["backend"] == "rocm_required"
        and record["execution"]["architecture"] == "gfx1151"
        and record["execution"]["requirement"] == "required"
    }
    expected = {
        (sample_id, operation)
        for sample_id in ROCMFPX_SAMPLE_IDS
        for operation in OPERATIONS
    }
    assert actual == expected


def test_blackwell_nvfp4_matrix_is_complete(
    records: dict[str, dict[str, Any]]
) -> None:
    actual = {
        (record["execution"]["architecture"], record["execution"]["operation"])
        for record in records.values()
        if record["sample"]["sample_id"] == "nvfp4"
        and record["execution"]["backend"] == "cuda_required"
        and record["execution"]["requirement"] == "required"
    }
    expected = {
        (architecture, operation)
        for architecture in BLACKWELL_ARCHITECTURES
        for operation in OPERATIONS
    }
    assert actual == expected, (
        "Blackwell NVFP4 matrix is unconditionally required; "
        "no scope escape hatch is permitted"
    )


def test_no_manifest_claims_radv_support(records: dict[str, dict[str, Any]]) -> None:
    assert "radv" not in json.dumps(records).lower()


# ---------------------------------------------------------------------------
# Status/result_state coherence
# ---------------------------------------------------------------------------


def test_status_and_completion_result_states_agree(
    records: dict[str, dict[str, Any]]
) -> None:
    for record in records.values():
        execution = record["execution"]
        result_state = record["evidence"]["completion_test"]["result_state"]
        if execution["requirement"] == "required":
            assert record["status"] not in {"pending", "deferred"}
        if record["status"] == "deferred":
            assert result_state == "pending"
        if record["status"] == "blocked":
            assert result_state == "blocked"


# ---------------------------------------------------------------------------
# ABI/layout policy
# ---------------------------------------------------------------------------


def _assert_rocmfpx_scale_bytes(record: dict[str, Any]) -> None:
    type_id = record["quantization"]["ggml_type_id"]
    expected = {100: 2, 101: 1, 102: 2, 103: 1, 104: 2}[type_id]
    assert record["recipe"]["scale_bytes"] == expected
    assert record["quantization"]["layout"]["scale_bytes"] == expected


def test_known_rocmfpx_scale_bytes_are_populated(
    records: dict[str, dict[str, Any]]
) -> None:
    for record in records.values():
        if record["execution"]["backend"] != "rocm_required":
            continue
        _assert_rocmfpx_scale_bytes(record)


def test_rocmfpx_scale_bytes_mutation_is_rejected(
    records: dict[str, dict[str, Any]]
) -> None:
    for record in records.values():
        if record["execution"]["backend"] != "rocm_required":
            continue
        mutant = copy.deepcopy(record)
        mutant["recipe"]["scale_bytes"] = 9
        mutant["quantization"]["layout"]["scale_bytes"] = 9
        with pytest.raises(AssertionError):
            _assert_rocmfpx_scale_bytes(mutant)


def test_unknown_nvfp4_and_mxfp4_layouts_are_all_null(
    records: dict[str, dict[str, Any]]
) -> None:
    for record in records.values():
        if record["sample"]["sample_id"] not in {"nvfp4", "mxfp4"}:
            continue
        assert record["recipe"]["kind"] == "unknown"
        assert all(value is None for value in record["quantization"]["layout"].values())


def test_rocmfpx_type_100_recipe_is_not_a_model_tensor_abi(
    records: dict[str, dict[str, Any]]
) -> None:
    type_100 = [
        record
        for record in records.values()
        if record["quantization"]["ggml_type_id"] == 100
    ]
    assert type_100
    assert all(record["recipe"]["kind"] != "tensor_abi" for record in type_100)


def test_kt_row_stride_is_shape_dependent(
    records: dict[str, dict[str, Any]]
) -> None:
    for record in records.values():
        if record["sample"]["sample_id"] not in {"iq3-kt", "iq4-kt"}:
            continue
        layout = record["quantization"]["layout"]
        assert layout["row_prefix_bytes"] and layout["row_prefix_bytes"] > 0
        assert layout["row_stride_bytes"] is None
        assert "blocks_per_row" in layout["row_stride_formula"]
