"""Freeze a complete detector-free H2 quality run for later scoring.

This is deliberately a narrow boundary between waveform-quality execution and
detector scoring.  It never imports, loads, or invokes a detector.  Instead it
hash-validates the immutable quality checkpoints against the committed
pre-score panel, requires all four pre-registered arms to clear their fixed
90% retained-pair gate, and writes a non-overwriting manifest containing only
passed pairs.  A downstream scorer must consume that manifest rather than the
mutable/diagnostic quality-run directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from src.h2_pre_score_pairs import (
    PRE_SCORE_MANIFEST_VERSION,
    QUALITY_ROW_VERSION,
    assert_score_independent_columns,
    quality_rows_frame,
)
from src.h2_quality_runner import (
    QUALITY_RUNNER_VERSION,
    RUN_PROVENANCE_KIND,
    ValidatedFrozenInputs,
    _all_expected_pairs,
    load_existing_checkpoints,
    sha256_file,
)


QUALITY_FREEZE_VERSION = "h2_quality_freeze_v1"
QUALITY_FREEZE_KIND = "h2_score_eligible_quality_freeze"
MINIMUM_ARM_RETAINED_FRACTION = 0.90
QUALITY_RUN_PROVENANCE_NAME = "quality_provenance.json"
QUALITY_RUN_SUMMARY_NAME = "quality_summary.json"
QUALITY_TABLE_NAME = "quality_pairs.parquet"
QUALITY_CHECKPOINT_DIRNAME = "pair_rows"
ELIGIBLE_CSV_NAME = "score_eligible_pairs.csv"
ELIGIBLE_PARQUET_NAME = "score_eligible_pairs.parquet"
FREEZE_REPORT_NAME = "quality_freeze.json"

_PAIR_ID_PATTERN = "0123456789abcdef"
_GATE_COLUMNS = (
    "pass_stoi",
    "pass_wer",
    "pass_loudness",
    "pass_clipping",
    "pass_target_direction",
)
_ROW_PROVENANCE_COLUMNS = (
    "quality_row_version",
    "dataset",
    "sample_id",
    "label",
    "source_id",
    "input_manifest_version",
    "source_input_csv_sha256",
    "selection_seed",
    "selection_method",
    "selection_key_sha256",
    "arm",
    "arm_definition_sha256",
    "arm_parameters_json",
    "target_feature",
    "target_direction",
    "negative_control",
    "detector_stage",
    "retained",
    "quality_status",
    "failure_reasons_json",
    "pair_id",
)
_PASSED_PAIR_REQUIRED_COLUMNS = (
    "sample_rate_hz",
    "original_samples",
    "transformed_samples",
    "original_wave_sha256",
    "transformed_wave_sha256",
    "stoi",
    "wer",
    "original_lufs",
    "transformed_lufs",
    "loudness_delta_lu",
    "original_clipping_fraction",
    "transformed_clipping_fraction",
    "added_clipping_fraction",
    "target_feature_before",
    "target_feature_after",
    "all_feature_deltas_json",
)


@dataclass(frozen=True)
class CompletedQualityRun:
    """A fully validated, complete detector-free H2 quality run."""

    run_dir: Path
    run_id: str
    validated_inputs: ValidatedFrozenInputs
    contract: Mapping[str, Any]
    rows: pd.DataFrame
    per_arm: Mapping[str, Mapping[str, Any]]
    quality_provenance_sha256: str
    quality_table_sha256: str
    checkpoint_aggregate_sha256: str


@dataclass(frozen=True)
class QualityFreezeOutputs:
    """Paths and compact provenance emitted by one score-eligibility freeze."""

    output_dir: Path
    csv_path: Path
    parquet_path: Path
    report_path: Path
    report: Mapping[str, Any]


def _json_safe(value: Any) -> Any:
    """Convert table scalars into strict, deterministic JSON-compatible data."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed H2 quality run lacks {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Completed H2 quality {description} must be a JSON object: {path}")
    return dict(value)


def _require_equal(observed: object, expected: object, *, field: str) -> None:
    if observed != expected:
        raise ValueError(f"H2 quality provenance mismatch for {field}: expected {expected!r}, observed {observed!r}")


def _expected_datasets(validated: ValidatedFrozenInputs) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "repo_id": value.get("repo_id"),
            "revision": value.get("revision"),
            "source_revision_resolved": value.get("source_revision_resolved"),
        }
        for name, value in sorted(validated.datasets.items())
    }


def _expected_arms(validated: ValidatedFrozenInputs) -> dict[str, str]:
    return {arm.arm_id: arm.as_record()["definition_sha256"] for arm in validated.arms}


def _validate_run_contract(
    contract: Mapping[str, Any],
    *,
    run_dir: Path,
    validated: ValidatedFrozenInputs,
) -> str:
    """Reject pilot, incomplete-contract, or provenance-mismatched runs."""
    _require_equal(contract.get("artifact_kind"), RUN_PROVENANCE_KIND, field="artifact_kind")
    _require_equal(contract.get("version"), QUALITY_RUNNER_VERSION, field="version")
    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("H2 quality provenance has no non-empty run_id")
    _require_equal(run_id, run_dir.name, field="run_id/run-directory name")
    mode = contract.get("mode")
    if mode == "pilot_not_panel_gate":
        raise ValueError("Pilot H2 quality runs are not eligible for a score freeze")
    _require_equal(mode, "full_panel_pre_score_quality", field="mode")
    if contract.get("limit_samples") is not None:
        raise ValueError("A bounded H2 quality run is not eligible for a score freeze")
    _require_equal(contract.get("detector_scoring_allowed"), False, field="detector_scoring_allowed")
    _require_equal(contract.get("panel_gate_eligible"), None, field="panel_gate_eligible")
    _require_equal(contract.get("manifest_version"), PRE_SCORE_MANIFEST_VERSION, field="manifest_version")
    _require_equal(contract.get("manifest_sha256"), validated.manifest_sha256, field="manifest_sha256")
    _require_equal(contract.get("arm_ledger_sha256"), validated.arm_ledger_sha256, field="arm_ledger_sha256")
    _require_equal(contract.get("arena_index_sha256"), validated.arena_index_sha256, field="arena_index_sha256")
    _require_equal(contract.get("arms"), _expected_arms(validated), field="arms")
    _require_equal(contract.get("datasets"), _expected_datasets(validated), field="datasets")
    return run_id


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in _PAIR_ID_PATTERN for character in value)


def _nonempty(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return not isinstance(value, str) or bool(value.strip())


def _canonical_parameters(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("H2 quality row arm_parameters_json must be a JSON object string")
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("H2 quality row arm_parameters_json must encode an object")
    return _canonical_json(dict(parsed))


def _validate_quality_row(
    row: Mapping[str, Any],
    *,
    manifest_row: Mapping[str, Any],
    arm: Any,
    pair_id: str,
) -> None:
    """Validate one checkpoint against its frozen source row and arm definition."""
    missing = [column for column in _ROW_PROVENANCE_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"H2 quality checkpoint {pair_id} lacks required fields: {missing}")
    _require_equal(row.get("pair_id"), pair_id, field=f"checkpoint {pair_id} pair_id")
    _require_equal(row.get("quality_row_version"), QUALITY_ROW_VERSION, field=f"checkpoint {pair_id} quality_row_version")
    _require_equal(row.get("detector_stage"), "blocked_pending_quality_freeze", field=f"checkpoint {pair_id} detector_stage")
    for column in (
        "dataset",
        "sample_id",
        "label",
        "source_id",
        "source_input_csv_sha256",
        "selection_seed",
        "selection_method",
        "selection_key_sha256",
    ):
        _require_equal(row.get(column), manifest_row.get(column), field=f"checkpoint {pair_id} {column}")
    _require_equal(
        row.get("input_manifest_version"),
        manifest_row.get("manifest_version"),
        field=f"checkpoint {pair_id} input_manifest_version",
    )
    record = arm.as_record()
    _require_equal(row.get("arm"), arm.arm_id, field=f"checkpoint {pair_id} arm")
    _require_equal(row.get("arm_definition_sha256"), record["definition_sha256"], field=f"checkpoint {pair_id} arm_definition_sha256")
    _require_equal(
        _canonical_parameters(row.get("arm_parameters_json")),
        _canonical_json(record["parameters"]),
        field=f"checkpoint {pair_id} arm_parameters_json",
    )
    _require_equal(row.get("target_feature"), arm.target_feature, field=f"checkpoint {pair_id} target_feature")
    _require_equal(row.get("target_direction"), arm.target_direction, field=f"checkpoint {pair_id} target_direction")
    _require_equal(bool(row.get("negative_control")), arm.negative_control, field=f"checkpoint {pair_id} negative_control")
    for column in _GATE_COLUMNS:
        if not isinstance(row.get(column), (bool, np.bool_)):
            raise ValueError(f"H2 quality checkpoint {pair_id} has a non-boolean {column}")
    retained = row.get("retained")
    if not isinstance(retained, (bool, np.bool_)):
        raise ValueError(f"H2 quality checkpoint {pair_id} has a non-boolean retained flag")
    expected_retained = all(bool(row[column]) for column in _GATE_COLUMNS)
    _require_equal(bool(retained), expected_retained, field=f"checkpoint {pair_id} retained/gates")
    expected_status = "passed" if expected_retained else "failed"
    _require_equal(row.get("quality_status"), expected_status, field=f"checkpoint {pair_id} quality_status")
    try:
        failures = json.loads(str(row.get("failure_reasons_json")))
    except json.JSONDecodeError as error:
        raise ValueError(f"H2 quality checkpoint {pair_id} has invalid failure_reasons_json") from error
    if not isinstance(failures, list):
        raise ValueError(f"H2 quality checkpoint {pair_id} failure_reasons_json must encode a list")
    if expected_retained and failures:
        raise ValueError(f"H2 quality checkpoint {pair_id} is marked retained but records failures")
    if expected_retained:
        missing_passed = [column for column in _PASSED_PAIR_REQUIRED_COLUMNS if not _nonempty(row.get(column))]
        if missing_passed:
            raise ValueError(f"Retained H2 quality checkpoint {pair_id} lacks score-routing fields: {missing_passed}")
        for column in ("original_wave_sha256", "transformed_wave_sha256"):
            if not _is_sha256(row.get(column)):
                raise ValueError(f"Retained H2 quality checkpoint {pair_id} has an invalid {column}")


def _canonical_pair_records(frame: pd.DataFrame) -> dict[str, str]:
    """Return one canonical JSON record per pair ID, rejecting duplicate IDs."""
    records: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        pair_id = row.get("pair_id")
        if not _is_sha256(pair_id):
            raise ValueError("H2 quality table contains an invalid pair_id")
        canonical = _canonical_json(row)
        if pair_id in records:
            raise ValueError(f"H2 quality table has duplicate pair_id {pair_id}")
        records[str(pair_id)] = canonical
    return records


def _validate_materialized_table(table_path: Path, checkpoint_table: pd.DataFrame) -> str:
    """Require the derived Parquet table to exactly mirror immutable checkpoints."""
    if not table_path.is_file():
        raise FileNotFoundError(f"Completed H2 quality run lacks materialized quality table: {table_path}")
    table = pd.read_parquet(table_path)
    table = quality_rows_frame(table.to_dict(orient="records"))
    source_records = _canonical_pair_records(checkpoint_table)
    table_records = _canonical_pair_records(table)
    if table_records != source_records:
        raise ValueError("Materialized H2 quality table does not exactly match immutable pair checkpoints")
    return sha256_file(table_path)


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    run_id: str,
    expected_pairs: int,
    per_arm: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_equal(summary.get("artifact_kind"), "h2_detector_free_quality_summary", field="quality_summary artifact_kind")
    _require_equal(summary.get("version"), QUALITY_RUNNER_VERSION, field="quality_summary version")
    _require_equal(summary.get("run_id"), run_id, field="quality_summary run_id")
    _require_equal(summary.get("mode"), "full_panel_pre_score_quality", field="quality_summary mode")
    _require_equal(summary.get("detector_scoring_allowed"), False, field="quality_summary detector_scoring_allowed")
    _require_equal(summary.get("panel_gate_status"), "not_frozen", field="quality_summary panel_gate_status")
    _require_equal(summary.get("expected_pairs"), expected_pairs, field="quality_summary expected_pairs")
    _require_equal(summary.get("completed_pairs"), expected_pairs, field="quality_summary completed_pairs")
    _require_equal(summary.get("remaining_pairs"), 0, field="quality_summary remaining_pairs")
    _require_equal(summary.get("complete"), True, field="quality_summary complete")
    _require_equal(summary.get("provenance_sha256"), _sha256_text(_canonical_json(contract)), field="quality_summary provenance_sha256")
    observed_per_arm = summary.get("per_arm")
    if not isinstance(observed_per_arm, Mapping):
        raise ValueError("H2 quality summary lacks per_arm data")
    if set(observed_per_arm) != set(per_arm):
        raise ValueError("H2 quality summary arm set differs from the frozen arm ledger")
    for arm_id, expected in per_arm.items():
        observed = observed_per_arm[arm_id]
        if not isinstance(observed, Mapping):
            raise ValueError(f"H2 quality summary arm {arm_id} is not an object")
        _require_equal(observed.get("completed_pairs"), expected["completed_pairs"], field=f"quality_summary {arm_id} completed_pairs")
        _require_equal(observed.get("retained_pairs"), expected["retained_pairs"], field=f"quality_summary {arm_id} retained_pairs")


def validate_completed_quality_run(
    run_dir: str | Path,
    *,
    validated_inputs: ValidatedFrozenInputs,
) -> CompletedQualityRun:
    """Strictly validate a complete, detector-free H2 quality-run directory.

    The source directory is never modified.  In particular, a completed run
    must contain every expected checkpoint, including explicit failed pairs;
    retaining only successful rows before this validation is prohibited.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"H2 quality run directory does not exist: {root}")
    contract_path = root / QUALITY_RUN_PROVENANCE_NAME
    summary_path = root / QUALITY_RUN_SUMMARY_NAME
    contract = _read_json_object(contract_path, description="quality provenance")
    run_id = _validate_run_contract(contract, run_dir=root, validated=validated_inputs)
    expected = _all_expected_pairs(validated_inputs.manifest, validated_inputs.arms)
    rows_by_pair = load_existing_checkpoints(root / QUALITY_CHECKPOINT_DIRNAME, expected)
    expected_ids = set(expected)
    observed_ids = set(rows_by_pair)
    if observed_ids != expected_ids:
        missing = len(expected_ids.difference(observed_ids))
        unexpected = len(observed_ids.difference(expected_ids))
        raise ValueError(
            "H2 quality run is incomplete or outside its frozen panel: "
            f"expected={len(expected_ids)}, observed={len(observed_ids)}, missing={missing}, unexpected={unexpected}"
        )
    for pair_id, row in rows_by_pair.items():
        manifest_row, arm = expected[pair_id]
        _validate_quality_row(row, manifest_row=manifest_row, arm=arm, pair_id=pair_id)
    checkpoint_table = quality_rows_frame(list(rows_by_pair.values()))
    table_sha256 = _validate_materialized_table(root / QUALITY_TABLE_NAME, checkpoint_table)
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in validated_inputs.arms:
        arm_rows = checkpoint_table.loc[checkpoint_table["arm"].eq(arm.arm_id)]
        expected_arm_pairs = len(validated_inputs.manifest)
        if len(arm_rows) != expected_arm_pairs:
            raise ValueError(
                f"H2 quality arm {arm.arm_id} has {len(arm_rows)} rows, expected exactly {expected_arm_pairs}"
            )
        retained_pairs = int(arm_rows["retained"].sum())
        retained_fraction = retained_pairs / expected_arm_pairs
        if retained_fraction < MINIMUM_ARM_RETAINED_FRACTION:
            raise ValueError(
                f"H2 quality arm {arm.arm_id} fails the predeclared {MINIMUM_ARM_RETAINED_FRACTION:.0%} retained-pair gate: "
                f"{retained_pairs}/{expected_arm_pairs} ({retained_fraction:.3f})"
            )
        per_arm[arm.arm_id] = {
            "expected_pairs": expected_arm_pairs,
            "completed_pairs": int(len(arm_rows)),
            "retained_pairs": retained_pairs,
            "retained_fraction": retained_fraction,
            "minimum_retained_fraction": MINIMUM_ARM_RETAINED_FRACTION,
            "gate_status": "passed",
        }
    summary = _read_json_object(summary_path, description="quality summary")
    _validate_summary(summary, contract=contract, run_id=run_id, expected_pairs=len(expected), per_arm=per_arm)
    checkpoint_hashes = [
        {"pair_id": pair_id, "checkpoint_sha256": sha256_file(root / QUALITY_CHECKPOINT_DIRNAME / f"{pair_id}.json")}
        for pair_id in sorted(rows_by_pair)
    ]
    return CompletedQualityRun(
        run_dir=root,
        run_id=run_id,
        validated_inputs=validated_inputs,
        contract=contract,
        rows=checkpoint_table,
        per_arm=per_arm,
        quality_provenance_sha256=sha256_file(contract_path),
        quality_table_sha256=table_sha256,
        checkpoint_aggregate_sha256=_sha256_text(_canonical_json(checkpoint_hashes)),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _eligible_rows(completed: CompletedQualityRun) -> pd.DataFrame:
    """Project only quality-passed pairs, retaining waveform-routing provenance."""
    eligible = completed.rows.loc[completed.rows["retained"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("No H2 pairs are score-eligible despite passing arm-level gates")
    if not eligible["quality_status"].eq("passed").all():
        raise ValueError("A non-passed quality row cannot enter the score-eligible manifest")
    if not eligible.loc[:, list(_GATE_COLUMNS)].all(axis=None):
        raise ValueError("A quality row with a failed gate cannot enter the score-eligible manifest")
    if not eligible["failure_reasons_json"].map(lambda value: json.loads(str(value)) == []).all():
        raise ValueError("A quality row with recorded failures cannot enter the score-eligible manifest")
    eligible["detector_stage"] = "eligible_after_quality_freeze"
    assert_score_independent_columns(eligible.columns)
    return eligible.sort_values(["dataset", "label", "sample_id", "arm"], kind="stable").reset_index(drop=True)


def write_score_eligible_freeze(
    completed: CompletedQualityRun,
    output_dir: str | Path,
) -> QualityFreezeOutputs:
    """Write a retained-pair-only, non-overwriting downstream scoring contract."""
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite H2 quality-freeze directory: {target}")
    eligible = _eligible_rows(completed)
    manifest_records = [json.loads(value) for value in _canonical_pair_records(eligible).values()]
    manifest_logical_sha256 = _sha256_text(_canonical_json(manifest_records))
    report: dict[str, Any] = {
        "artifact_kind": QUALITY_FREEZE_KIND,
        "version": QUALITY_FREEZE_VERSION,
        "freeze_status": "score_eligible",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_guard": "This permits detector scoring only for listed quality-passed pairs; it is not a detector result or causal estimate.",
        "detector_scores_read": False,
        "detector_scoring_performed": False,
        "detector_scoring_allowed": True,
        "quality_run_id": completed.run_id,
        "quality_run_directory": str(completed.run_dir),
        "input_manifest_sha256": completed.validated_inputs.manifest_sha256,
        "arm_ledger_sha256": completed.validated_inputs.arm_ledger_sha256,
        "arena_index_sha256": completed.validated_inputs.arena_index_sha256,
        "quality_provenance_sha256": completed.quality_provenance_sha256,
        "quality_table_sha256": completed.quality_table_sha256,
        "quality_checkpoint_aggregate_sha256": completed.checkpoint_aggregate_sha256,
        "predeclared_minimum_retained_fraction": MINIMUM_ARM_RETAINED_FRACTION,
        "expected_pairs": int(len(completed.rows)),
        "score_eligible_pairs": int(len(eligible)),
        "per_arm": completed.per_arm,
        "source_quality_detector_stage": "blocked_pending_quality_freeze",
        "eligible_detector_stage": "eligible_after_quality_freeze",
        "score_eligible_manifest_logical_sha256": manifest_logical_sha256,
        "score_eligible_manifest_columns": eligible.columns.tolist(),
    }
    target.mkdir(parents=True, exist_ok=False)
    csv_path = target / ELIGIBLE_CSV_NAME
    parquet_path = target / ELIGIBLE_PARQUET_NAME
    report_path = target / FREEZE_REPORT_NAME
    try:
        _atomic_write_text(csv_path, eligible.to_csv(index=False))
        _atomic_write_parquet(parquet_path, eligible)
        # The byte hashes let a downstream scorer verify the exact files it
        # reads; the logical hash above remains stable across CSV formatting.
        report["score_eligible_manifest_sha256"] = sha256_file(csv_path)
        report["score_eligible_parquet_sha256"] = sha256_file(parquet_path)
        _atomic_write_text(report_path, json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except Exception:
        # The directory deliberately remains for forensic inspection rather than
        # being silently deleted; a future attempt must use a new output path.
        raise
    return QualityFreezeOutputs(
        output_dir=target,
        csv_path=csv_path,
        parquet_path=parquet_path,
        report_path=report_path,
        report=report,
    )
