"""Validation and execution wrapper for the detector-free H2B Q1 calibration.

This module accepts the committed H2B Q0 score-blind manifest and a finite Q1
arm manifest.  It reuses the repository's resumable waveform/ASR quality
engine but supplies a separate HDD/repository namespace and never exposes a
detector callback or response field.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.h2_asr_wer import WhisperQualityGateConfig
from src.h2_pre_score_pairs import ArmDefinition, PRE_SCORE_MANIFEST_VERSION, assert_score_independent_columns
from src.h2_quality_runner import QualityRunPaths, ValidatedFrozenInputs, run_quality_panel, sha256_file


H2B_Q1_ARM_MANIFEST_VERSION = "h2b_q1_quality_arm_manifest_v1"
H2B_Q1_RUN_VERSION = "h2b_q1_quality_calibration_v1"
H2B_Q0_STAGE = "q0_frozen_score_blind"
ALLOWED_TRANSFORMS = {"endpoint_silence_fixed_length", "spectral_tilt", "allpass_phase", "gain", "polarity"}
ALLOWED_DIRECTIONS = {"increase", "decrease", "invariant", "absolute_change"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(source: str | Path) -> Mapping[str, Any]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Required H2B Q1 JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid H2B Q1 JSON artifact: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"H2B Q1 JSON artifact must be an object: {path}")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"H2B Q1 {field} must be a lowercase SHA-256")
    return text


def _load_q0_manifest(q0_manifest: str | Path, q0_provenance: str | Path) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    manifest_path = Path(q0_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"H2B Q0 manifest is missing: {manifest_path}")
    if manifest_path.suffix.casefold() != ".csv":
        raise ValueError("H2B Q0 manifest must be CSV")
    provenance = _read_json(q0_provenance)
    if provenance.get("artifact_kind") != "h2b_q0_score_blind_input_freeze":
        raise ValueError("H2B Q1 requires an H2B Q0 score-blind provenance artifact")
    if provenance.get("h2b_stage") != H2B_Q0_STAGE or provenance.get("detector_scoring_allowed") is not False:
        raise ValueError("H2B Q0 provenance does not preserve the detector-ineligible Q0 boundary")
    output = provenance.get("output_manifest")
    if not isinstance(output, Mapping):
        raise ValueError("H2B Q0 provenance lacks output-manifest hash")
    expected_hash = _require_sha256(output.get("sha256"), "Q0 manifest hash")
    observed_hash = sha256_file(manifest_path)
    if observed_hash != expected_hash:
        raise RuntimeError(f"H2B Q0 manifest SHA-256 mismatch: expected {expected_hash}, observed {observed_hash}")
    rows = pd.read_csv(manifest_path)
    assert_score_independent_columns(rows.columns)
    required = {
        "dataset",
        "sample_id",
        "label",
        "dataset_revision",
        "selection_key_sha256",
        "selection_rank",
        "manifest_version",
        "selection_seed",
        "selection_method",
        "source_input_csv_sha256",
        "h2b_stage",
        "h2b_manifest_version",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"H2B Q0 manifest lacks required columns: {missing}")
    if rows.empty or rows.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("H2B Q0 manifest is empty or has duplicate identities")
    if not rows["manifest_version"].eq(PRE_SCORE_MANIFEST_VERSION).all():
        raise ValueError("H2B Q0 manifest has an unexpected underlying selection marker")
    if not rows["h2b_stage"].eq(H2B_Q0_STAGE).all():
        raise ValueError("H2B Q0 manifest has an unexpected H2B stage")
    if not rows["selection_key_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("H2B Q0 manifest selection keys are invalid")
    return rows.sort_values(["dataset", "label", "selection_rank", "sample_id"], kind="stable").reset_index(drop=True), provenance


def _arm_from_record(record: Mapping[str, Any]) -> ArmDefinition:
    required = {
        "arm_id",
        "transform",
        "parameters",
        "negative_control",
        "target_feature",
        "target_direction",
        "target_tolerance",
        "definition_sha256",
    }
    missing, extra = sorted(required.difference(record)), sorted(set(record).difference(required))
    if missing or extra:
        raise ValueError(f"H2B Q1 arm record fields mismatch; missing={missing}, extra={extra}")
    if record["transform"] not in ALLOWED_TRANSFORMS:
        raise ValueError(f"H2B Q1 arm uses unsupported transform {record['transform']!r}")
    if record["target_direction"] not in ALLOWED_DIRECTIONS:
        raise ValueError(f"H2B Q1 arm uses unsupported target direction {record['target_direction']!r}")
    raw_parameters = record["parameters"]
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("H2B Q1 arm parameters must be an object")
    arm = ArmDefinition(
        arm_id=str(record["arm_id"]),
        transform=str(record["transform"]),
        parameters={str(key): float(value) for key, value in raw_parameters.items()},
        negative_control=bool(record["negative_control"]),
        target_feature=str(record["target_feature"]),
        target_direction=str(record["target_direction"]),
        target_tolerance=float(record["target_tolerance"]),
    )
    expected_hash = _require_sha256(record["definition_sha256"], f"arm {arm.arm_id} definition hash")
    if arm.as_record()["definition_sha256"] != expected_hash:
        raise ValueError(f"H2B Q1 arm definition hash mismatch for {arm.arm_id}")
    if not arm.arm_id or not arm.target_feature or arm.target_tolerance < 0.0:
        raise ValueError("H2B Q1 arm identifier, target feature, and tolerance must be valid")
    return arm


def _load_q1_arm_manifest(source: str | Path) -> tuple[tuple[ArmDefinition, ...], Mapping[str, Any]]:
    manifest = _read_json(source)
    if manifest.get("artifact_kind") != "h2b_q1_quality_arm_manifest" or manifest.get("version") != H2B_Q1_ARM_MANIFEST_VERSION:
        raise ValueError("Unexpected H2B Q1 arm manifest kind/version")
    if "score" in _canonical_json(manifest).casefold() and "claim_guard" not in manifest:
        raise ValueError("H2B Q1 arm manifest may not carry response-like content")
    raw_arms = manifest.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("H2B Q1 arm manifest requires a nonempty arms list")
    if not all(isinstance(record, Mapping) for record in raw_arms):
        raise ValueError("H2B Q1 arm manifest has a non-object arm")
    arms = tuple(sorted((_arm_from_record(record) for record in raw_arms), key=lambda arm: arm.arm_id))
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("H2B Q1 arm identifiers must be unique")

    controls = manifest.get("negative_control_arm_ids")
    if not isinstance(controls, list) or set(map(str, controls)) != {arm.arm_id for arm in arms if arm.negative_control}:
        raise ValueError("H2B Q1 negative-control list must exactly match negative-control arms")
    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("H2B Q1 arm manifest requires selection families")
    covered: set[str] = set()
    family_ids: set[str] = set()
    known = {arm.arm_id: arm for arm in arms}
    for family in families:
        if not isinstance(family, Mapping):
            raise ValueError("H2B Q1 family must be an object")
        family_id = str(family.get("family_id", ""))
        candidate_ids = family.get("candidate_arm_ids")
        minimum = family.get("minimum_abs_median_target_delta")
        target = str(family.get("target_feature", ""))
        if not family_id or family_id in family_ids or not isinstance(candidate_ids, list) or not candidate_ids:
            raise ValueError("H2B Q1 family identity/candidates are invalid")
        family_ids.add(family_id)
        if not isinstance(minimum, (int, float)) or float(minimum) <= 0.0:
            raise ValueError("H2B Q1 family minimum target change must be positive")
        for arm_id in map(str, candidate_ids):
            arm = known.get(arm_id)
            if arm is None or arm.negative_control or arm.target_feature != target or arm_id in covered:
                raise ValueError("H2B Q1 family has invalid, control, mismatched, or duplicated candidate arm")
            covered.add(arm_id)
    if covered != {arm.arm_id for arm in arms if not arm.negative_control}:
        raise ValueError("H2B Q1 families must cover every non-control arm exactly once")
    policy = manifest.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("H2B Q1 arm manifest lacks a selection policy")
    if policy.get("wilson_confidence_level") != 0.95 or policy.get("minimum_lower_wilson_retention_bound") != 0.9:
        raise ValueError("H2B Q1 selection policy must retain its locked Wilson thresholds")
    return arms, manifest


def h2b_q1_quality_paths(hdd_root: str | Path, run_id: str, repo_root: str | Path) -> QualityRunPaths:
    if not run_id or not all(character.isalnum() or character in "_-" for character in run_id):
        raise ValueError("H2B Q1 run ID must be nonempty and contain only letters, digits, underscores, or hyphens")
    hdd = Path(hdd_root).resolve()
    repo = Path(repo_root).resolve()
    root = hdd / "runs" / "h2b_quality_calibration" / run_id
    return QualityRunPaths(
        run_root=root,
        pair_rows_dir=root / "pair_rows",
        transcript_cache_dir=root / "original_transcripts",
        full_table_path=root / "quality_pairs.parquet",
        provenance_path=root / "quality_provenance.json",
        repo_summary_path=repo / "experiments" / "future_directions" / "results" / "q1_quality_runs" / f"{run_id}.summary.json",
    )


@contextmanager
def h2b_q1_run_lock(paths: QualityRunPaths, run_id: str):
    """Hold an exclusive per-run lock, preventing competing checkpoint writers.

    Q1 checkpoints are individually immutable, but concurrent ASR processes
    can still waste GPU time and race transcript-cache creation.  A lock is
    therefore required around the entire runner invocation.  The lock file is
    removed after release; it is never a scientific artifact.
    """
    paths.run_root.mkdir(parents=True, exist_ok=True)
    target = paths.run_root / ".h2b_q1_run.lock"
    handle = target.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"H2B Q1 run is already active: {paths.run_root}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"run_id={run_id}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            target.unlink(missing_ok=True)


def validate_h2b_q1_inputs(
    q0_manifest: str | Path,
    q0_provenance: str | Path,
    arm_manifest: str | Path,
    arena_index: str | Path,
) -> tuple[ValidatedFrozenInputs, Mapping[str, Any]]:
    """Validate committed Q0/Q1 inputs without decoding a waveform or loading ASR."""
    rows, q0 = _load_q0_manifest(q0_manifest, q0_provenance)
    arms, arm_document = _load_q1_arm_manifest(arm_manifest)
    source = q0.get("declared_source")
    if not isinstance(source, Mapping):
        raise ValueError("H2B Q0 provenance lacks declared source")
    dataset, revision = str(source.get("dataset", "")), str(source.get("revision", ""))
    if set(rows["dataset"].astype(str)) != {dataset} or set(rows["dataset_revision"].astype(str)) != {revision}:
        raise ValueError("H2B Q0 rows do not match their declared dataset/revision")
    index_path = Path(arena_index)
    if not index_path.is_file():
        raise FileNotFoundError(f"Arena index is missing: {index_path}")
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    datasets = index.get("datasets") if isinstance(index, Mapping) else None
    if not isinstance(datasets, Mapping) or dataset not in datasets or not isinstance(datasets[dataset], Mapping):
        raise ValueError("H2B Q1 declared dataset is absent from Arena index")
    entry = datasets[dataset]
    if entry.get("revision") != revision:
        raise ValueError("H2B Q1 declared source revision differs from Arena index")
    validated = ValidatedFrozenInputs(
        manifest=rows,
        arms=arms,
        datasets={dataset: entry},
        manifest_sha256=sha256_file(q0_manifest),
        arm_ledger_sha256=sha256_file(arm_manifest),
        arena_index_sha256=sha256_file(index_path),
    )
    info: Mapping[str, Any] = {
        "h2b_q1_version": H2B_Q1_RUN_VERSION,
        "q0_provenance_sha256": sha256_file(q0_provenance),
        "q1_arm_manifest_sha256": sha256_file(arm_manifest),
        "q1_arm_document_sha256": _sha256_json(arm_document),
        "declared_source": {"dataset": dataset, "revision": revision},
        "detector_scoring_allowed": False,
    }
    return validated, info


def run_h2b_q1_quality(
    validated: ValidatedFrozenInputs,
    *,
    paths: QualityRunPaths,
    run_id: str,
    q1_info: Mapping[str, Any],
    whisper_config: WhisperQualityGateConfig,
) -> Mapping[str, Any]:
    """Run all Q0 rows as a detector-free calibration, never a score panel."""
    raw = run_quality_panel(
        validated,
        paths=paths,
        run_id=run_id,
        mode="pilot_not_panel_gate",
        limit=len(validated.manifest),
        whisper_config=whisper_config,
    )
    summary = dict(raw)
    summary.update(
        {
            "artifact_kind": "h2b_q1_detector_free_quality_summary",
            "version": H2B_Q1_RUN_VERSION,
            "claim_guard": (
                "H2B Q1 waveform/ASR quality calibration only; no detector was imported, loaded, or run. "
                "This pilot cannot select a feature, score pairs, estimate EER, or establish causal sensitivity."
            ),
            "detector_scoring_allowed": False,
            "h2b_q1": dict(q1_info),
        }
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for target in (paths.run_root / "h2b_q1_quality_summary.json", paths.repo_summary_path):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
    return summary
