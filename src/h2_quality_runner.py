"""Resumable, detector-free H2 waveform-quality execution support.

This module is deliberately a *pre-score* layer.  It validates an immutable
input panel and registered arm ledger, streams the indexed source audio, runs
only waveform/quality measurements, and persists one immutable checkpoint per
input--arm pair.  It has no detector import, callback, model path, or response
column.  A separate, later protocol step must still freeze the completed
quality table before any detector can consume retained pairs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from src.arena_io import AudioRecord, decode_audio, iter_selected_audio
from src.h2_asr_wer import LazyWhisperTranscriber, WhisperQualityGateConfig
from src.h2_pre_score_pairs import (
    PRE_SCORE_MANIFEST_VERSION,
    QUALITY_ROW_VERSION,
    ArmDefinition,
    apply_registered_arm,
    assert_score_independent_columns,
    evaluate_quality_pair,
    quality_rows_frame,
    registered_crest_factor_arms,
    waveform_sha256,
)


QUALITY_RUNNER_VERSION = "h2_quality_runner_v1"
QUALITY_SEGMENT_KIND = "h2_quality_pair_checkpoint"
TRANSCRIPT_CACHE_KIND = "h2_original_transcript_cache"
RUN_PROVENANCE_KIND = "h2_detector_free_quality_run"
DEFAULT_HDD_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing")
REQUIRED_MANIFEST_COLUMNS = (
    "dataset",
    "sample_id",
    "source_id",
    "label",
    "dataset_revision",
    "selection_key_sha256",
    "selection_rank",
    "manifest_version",
    "selection_seed",
    "selection_method",
    "source_input_csv_sha256",
)
PAIR_FAILURE_FIELDS = (
    "pass_stoi",
    "pass_wer",
    "pass_loudness",
    "pass_clipping",
    "pass_target_direction",
)


class Transcriber(Protocol):
    """Minimal interface used by the quality layer; no detector is accepted."""

    def transcribe(self, audio: np.ndarray) -> str: ...


@dataclass(frozen=True)
class ValidatedFrozenInputs:
    """Validated immutable H2 selection inputs and their index entries."""

    manifest: pd.DataFrame
    arms: tuple[ArmDefinition, ...]
    datasets: Mapping[str, Mapping[str, Any]]
    manifest_sha256: str
    arm_ledger_sha256: str
    arena_index_sha256: str


@dataclass(frozen=True)
class QualityRunPaths:
    """All paths owned by one explicit H2 quality execution ID."""

    run_root: Path
    pair_rows_dir: Path
    transcript_cache_dir: Path
    full_table_path: Path
    provenance_path: Path
    repo_summary_path: Path


class CachedOriginalTranscriptError(RuntimeError):
    """A durable original-ASR failure replayed to every arm of that sample."""


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values and non-finite floats to strict JSON values."""
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


def _normalise_arm_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical arm record, including its content-addressed hash."""
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
    missing = sorted(required.difference(value))
    extra = sorted(set(value).difference(required))
    if missing or extra:
        raise ValueError(f"Arm ledger record has invalid fields; missing={missing}, extra={extra}")
    parameters = value["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("Arm ledger parameters must be an object")
    arm = ArmDefinition(
        arm_id=str(value["arm_id"]),
        transform=str(value["transform"]),
        parameters={str(key): float(item) for key, item in parameters.items()},
        negative_control=bool(value["negative_control"]),
        target_feature=str(value["target_feature"]),
        target_direction=str(value["target_direction"]),
        target_tolerance=float(value["target_tolerance"]),
    )
    record = arm.as_record()
    if str(value["definition_sha256"]) != record["definition_sha256"]:
        raise ValueError(f"Arm ledger hash mismatch for {arm.arm_id}")
    return record


def validate_arm_ledger(ledger: object) -> tuple[ArmDefinition, ...]:
    """Require exactly the four registered crest arms with exact content hashes."""
    if not isinstance(ledger, list):
        raise ValueError("H2 arm ledger must be a JSON list")
    supplied = [_normalise_arm_record(item) for item in ledger if isinstance(item, Mapping)]
    if len(supplied) != len(ledger):
        raise ValueError("H2 arm ledger contains a non-object arm record")
    expected = sorted((arm.as_record() for arm in registered_crest_factor_arms()), key=lambda item: item["arm_id"])
    supplied = sorted(supplied, key=lambda item: item["arm_id"])
    if supplied != expected:
        raise ValueError("H2 arm ledger must contain exactly the four registered crest-factor arms")
    return tuple(sorted(registered_crest_factor_arms(), key=lambda arm: arm.arm_id))


def _validate_manifest_frame(manifest: pd.DataFrame, arena_index: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Mapping[str, Any]]]:
    """Validate frozen markers, identities, and indexed dataset revisions."""
    assert_score_independent_columns(manifest.columns)
    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS).difference(manifest.columns))
    if missing:
        raise ValueError(f"Frozen H2 manifest lacks required columns: {missing}")
    if manifest.empty:
        raise ValueError("Frozen H2 manifest is empty")
    result = manifest.loc[:, list(REQUIRED_MANIFEST_COLUMNS)].copy()
    if result.isna().any().any():
        raise ValueError("Frozen H2 manifest contains missing values in required fields")
    for column in ("dataset", "sample_id", "source_id", "dataset_revision", "selection_key_sha256", "manifest_version"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Frozen H2 manifest has an empty {column}")
    if not result["manifest_version"].eq(PRE_SCORE_MANIFEST_VERSION).all():
        observed = sorted(result["manifest_version"].unique().tolist())
        raise ValueError(f"Frozen H2 manifest marker must be {PRE_SCORE_MANIFEST_VERSION}, got {observed}")
    if result.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Frozen H2 manifest has duplicate dataset/sample_id identities")
    for column in ("selection_key_sha256", "source_input_csv_sha256"):
        if not result[column].str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Frozen H2 manifest column {column} must contain lowercase SHA-256 values")
    datasets_raw = arena_index.get("datasets") if isinstance(arena_index, Mapping) else None
    if not isinstance(datasets_raw, Mapping):
        raise ValueError("Arena index lacks a datasets mapping")
    indexed: dict[str, Mapping[str, Any]] = {}
    for dataset_name, group in result.groupby("dataset", sort=True):
        if dataset_name not in datasets_raw or not isinstance(datasets_raw[dataset_name], Mapping):
            raise ValueError(f"Frozen H2 manifest names unknown Arena dataset {dataset_name!r}")
        dataset = datasets_raw[dataset_name]
        revision = str(dataset.get("revision", "")).strip()
        repo_id = str(dataset.get("repo_id", "")).strip()
        if not revision or not repo_id:
            raise ValueError(f"Arena index entry {dataset_name!r} lacks a pinned repo_id or revision")
        observed = set(group["dataset_revision"].tolist())
        if observed != {revision}:
            raise ValueError(
                f"Frozen H2 manifest revision mismatch for {dataset_name}: expected {revision}, observed {sorted(observed)}"
            )
        indexed[dataset_name] = dataset
    return result.sort_values(["dataset", "label", "selection_rank", "sample_id"], kind="stable").reset_index(drop=True), indexed


def validate_frozen_inputs(
    manifest_path: str | Path,
    arm_ledger_path: str | Path,
    arena_index_path: str | Path,
) -> ValidatedFrozenInputs:
    """Read and validate the exact frozen artifacts without loading audio or ASR."""
    manifest_source = Path(manifest_path).resolve()
    ledger_source = Path(arm_ledger_path).resolve()
    index_source = Path(arena_index_path).resolve()
    for source in (manifest_source, ledger_source, index_source):
        if not source.is_file():
            raise FileNotFoundError(f"Required H2 input is missing: {source}")
    manifest = pd.read_csv(manifest_source)
    ledger = json.loads(ledger_source.read_text(encoding="utf-8"))
    index = yaml.safe_load(index_source.read_text(encoding="utf-8"))
    if not isinstance(index, Mapping):
        raise ValueError("Arena index is not a YAML mapping")
    valid_manifest, datasets = _validate_manifest_frame(manifest, index)
    arms = validate_arm_ledger(ledger)
    return ValidatedFrozenInputs(
        manifest=valid_manifest,
        arms=arms,
        datasets=datasets,
        manifest_sha256=sha256_file(manifest_source),
        arm_ledger_sha256=sha256_file(ledger_source),
        arena_index_sha256=sha256_file(index_source),
    )


def assert_git_clean_and_committed(paths: Sequence[str | Path], *, repo_root: str | Path) -> None:
    """Refuse an untracked, staged, or modified frozen protocol artifact."""
    root = Path(repo_root).resolve()
    for raw_path in paths:
        source = Path(raw_path).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Frozen H2 artifact must live under the repository: {source}") from error
        text_path = str(relative)
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", text_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"Frozen H2 artifact is not committed/tracked: {relative}")
        for command in (("diff", "--quiet", "HEAD", "--", text_path), ("diff", "--cached", "--quiet", "HEAD", "--", text_path)):
            clean = subprocess.run(["git", "-C", str(root), *command], check=False)
            if clean.returncode != 0:
                raise RuntimeError(f"Frozen H2 artifact has uncommitted changes: {relative}")


def pair_id_for(manifest_row: Mapping[str, Any], arm: ArmDefinition) -> str:
    """Match :func:`evaluate_quality_pair`'s content-addressed pair identity."""
    required = ("dataset", "sample_id", "label", "selection_key_sha256")
    missing = [column for column in required if column not in manifest_row]
    if missing:
        raise ValueError(f"Cannot create pair ID without: {missing}")
    payload = {
        "dataset": str(manifest_row["dataset"]),
        "sample_id": str(manifest_row["sample_id"]),
        "label": _json_safe(manifest_row["label"]),
        "selection_key_sha256": str(manifest_row["selection_key_sha256"]),
        "arm": arm.arm_id,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _base_failure_row(manifest_row: Mapping[str, Any], arm: ArmDefinition, reason: str) -> dict[str, Any]:
    """Create a schema-valid retained failure when source audio is unavailable."""
    record = arm.as_record()
    row: dict[str, Any] = {
        "quality_row_version": QUALITY_ROW_VERSION,
        "dataset": str(manifest_row["dataset"]),
        "sample_id": str(manifest_row["sample_id"]),
        "label": _json_safe(manifest_row["label"]),
        "source_id": _json_safe(manifest_row.get("source_id")),
        "speaker_id": _json_safe(manifest_row.get("speaker_id")),
        "input_manifest_version": _json_safe(manifest_row.get("manifest_version")),
        "source_input_csv_sha256": _json_safe(manifest_row.get("source_input_csv_sha256")),
        "selection_seed": _json_safe(manifest_row.get("selection_seed")),
        "selection_method": _json_safe(manifest_row.get("selection_method")),
        "selection_key_sha256": _json_safe(manifest_row.get("selection_key_sha256")),
        "arm": arm.arm_id,
        "arm_definition_sha256": record["definition_sha256"],
        "arm_parameters_json": _canonical_json(record["parameters"]),
        "target_feature": arm.target_feature,
        "target_direction": arm.target_direction,
        "negative_control": arm.negative_control,
        "detector_stage": "blocked_pending_quality_freeze",
        "retained": False,
        "quality_status": "failed",
        "failure_reasons_json": _canonical_json([reason]),
        "pair_id": pair_id_for(manifest_row, arm),
    }
    row.update({field: False for field in PAIR_FAILURE_FIELDS})
    return row


def _atomic_write_if_absent(path: Path, text: str) -> bool:
    """Atomically publish a file once; return ``False`` if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace(path: Path, text: str) -> None:
    """Atomically refresh a deterministic derived artifact such as a summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    safe_row = _json_safe(dict(row))
    pair_id = str(safe_row.get("pair_id", ""))
    if not pair_id or not all(character in "0123456789abcdef" for character in pair_id) or len(pair_id) != 64:
        raise ValueError("Quality row must contain a lowercase SHA-256 pair_id")
    quality_rows_frame([safe_row])
    return {
        "artifact_kind": QUALITY_SEGMENT_KIND,
        "version": QUALITY_RUNNER_VERSION,
        "pair_id": pair_id,
        "row": safe_row,
        "row_sha256": _sha256_bytes(_canonical_json(safe_row).encode("utf-8")),
    }


def read_pair_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read, hash-validate, and schema-validate one immutable pair checkpoint."""
    payload = _read_json(Path(path))
    if not isinstance(payload, Mapping) or payload.get("artifact_kind") != QUALITY_SEGMENT_KIND:
        raise ValueError(f"Invalid H2 quality checkpoint: {path}")
    row = payload.get("row")
    if not isinstance(row, Mapping):
        raise ValueError(f"H2 quality checkpoint lacks a row: {path}")
    if payload.get("pair_id") != row.get("pair_id"):
        raise ValueError(f"H2 quality checkpoint pair ID mismatch: {path}")
    observed = _sha256_bytes(_canonical_json(row).encode("utf-8"))
    if payload.get("row_sha256") != observed:
        raise ValueError(f"H2 quality checkpoint row hash mismatch: {path}")
    quality_rows_frame([row])
    return dict(row)


def persist_pair_checkpoint(path: str | Path, row: Mapping[str, Any]) -> bool:
    """Persist an immutable pair row or reject a same-ID conflicting result."""
    target = Path(path)
    payload = _segment_payload(row)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    created = _atomic_write_if_absent(target, text)
    if created:
        return True
    existing = read_pair_checkpoint(target)
    if _canonical_json(existing) != _canonical_json(payload["row"]):
        raise RuntimeError(f"Refusing to overwrite conflicting H2 pair checkpoint: {target}")
    return False


def _transcript_cache_name(manifest_row: Mapping[str, Any]) -> str:
    identity = {
        "dataset": str(manifest_row["dataset"]),
        "sample_id": str(manifest_row["sample_id"]),
        "source_id": str(manifest_row.get("source_id", "")),
        "selection_key_sha256": str(manifest_row["selection_key_sha256"]),
    }
    return _sha256_bytes(_canonical_json(identity).encode("utf-8")) + ".json"


class OriginalTranscriptCache:
    """Durable original-waveform ASR cache shared by all arms of one sample."""

    def __init__(self, root: str | Path, transcriber: Transcriber, runtime_provenance: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.transcriber = transcriber
        self.runtime_provenance = _json_safe(dict(runtime_provenance))

    def _path(self, manifest_row: Mapping[str, Any]) -> Path:
        return self.root / _transcript_cache_name(manifest_row)

    def get_or_create(self, manifest_row: Mapping[str, Any], audio: np.ndarray, sample_rate: int) -> str:
        """Reuse an identical cached original transcript or persist its first result/failure."""
        source_hash = waveform_sha256(audio)
        target = self._path(manifest_row)
        if target.exists():
            return self._read_and_resolve(target, manifest_row, source_hash, sample_rate)
        base: dict[str, Any] = {
            "artifact_kind": TRANSCRIPT_CACHE_KIND,
            "version": QUALITY_RUNNER_VERSION,
            "dataset": str(manifest_row["dataset"]),
            "sample_id": str(manifest_row["sample_id"]),
            "source_id": _json_safe(manifest_row.get("source_id")),
            "selection_key_sha256": str(manifest_row["selection_key_sha256"]),
            "original_wave_sha256": source_hash,
            "sample_rate_hz": int(sample_rate),
            "asr_runtime": self.runtime_provenance,
        }
        try:
            transcript = self.transcriber.transcribe(audio)
            if not isinstance(transcript, str):
                raise TypeError("ASR transcriber must return a string")
            payload = {**base, "status": "ok", "transcript": transcript}
        except Exception as error:  # cache failures so every arm sees the same original-ASR state
            payload = {**base, "status": "error", "error_type": type(error).__name__}
        _atomic_write_if_absent(target, json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return self._read_and_resolve(target, manifest_row, source_hash, sample_rate)

    def _read_and_resolve(
        self,
        path: Path,
        manifest_row: Mapping[str, Any],
        source_hash: str,
        sample_rate: int,
    ) -> str:
        payload = _read_json(path)
        expected = {
            "artifact_kind": TRANSCRIPT_CACHE_KIND,
            "dataset": str(manifest_row["dataset"]),
            "sample_id": str(manifest_row["sample_id"]),
            "selection_key_sha256": str(manifest_row["selection_key_sha256"]),
            "original_wave_sha256": source_hash,
            "sample_rate_hz": int(sample_rate),
        }
        if not isinstance(payload, Mapping) or any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Conflicting original-transcript cache entry: {path}")
        if payload.get("status") == "ok" and isinstance(payload.get("transcript"), str):
            return str(payload["transcript"])
        if payload.get("status") == "error":
            raise CachedOriginalTranscriptError(str(payload.get("error_type", "unknown")))
        raise RuntimeError(f"Invalid original-transcript cache entry: {path}")


def quality_run_paths(
    hdd_root: str | Path,
    run_id: str,
    repo_root: str | Path,
) -> QualityRunPaths:
    """Resolve non-ambiguous HDD and compact repository paths for one run ID."""
    if not run_id or not all(character.isalnum() or character in "_-" for character in run_id):
        raise ValueError("--run-id must be non-empty and contain only letters, numbers, underscores, or hyphens")
    hdd = Path(hdd_root).resolve()
    root = hdd / "runs" / "h2_causal_interventions" / run_id
    repo = Path(repo_root).resolve()
    return QualityRunPaths(
        run_root=root,
        pair_rows_dir=root / "pair_rows",
        transcript_cache_dir=root / "original_transcripts",
        full_table_path=root / "quality_pairs.parquet",
        provenance_path=root / "quality_provenance.json",
        repo_summary_path=repo / "experiments" / "h2_causal_interventions" / "results" / "quality_runs" / f"{run_id}.summary.json",
    )


def _run_contract(
    *,
    run_id: str,
    validated: ValidatedFrozenInputs,
    mode: str,
    limit: int | None,
    whisper_config: WhisperQualityGateConfig,
) -> dict[str, Any]:
    return {
        "artifact_kind": RUN_PROVENANCE_KIND,
        "version": QUALITY_RUNNER_VERSION,
        "run_id": run_id,
        "mode": mode,
        "limit_samples": limit,
        "claim_guard": "Detector-free waveform quality only; no detector was imported, loaded, or run.",
        "detector_scoring_allowed": False,
        "panel_gate_eligible": False if mode == "pilot_not_panel_gate" else None,
        "manifest_sha256": validated.manifest_sha256,
        "arm_ledger_sha256": validated.arm_ledger_sha256,
        "arena_index_sha256": validated.arena_index_sha256,
        "manifest_version": PRE_SCORE_MANIFEST_VERSION,
        "arms": {arm.arm_id: arm.as_record()["definition_sha256"] for arm in validated.arms},
        "datasets": {
            name: {
                "repo_id": value.get("repo_id"),
                "revision": value.get("revision"),
                "source_revision_resolved": value.get("source_revision_resolved"),
            }
            for name, value in sorted(validated.datasets.items())
        },
        "asr_runtime": whisper_config.provenance(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_whisper_device": whisper_config.device,
    }


def initialize_run(paths: QualityRunPaths, contract: Mapping[str, Any]) -> None:
    """Create immutable run provenance, or require an exact resume contract."""
    text = json.dumps(_json_safe(contract), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if _atomic_write_if_absent(paths.provenance_path, text):
        return
    existing = _read_json(paths.provenance_path)
    if _canonical_json(existing) != _canonical_json(contract):
        raise RuntimeError(f"Refusing to resume run with a conflicting quality contract: {paths.run_root}")


def _all_expected_pairs(manifest: pd.DataFrame, arms: Sequence[ArmDefinition]) -> dict[str, tuple[dict[str, Any], ArmDefinition]]:
    result: dict[str, tuple[dict[str, Any], ArmDefinition]] = {}
    for manifest_row in manifest.to_dict(orient="records"):
        for arm in arms:
            pair_id = pair_id_for(manifest_row, arm)
            if pair_id in result:
                raise RuntimeError(f"Duplicate expected H2 pair ID: {pair_id}")
            result[pair_id] = (manifest_row, arm)
    return result


def load_existing_checkpoints(
    pair_rows_dir: str | Path,
    expected: Mapping[str, tuple[dict[str, Any], ArmDefinition]],
) -> dict[str, dict[str, Any]]:
    """Load only expected immutable rows and refuse stale/corrupt/conflicting entries."""
    directory = Path(pair_rows_dir)
    if not directory.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for source in sorted(directory.glob("*.json")):
        row = read_pair_checkpoint(source)
        pair_id = str(row["pair_id"])
        if pair_id not in expected:
            raise RuntimeError(f"Quality run contains a checkpoint outside its frozen panel: {source}")
        manifest_row, arm = expected[pair_id]
        if row.get("dataset") != manifest_row["dataset"] or row.get("sample_id") != manifest_row["sample_id"] or row.get("arm") != arm.arm_id:
            raise RuntimeError(f"Quality checkpoint identity conflicts with frozen panel: {source}")
        if pair_id in rows:
            raise RuntimeError(f"Duplicate quality pair checkpoint: {pair_id}")
        rows[pair_id] = row
    return rows


def _checkpoint_path(paths: QualityRunPaths, pair_id: str) -> Path:
    return paths.pair_rows_dir / f"{pair_id}.json"


def _persist_failure_for_sample(
    paths: QualityRunPaths,
    manifest_row: Mapping[str, Any],
    arms: Sequence[ArmDefinition],
    complete: set[str],
    reason: str,
) -> int:
    count = 0
    for arm in arms:
        pair_id = pair_id_for(manifest_row, arm)
        if pair_id in complete:
            continue
        row = _base_failure_row(manifest_row, arm, reason)
        persist_pair_checkpoint(_checkpoint_path(paths, pair_id), row)
        complete.add(pair_id)
        count += 1
    return count


def process_audio_record(
    record: AudioRecord,
    manifest_rows: Sequence[Mapping[str, Any]],
    arms: Sequence[ArmDefinition],
    complete: set[str],
    paths: QualityRunPaths,
    transcript_cache: OriginalTranscriptCache,
    transcriber: Transcriber,
) -> int:
    """Evaluate all still-pending arms for one decoded source record."""
    by_id = {str(row["sample_id"]): row for row in manifest_rows}
    manifest_row = by_id[str(record.sample_id)]
    pending = [arm for arm in arms if pair_id_for(manifest_row, arm) not in complete]
    if not pending:
        return 0
    try:
        original, sample_rate = decode_audio(record.audio_bytes)
    except Exception as error:
        return _persist_failure_for_sample(
            paths, manifest_row, pending, complete, f"audio_decode_error:{type(error).__name__}"
        )
    original_hash = waveform_sha256(original)
    try:
        original_transcript = transcript_cache.get_or_create(manifest_row, original, sample_rate)
    except Exception as error:
        original_error = error
        original_transcript = None
    else:
        original_error = None

    def transcribe_pair(audio: np.ndarray, _rate: int) -> str:
        if waveform_sha256(audio) == original_hash:
            if original_error is not None:
                raise original_error
            assert original_transcript is not None
            return original_transcript
        return transcriber.transcribe(audio)

    count = 0
    for arm in pending:
        row = evaluate_quality_pair(
            manifest_row,
            audio=original,
            sample_rate=sample_rate,
            arm=arm,
            transcribe=transcribe_pair,
        )
        pair_id = pair_id_for(manifest_row, arm)
        if row["pair_id"] != pair_id:
            raise RuntimeError("Internal H2 pair identity mismatch")
        persist_pair_checkpoint(_checkpoint_path(paths, pair_id), row)
        complete.add(pair_id)
        count += 1
    return count


def materialize_quality_table(paths: QualityRunPaths) -> pd.DataFrame:
    """Build the derived complete table from immutable pair checkpoints."""
    rows = [read_pair_checkpoint(source) for source in sorted(paths.pair_rows_dir.glob("*.json"))]
    if not rows:
        return pd.DataFrame()
    table = quality_rows_frame(rows)
    temporary = paths.full_table_path.with_name(f".{paths.full_table_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(temporary, index=False)
    os.replace(temporary, paths.full_table_path)
    return table


def build_quality_summary(
    table: pd.DataFrame,
    expected_pairs: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Make a compact, non-claiming status summary from quality rows only."""
    completed = int(len(table))
    per_arm: dict[str, Any] = {}
    if not table.empty:
        for arm, group in table.groupby("arm", sort=True):
            per_arm[str(arm)] = {
                "completed_pairs": int(len(group)),
                "retained_pairs": int(group["retained"].sum()),
                "retained_fraction": float(group["retained"].mean()),
                "status": "diagnostic_only_not_panel_gate",
            }
    return {
        "artifact_kind": "h2_detector_free_quality_summary",
        "version": QUALITY_RUNNER_VERSION,
        "run_id": contract["run_id"],
        "mode": contract["mode"],
        "claim_guard": "Quality diagnostics only; no detector was imported, loaded, or run, and this summary is not causal evidence.",
        "detector_scoring_allowed": False,
        "panel_gate_status": "not_panel_gate" if contract["mode"] == "pilot_not_panel_gate" else "not_frozen",
        "expected_pairs": expected_pairs,
        "completed_pairs": completed,
        "remaining_pairs": expected_pairs - completed,
        "complete": completed == expected_pairs,
        "per_arm": per_arm,
        "provenance_sha256": _sha256_bytes(_canonical_json(contract).encode("utf-8")),
    }


def run_quality_panel(
    validated: ValidatedFrozenInputs,
    *,
    paths: QualityRunPaths,
    run_id: str,
    mode: str,
    limit: int | None,
    whisper_config: WhisperQualityGateConfig,
    transcriber: Transcriber | None = None,
    audio_iterator: Callable[[Mapping[str, Any], pd.DataFrame], Iterable[AudioRecord]] = iter_selected_audio,
) -> dict[str, Any]:
    """Execute/resume a detector-free H2 quality panel from immutable protocol inputs."""
    if mode not in {"full_panel_pre_score_quality", "pilot_not_panel_gate"}:
        raise ValueError(f"Unsupported H2 quality mode: {mode}")
    manifest = validated.manifest
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        if mode != "pilot_not_panel_gate":
            raise ValueError("A bounded H2 quality run must be labeled pilot_not_panel_gate")
        manifest = manifest.iloc[: int(limit)].copy()
    contract = _run_contract(
        run_id=run_id,
        validated=validated,
        mode=mode,
        limit=limit,
        whisper_config=whisper_config,
    )
    initialize_run(paths, contract)
    expected = _all_expected_pairs(manifest, validated.arms)
    existing = load_existing_checkpoints(paths.pair_rows_dir, expected)
    complete = set(existing)
    worker: Transcriber = LazyWhisperTranscriber(whisper_config) if transcriber is None else transcriber
    cache = OriginalTranscriptCache(paths.transcript_cache_dir, worker, whisper_config.provenance())
    rows_by_dataset = {name: group.to_dict(orient="records") for name, group in manifest.groupby("dataset", sort=True)}
    newly_written = 0
    for dataset_name, rows in rows_by_dataset.items():
        selected = pd.DataFrame(rows)
        observed: set[str] = set()
        try:
            for record in audio_iterator(validated.datasets[dataset_name], selected):
                observed.add(str(record.sample_id))
                newly_written += process_audio_record(
                    record,
                    rows,
                    validated.arms,
                    complete,
                    paths,
                    cache,
                    worker,
                )
        except Exception as error:
            stream_reason = f"audio_stream_error:{type(error).__name__}"
            for row in rows:
                if str(row["sample_id"]) not in observed:
                    newly_written += _persist_failure_for_sample(paths, row, validated.arms, complete, stream_reason)
        else:
            for row in rows:
                if str(row["sample_id"]) not in observed:
                    newly_written += _persist_failure_for_sample(
                        paths, row, validated.arms, complete, "audio_missing_from_indexed_shards"
                    )
    table = materialize_quality_table(paths)
    summary = build_quality_summary(table, len(expected), contract)
    summary["newly_written_pairs"] = newly_written
    summary_text = json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_replace(paths.run_root / "quality_summary.json", summary_text)
    _atomic_replace(paths.repo_summary_path, summary_text)
    return summary
