"""Strict, resumable H2 paired detector scoring after a quality freeze.

This module is deliberately downstream of the detector-free quality stage.  It
will only consume a committed score-eligible quality freeze, then regenerates
every selected source/transformed waveform from the committed pre-score panel
and verifies both waveform hashes before a detector is constructed.  Published
Arena scores are not an input to this module: parity reports determine the
fixed scorer contracts, while H2 scores are generated locally per pair.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.arena_io import AudioRecord, decode_audio, iter_selected_audio
from src.h2_pre_score_pairs import apply_registered_arm, waveform_sha256
from src.h2_quality_runner import (
    DEFAULT_HDD_ROOT,
    ValidatedFrozenInputs,
    assert_git_clean_and_committed,
    pair_id_for,
    sha256_file,
    validate_frozen_inputs,
)
from src.onnx_fixed_window import canonicalize_raw_score


QUALITY_FREEZE_VERSION = "h2_quality_freeze_v1"
SCORING_VERSION = "h2_paired_scoring_v1"
QUALITY_FREEZE_KIND = "h2_score_eligible_quality_freeze"
SCORE_SEGMENT_KIND = "h2_paired_detector_score_segment"
SCORE_PROVENANCE_KIND = "h2_paired_detector_score_run"
DEFAULT_PARITY_ROOT = Path("experiments/h2_causal_interventions/results/parity_calibration/ASVspoof2019_LA")

REQUIRED_QUALITY_COLUMNS = (
    "quality_row_version",
    "pair_id",
    "dataset",
    "sample_id",
    "label",
    "source_id",
    "selection_key_sha256",
    "arm",
    "arm_definition_sha256",
    "detector_stage",
    "retained",
    "quality_status",
    "failure_reasons_json",
    "sample_rate_hz",
    "original_samples",
    "transformed_samples",
    "original_wave_sha256",
    "transformed_wave_sha256",
)


@dataclass(frozen=True)
class ScorerSpec:
    """A parity-validated scorer with a fixed physical-GPU assignment."""

    model: str
    kind: str
    filename: str | None
    preprocessing: str
    batch_size: int
    physical_gpu: int
    canonical_orientation: str


SCORER_SPECS: dict[str, ScorerSpec] = {
    "Spectra-AASIST": ScorerSpec(
        model="Spectra-AASIST",
        kind="onnx",
        filename="spectra-aasist.onnx",
        preprocessing="preemphasis_0.97",
        batch_size=8,
        physical_gpu=0,
        canonical_orientation="negated_raw_is_spoof",
    ),
    "AASIST": ScorerSpec(
        model="AASIST",
        kind="onnx",
        filename="aasist.onnx",
        preprocessing="raw",
        batch_size=2,
        physical_gpu=1,
        canonical_orientation="negated_raw_is_spoof",
    ),
    "Res2TCNGuard": ScorerSpec(
        model="Res2TCNGuard",
        kind="pytorch",
        filename=None,
        preprocessing="raw",
        batch_size=2,
        physical_gpu=2,
        canonical_orientation="negated_raw_is_spoof",
    ),
}


class PairScorer(Protocol):
    """The narrow scorer surface used after all waveform hashes validate."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def score(self, waveforms: Sequence[np.ndarray] | np.ndarray) -> Any: ...


@dataclass(frozen=True)
class ValidatedScoreEligibility:
    """Validated freeze, original pre-score inputs, and retained-only pairs."""

    frozen_inputs: ValidatedFrozenInputs
    pairs: pd.DataFrame
    quality_freeze: Mapping[str, Any]
    quality_manifest_sha256: str
    quality_freeze_sha256: str


@dataclass(frozen=True)
class RegeneratedPair:
    """One independently regenerated and hash-matched source/transformed pair."""

    pair: Mapping[str, Any]
    original: np.ndarray
    transformed: np.ndarray
    sample_rate_hz: int


@dataclass(frozen=True)
class ScoreRunPaths:
    """HDD-only results for one explicit model/run combination."""

    run_root: Path
    pair_rows_dir: Path
    scores_path: Path
    provenance_path: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_safe(value: Any) -> Any:
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


def _as_true(value: Any, *, field: str) -> bool:
    """Require a literal true value rather than relying on Python truthiness."""
    if isinstance(value, (bool, np.bool_)):
        result = bool(value)
    elif isinstance(value, str) and value.strip().casefold() == "true":
        result = True
    else:
        result = False
    if not result:
        raise ValueError(f"Score-eligible quality freeze requires {field}=true")
    return True


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _same_scalar(left: Any, right: Any) -> bool:
    """Compare scalar IDs while rejecting numeric/string coercion surprises."""
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    return left == right and type(left) is type(right) or str(left) == str(right)


def _read_quality_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() != ".csv":
        raise ValueError("Score-eligible quality manifest must be a CSV so its freeze hash is portable and inspectable")
    return pd.read_csv(path)


def _validate_quality_freeze(
    path: Path,
    *,
    quality_manifest_sha256: str,
    frozen_inputs: ValidatedFrozenInputs,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Quality freeze must be a JSON object")
    required = {
        "artifact_kind": QUALITY_FREEZE_KIND,
        "version": QUALITY_FREEZE_VERSION,
        "freeze_status": "score_eligible",
        "detector_scoring_allowed": True,
        "detector_scores_read": False,
        "detector_scoring_performed": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(f"Quality freeze field {field!r} must equal {expected!r}")
    expected_hashes = {
        "input_manifest_sha256": frozen_inputs.manifest_sha256,
        "arm_ledger_sha256": frozen_inputs.arm_ledger_sha256,
        "arena_index_sha256": frozen_inputs.arena_index_sha256,
        "score_eligible_manifest_sha256": quality_manifest_sha256,
    }
    for field, expected in expected_hashes.items():
        observed = _require_sha256(payload.get(field), field=field)
        if observed != expected:
            raise ValueError(f"Quality freeze {field} does not match the supplied immutable artifact")
    for field in ("quality_provenance_sha256", "quality_table_sha256", "quality_checkpoint_aggregate_sha256"):
        _require_sha256(payload.get(field), field=field)
    if not str(payload.get("quality_run_id", "")).strip():
        raise ValueError("Quality freeze lacks a non-empty quality_run_id")
    if payload.get("predeclared_minimum_retained_fraction") != 0.90:
        raise ValueError("Quality freeze must preserve the predeclared 90% per-arm retained-pair gate")
    per_arm = payload.get("per_arm")
    expected_arms = {arm.arm_id for arm in frozen_inputs.arms}
    if not isinstance(per_arm, Mapping) or set(per_arm) != expected_arms:
        raise ValueError("Quality freeze per_arm report differs from the committed arm ledger")
    for arm_id, details in per_arm.items():
        if not isinstance(details, Mapping) or details.get("gate_status") != "passed":
            raise ValueError(f"Quality freeze arm {arm_id} did not pass its retained-pair gate")
        try:
            retained_fraction = float(details.get("retained_fraction"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Quality freeze arm {arm_id} lacks a numeric retained fraction") from error
        if retained_fraction < 0.90:
            raise ValueError(f"Quality freeze arm {arm_id} is below the required 90% retained fraction")
    return payload


def validate_score_eligibility(
    *,
    quality_manifest_path: str | Path,
    quality_freeze_path: str | Path,
    input_manifest_path: str | Path,
    arm_ledger_path: str | Path,
    arena_index_path: str | Path,
) -> ValidatedScoreEligibility:
    """Accept only a fully declared, retained-only quality freeze.

    This is intentionally independent of a detector runtime.  It validates the
    retained rows against the committed pre-score panel and arm ledger before a
    scorer can be constructed.
    """
    paths = tuple(Path(value).resolve() for value in (
        quality_manifest_path,
        quality_freeze_path,
        input_manifest_path,
        arm_ledger_path,
        arena_index_path,
    ))
    quality_path, freeze_path, manifest_path, ledger_path, index_path = paths
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required H2 paired-scoring input is missing: {path}")
    frozen = validate_frozen_inputs(manifest_path, ledger_path, index_path)
    pairs = _read_quality_manifest(quality_path)
    if pairs.empty:
        raise ValueError("Score-eligible quality manifest is empty")
    if pairs.columns.duplicated().any():
        raise ValueError("Score-eligible quality manifest contains duplicate columns")
    missing = sorted(set(REQUIRED_QUALITY_COLUMNS).difference(pairs.columns))
    if missing:
        raise ValueError(f"Score-eligible quality manifest lacks columns: {missing}")
    quality_hash = sha256_file(quality_path)
    freeze = _validate_quality_freeze(freeze_path, quality_manifest_sha256=quality_hash, frozen_inputs=frozen)
    records = pairs.to_dict(orient="records")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("Score-eligible quality manifest contains duplicate pair_id values")
    if int(freeze.get("score_eligible_pairs", -1)) != len(pairs):
        raise ValueError("Quality freeze score_eligible_pairs disagrees with its CSV manifest")
    if not all(_as_true(row.get("retained"), field="retained") for row in records):
        raise AssertionError("unreachable")
    if not pairs["detector_stage"].astype(str).eq("eligible_after_quality_freeze").all():
        raise ValueError("Score-eligible rows must have detector_stage=eligible_after_quality_freeze")
    if not pairs["quality_status"].astype(str).eq("passed").all():
        raise ValueError("Score-eligible rows must have quality_status=passed")
    if not pairs["failure_reasons_json"].astype(str).eq("[]").all():
        raise ValueError("Score-eligible rows must retain an empty failure_reasons_json")

    input_rows = {
        (str(row["dataset"]), str(row["sample_id"])): row
        for row in frozen.manifest.to_dict(orient="records")
    }
    arms = {arm.arm_id: arm for arm in frozen.arms}
    normalized: list[dict[str, Any]] = []
    for row in records:
        key = (str(row["dataset"]), str(row["sample_id"]))
        source = input_rows.get(key)
        if source is None:
            raise ValueError(f"Quality-freeze pair is outside the committed H2 input panel: {key}")
        arm = arms.get(str(row["arm"]))
        if arm is None:
            raise ValueError(f"Quality-freeze pair names an unregistered arm: {row['arm']!r}")
        expected_pair_id = pair_id_for(source, arm)
        if str(row["pair_id"]) != expected_pair_id:
            raise ValueError(f"Quality-freeze pair_id mismatch for {key}/{arm.arm_id}")
        if str(row["arm_definition_sha256"]) != arm.as_record()["definition_sha256"]:
            raise ValueError(f"Quality-freeze arm hash mismatch for {key}/{arm.arm_id}")
        for field in ("label", "source_id", "selection_key_sha256"):
            if not _same_scalar(row[field], source[field]):
                raise ValueError(f"Quality-freeze {field} conflicts with frozen input for {key}")
        for field in ("original_wave_sha256", "transformed_wave_sha256"):
            _require_sha256(row[field], field=field)
        try:
            sample_rate = int(row["sample_rate_hz"])
            original_samples = int(row["original_samples"])
            transformed_samples = int(row["transformed_samples"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Quality-freeze waveform metadata is not integral for {key}/{arm.arm_id}") from error
        if sample_rate <= 0 or original_samples <= 0 or transformed_samples <= 0:
            raise ValueError(f"Quality-freeze waveform metadata must be positive for {key}/{arm.arm_id}")
        record = dict(row)
        record.update({"sample_rate_hz": sample_rate, "original_samples": original_samples, "transformed_samples": transformed_samples})
        normalized.append(record)
    result = pd.DataFrame(normalized).sort_values(["dataset", "sample_id", "arm", "pair_id"], kind="stable").reset_index(drop=True)
    return ValidatedScoreEligibility(
        frozen_inputs=frozen,
        pairs=result,
        quality_freeze=freeze,
        quality_manifest_sha256=quality_hash,
        quality_freeze_sha256=sha256_file(freeze_path),
    )


def assert_committed_score_inputs(
    *,
    repo_root: str | Path,
    quality_manifest_path: str | Path,
    quality_freeze_path: str | Path,
    input_manifest_path: str | Path,
    arm_ledger_path: str | Path,
    arena_index_path: str | Path,
    parity_root: str | Path,
    model: str,
) -> None:
    """Refuse uncommitted freezes or mutable readiness records in the CLI path."""
    if model not in SCORER_SPECS:
        raise ValueError(f"Unsupported parity-eligible model: {model}")
    root = Path(parity_root)
    parity_paths = [root / "parity_report.json"]
    if model == "Res2TCNGuard":
        parity_paths.append(root / "Res2TCNGuard__pytorch_parity_report.json")
    all_paths = [quality_manifest_path, quality_freeze_path, input_manifest_path, arm_ledger_path, arena_index_path, *parity_paths]
    assert_git_clean_and_committed(all_paths, repo_root=repo_root)


def validate_pinned_runtime_assignment(spec: ScorerSpec) -> None:
    """Require one visible physical GPU, making the recorded assignment explicit."""
    expected = str(spec.physical_gpu)
    observed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if observed != expected:
        raise RuntimeError(
            f"{spec.model} is pinned to physical GPU {expected}; set CUDA_VISIBLE_DEVICES={expected!r} "
            "so the runner can use logical cuda:0"
        )


def validate_parity_eligibility(
    spec: ScorerSpec,
    *,
    parity_root: str | Path = DEFAULT_PARITY_ROOT,
) -> dict[str, Any]:
    """Read compact parity reports only; never load Arena score artifacts."""
    root = Path(parity_root)
    onnx_report_path = root / "parity_report.json"
    if not onnx_report_path.is_file():
        raise FileNotFoundError(f"Pinned ONNX parity report is missing: {onnx_report_path}")
    onnx_report = json.loads(onnx_report_path.read_text(encoding="utf-8"))
    provenance: dict[str, Any] = {"onnx_parity_report_sha256": sha256_file(onnx_report_path)}
    if spec.kind == "onnx":
        decision = onnx_report.get("decisions", {}).get(spec.model)
        if not isinstance(decision, Mapping) or decision.get("status") != "eligible":
            raise ValueError(f"{spec.model} is not eligible in the pinned ONNX parity report")
        if decision.get("chosen_preprocessing") != spec.preprocessing:
            raise ValueError(f"{spec.model} preprocessing differs from the pinned parity decision")
        if decision.get("chosen_provider") != "CUDAExecutionProvider":
            raise ValueError(f"{spec.model} parity did not establish the required CUDA ONNX provider")
        provenance["parity_decision"] = dict(decision)
        return provenance
    report_path = root / "Res2TCNGuard__pytorch_parity_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Pinned Res2TCNGuard PyTorch parity report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("model") != spec.model or report.get("runner") != "pinned_pytorch_evaluate_py":
        raise ValueError("Pinned Res2TCNGuard report does not name the source-PyTorch scorer")
    if report.get("ordering_gate_pass") is not True or report.get("arena_orientation") != spec.canonical_orientation:
        raise ValueError("Pinned Res2TCNGuard source-PyTorch parity gate is not eligible")
    provenance["pytorch_parity_report_sha256"] = sha256_file(report_path)
    provenance["parity_decision"] = {
        "ordering_gate_pass": True,
        "runner": "pinned_pytorch_evaluate_py",
        "arena_orientation": spec.canonical_orientation,
    }
    return provenance


def build_scorer(spec: ScorerSpec, model_directory: str | Path) -> PairScorer:
    """Construct exactly one parity-validated scorer after all input checks pass."""
    directory = Path(model_directory)
    if spec.kind == "onnx":
        from src.onnx_fixed_window import FixedWindowOnnxRunner

        assert spec.filename is not None
        runner = FixedWindowOnnxRunner(
            directory / spec.filename,
            preprocessing=spec.preprocessing,
            raw_scalar_index=1,
            cuda_device_id=0,
            allow_cpu_fallback=False,
        )
        if runner.provider != "CUDAExecutionProvider" or runner.used_cpu_fallback:
            raise RuntimeError(f"{spec.model} requires CUDAExecutionProvider without CPU fallback")
        return runner
    from src.res2tcn_pytorch import Res2TCNGuardPyTorchRunner

    return Res2TCNGuardPyTorchRunner(directory, device="cuda:0")


def _atomic_write_if_absent(path: Path, text: str) -> bool:
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


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    writer(temporary)
    os.replace(temporary, path)


def score_run_paths(hdd_root: str | Path, run_id: str, model: str) -> ScoreRunPaths:
    if not run_id or not all(character.isalnum() or character in "_-" for character in run_id):
        raise ValueError("--run-id must be non-empty and contain only letters, numbers, underscores, or hyphens")
    if model not in SCORER_SPECS:
        raise ValueError(f"Unsupported parity-eligible model: {model}")
    root = Path(hdd_root).resolve() / "runs" / "h2_causal_interventions" / "paired_scoring" / run_id / model
    return ScoreRunPaths(root, root / "pair_rows", root / "paired_scores.parquet", root / "score_provenance.json")


def _score_contract(
    *,
    run_id: str,
    spec: ScorerSpec,
    eligibility: ValidatedScoreEligibility,
    parity: Mapping[str, Any],
    model_directory: Path,
) -> dict[str, Any]:
    model_file_sha256: str | None = None
    if spec.filename is not None:
        model_file_sha256 = sha256_file(model_directory / spec.filename)
    return {
        "artifact_kind": SCORE_PROVENANCE_KIND,
        "version": SCORING_VERSION,
        "run_id": run_id,
        "model": spec.model,
        "scorer_kind": spec.kind,
        "preprocessing": spec.preprocessing,
        "batch_size": spec.batch_size,
        "physical_gpu": spec.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "canonical_orientation": spec.canonical_orientation,
        "n_quality_eligible_pairs": int(len(eligibility.pairs)),
        "quality_manifest_sha256": eligibility.quality_manifest_sha256,
        "quality_freeze_sha256": eligibility.quality_freeze_sha256,
        "input_manifest_sha256": eligibility.frozen_inputs.manifest_sha256,
        "arm_ledger_sha256": eligibility.frozen_inputs.arm_ledger_sha256,
        "arena_index_sha256": eligibility.frozen_inputs.arena_index_sha256,
        "quality_run_id": eligibility.quality_freeze["quality_run_id"],
        "quality_provenance_sha256": eligibility.quality_freeze["quality_provenance_sha256"],
        "quality_table_sha256": eligibility.quality_freeze["quality_table_sha256"],
        "parity": _json_safe(dict(parity)),
        "model_directory": str(model_directory),
        "model_file_sha256": model_file_sha256,
        "claim_guard": "Paired detector scores on quality-frozen waveforms only; no Arena score output was read as intervention input.",
    }


def initialize_score_run(paths: ScoreRunPaths, contract: Mapping[str, Any]) -> str:
    """Create immutable provenance or require an exact contract on resume."""
    safe_contract = _json_safe(dict(contract))
    text = json.dumps(safe_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if _atomic_write_if_absent(paths.provenance_path, text):
        return _sha256_bytes(_canonical_json(safe_contract).encode("utf-8"))
    existing = json.loads(paths.provenance_path.read_text(encoding="utf-8"))
    if _canonical_json(existing) != _canonical_json(safe_contract):
        raise RuntimeError(f"Refusing to resume an H2 score run with a conflicting contract: {paths.run_root}")
    return _sha256_bytes(_canonical_json(safe_contract).encode("utf-8"))


def _segment_path(paths: ScoreRunPaths, pair_id: str) -> Path:
    return paths.pair_rows_dir / f"{pair_id}.json"


def _read_segment(path: Path, *, model: str, contract_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_kind") != SCORE_SEGMENT_KIND:
        raise ValueError(f"Invalid H2 paired-score segment: {path}")
    row = payload.get("row")
    if not isinstance(row, Mapping) or payload.get("pair_id") != row.get("pair_id"):
        raise ValueError(f"H2 paired-score segment identity mismatch: {path}")
    if payload.get("model") != model or payload.get("contract_sha256") != contract_sha256:
        raise RuntimeError(f"H2 paired-score segment is from a different immutable contract: {path}")
    observed = _sha256_bytes(_canonical_json(row).encode("utf-8"))
    if payload.get("row_sha256") != observed:
        raise ValueError(f"H2 paired-score segment row hash mismatch: {path}")
    return dict(row)


def persist_score_segment(paths: ScoreRunPaths, row: Mapping[str, Any], *, model: str, contract_sha256: str) -> bool:
    """Persist a pair/model response once; exact existing checkpoints are skipped."""
    pair_id = str(row.get("pair_id", ""))
    _require_sha256(pair_id, field="pair_id")
    safe_row = _json_safe(dict(row))
    payload = {
        "artifact_kind": SCORE_SEGMENT_KIND,
        "version": SCORING_VERSION,
        "model": model,
        "contract_sha256": contract_sha256,
        "pair_id": pair_id,
        "row": safe_row,
        "row_sha256": _sha256_bytes(_canonical_json(safe_row).encode("utf-8")),
    }
    path = _segment_path(paths, pair_id)
    created = _atomic_write_if_absent(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if created:
        return True
    _read_segment(path, model=model, contract_sha256=contract_sha256)
    return False


def load_existing_score_segments(paths: ScoreRunPaths, *, model: str, contract_sha256: str) -> dict[str, dict[str, Any]]:
    if not paths.pair_rows_dir.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.pair_rows_dir.glob("*.json")):
        row = _read_segment(path, model=model, contract_sha256=contract_sha256)
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"Duplicate H2 paired-score segment: {pair_id}")
        result[pair_id] = row
    return result


def regenerate_quality_pairs(
    eligibility: ValidatedScoreEligibility,
    *,
    audio_iterator: Callable[[Mapping[str, Any], pd.DataFrame], Iterable[AudioRecord]] = iter_selected_audio,
) -> list[RegeneratedPair]:
    """Regenerate all frozen pairs and reject any source/transform hash drift."""
    arms = {arm.arm_id: arm for arm in eligibility.frozen_inputs.arms}
    by_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in eligibility.pairs.to_dict(orient="records"):
        by_source.setdefault((str(pair["dataset"]), str(pair["sample_id"])), []).append(pair)
    regenerated: list[RegeneratedPair] = []
    for dataset_name, dataset in eligibility.frozen_inputs.datasets.items():
        selected_keys = {key for key in by_source if key[0] == dataset_name}
        if not selected_keys:
            continue
        selected = eligibility.frozen_inputs.manifest.loc[
            eligibility.frozen_inputs.manifest.apply(lambda row: (str(row["dataset"]), str(row["sample_id"])) in selected_keys, axis=1)
        ]
        seen: set[tuple[str, str]] = set()
        for record in audio_iterator(dataset, selected):
            key = (dataset_name, str(record.sample_id))
            if key not in selected_keys:
                raise RuntimeError(f"Audio iterator returned a source outside the quality freeze: {key}")
            if key in seen:
                raise RuntimeError(f"Audio iterator returned duplicate source audio: {key}")
            seen.add(key)
            original, sample_rate = decode_audio(record.audio_bytes)
            original = np.asarray(original, dtype=np.float32).reshape(-1)
            source_hash = waveform_sha256(original)
            for pair in by_source[key]:
                if int(sample_rate) != int(pair["sample_rate_hz"]) or original.size != int(pair["original_samples"]):
                    raise RuntimeError(f"Regenerated source metadata differs from quality freeze for {pair['pair_id']}")
                if source_hash != str(pair["original_wave_sha256"]):
                    raise RuntimeError(f"Regenerated source waveform hash differs from quality freeze for {pair['pair_id']}")
                transformed = np.asarray(apply_registered_arm(original, int(sample_rate), arms[str(pair["arm"])]).waveform, dtype=np.float32).reshape(-1)
                if transformed.size != int(pair["transformed_samples"]):
                    raise RuntimeError(f"Regenerated transformed length differs from quality freeze for {pair['pair_id']}")
                if waveform_sha256(transformed) != str(pair["transformed_wave_sha256"]):
                    raise RuntimeError(f"Regenerated transformed waveform hash differs from quality freeze for {pair['pair_id']}")
                regenerated.append(RegeneratedPair(pair, original, transformed, int(sample_rate)))
        missing = selected_keys.difference(seen)
        if missing:
            raise RuntimeError(f"Indexed audio is missing score-eligible quality sources: {sorted(missing)[:4]}")
    if len(regenerated) != len(eligibility.pairs):
        raise RuntimeError("Internal regenerated H2 pair count mismatch")
    return sorted(regenerated, key=lambda item: str(item.pair["pair_id"]))


def _runner_provenance(scorer: PairScorer) -> dict[str, Any]:
    """Normalize provenance across ONNX and source-PyTorch scorer objects."""
    declared = getattr(scorer, "provenance", None)
    if isinstance(declared, Mapping):
        return _json_safe(dict(declared))
    return {
        "model_path": str(getattr(scorer, "model_path", "<unknown>")),
        "provider": str(getattr(scorer, "provider", "<unknown>")),
        "used_cpu_fallback": bool(getattr(scorer, "used_cpu_fallback", False)),
    }


def _score_rows(
    spec: ScorerSpec,
    batch: Sequence[RegeneratedPair],
    scorer: PairScorer,
    contract_sha256: str,
    quality_manifest_sha256: str,
) -> list[dict[str, Any]]:
    inputs: list[np.ndarray] = []
    for pair in batch:
        inputs.extend((pair.original, pair.transformed))
    output = scorer.score(inputs)
    raw = np.asarray(output.raw_score, dtype=np.float64)
    logits = np.asarray(output.logits, dtype=np.float64)
    detector_inputs = np.asarray(output.detector_waveforms, dtype=np.float32)
    expected = 2 * len(batch)
    if raw.shape != (expected,) or logits.shape != (expected, 2) or detector_inputs.shape[0] != expected:
        raise RuntimeError("Scorer output does not align with paired H2 input batch")
    canonical = canonicalize_raw_score(raw, spec.canonical_orientation)
    rows: list[dict[str, Any]] = []
    runner_provenance = _runner_provenance(scorer)
    for offset, pair in enumerate(batch):
        source_index, transformed_index = 2 * offset, 2 * offset + 1
        quality = pair.pair
        rows.append(
            {
                "pair_id": quality["pair_id"],
                "dataset": quality["dataset"],
                "sample_id": quality["sample_id"],
                "source_id": quality["source_id"],
                "label": quality["label"],
                "arm": quality["arm"],
                "arm_definition_sha256": quality["arm_definition_sha256"],
                "quality_manifest_sha256": quality_manifest_sha256,
                "original_wave_sha256": quality["original_wave_sha256"],
                "transformed_wave_sha256": quality["transformed_wave_sha256"],
                "model": spec.model,
                "preprocessing": spec.preprocessing,
                "batch_size": spec.batch_size,
                "physical_gpu": spec.physical_gpu,
                "logical_device": "cuda:0",
                "canonical_orientation": spec.canonical_orientation,
                "raw_scalar": "logit_1",
                "original_detector_input_sha256": waveform_sha256(detector_inputs[source_index]),
                "transformed_detector_input_sha256": waveform_sha256(detector_inputs[transformed_index]),
                "original_logit_0": float(logits[source_index, 0]),
                "original_logit_1": float(logits[source_index, 1]),
                "transformed_logit_0": float(logits[transformed_index, 0]),
                "transformed_logit_1": float(logits[transformed_index, 1]),
                "original_raw_score": float(raw[source_index]),
                "transformed_raw_score": float(raw[transformed_index]),
                "original_score_spoof": float(canonical[source_index]),
                "transformed_score_spoof": float(canonical[transformed_index]),
                "score_delta_spoof": float(canonical[transformed_index] - canonical[source_index]),
                "runner_provenance_json": _canonical_json(runner_provenance),
                "score_contract_sha256": contract_sha256,
            }
        )
    return rows


def materialize_score_table(paths: ScoreRunPaths, *, model: str, contract_sha256: str) -> pd.DataFrame:
    rows = list(load_existing_score_segments(paths, model=model, contract_sha256=contract_sha256).values())
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values(["dataset", "sample_id", "arm", "pair_id"], kind="stable").reset_index(drop=True)
    _atomic_replace(paths.scores_path, lambda target: table.to_parquet(target, index=False))
    return table


def run_paired_scoring(
    eligibility: ValidatedScoreEligibility,
    *,
    paths: ScoreRunPaths,
    run_id: str,
    spec: ScorerSpec,
    parity: Mapping[str, Any],
    model_directory: str | Path,
    scorer_factory: Callable[[ScorerSpec, Path], PairScorer] = build_scorer,
    audio_iterator: Callable[[Mapping[str, Any], pd.DataFrame], Iterable[AudioRecord]] = iter_selected_audio,
) -> dict[str, Any]:
    """Hash-validate all pairs, then score only missing immutable checkpoints."""
    directory = Path(model_directory)
    contract = _score_contract(
        run_id=run_id,
        spec=spec,
        eligibility=eligibility,
        parity=parity,
        model_directory=directory,
    )
    contract_sha256 = initialize_score_run(paths, contract)
    existing = load_existing_score_segments(paths, model=spec.model, contract_sha256=contract_sha256)
    expected = set(eligibility.pairs["pair_id"].astype(str))
    if set(existing).difference(expected):
        raise RuntimeError("Existing H2 paired-score checkpoints are outside the frozen quality manifest")
    regenerated = regenerate_quality_pairs(eligibility, audio_iterator=audio_iterator)
    pending = [pair for pair in regenerated if str(pair.pair["pair_id"]) not in existing]
    written = 0
    if pending:
        scorer = scorer_factory(spec, directory)
        for start in range(0, len(pending), spec.batch_size):
            for row in _score_rows(
                spec,
                pending[start : start + spec.batch_size],
                scorer,
                contract_sha256,
                eligibility.quality_manifest_sha256,
            ):
                if persist_score_segment(paths, row, model=spec.model, contract_sha256=contract_sha256):
                    written += 1
    table = materialize_score_table(paths, model=spec.model, contract_sha256=contract_sha256)
    if set(table.get("pair_id", pd.Series(dtype=str)).astype(str)) != expected:
        raise RuntimeError("H2 paired score table is incomplete after scoring")
    return {
        "artifact_kind": "h2_paired_detector_score_summary",
        "version": SCORING_VERSION,
        "run_id": run_id,
        "model": spec.model,
        "quality_eligible_pairs": len(expected),
        "completed_pairs": len(table),
        "newly_written_pairs": written,
        "score_table": str(paths.scores_path),
        "score_table_sha256": sha256_file(paths.scores_path),
        "provenance": str(paths.provenance_path),
        "claim_guard": "Paired scores are an input to a separate causal analysis; this runner alone makes no causal claim.",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
