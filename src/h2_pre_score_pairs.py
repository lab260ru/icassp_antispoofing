"""Score-independent H2 pair freezing and transparent waveform quality rows.

This module is intentionally upstream of every detector.  It can (a) freeze a
balanced, seeded input panel from an explicit CSV, (b) expose only the
pre-registered crest-factor arms and negative controls, and (c) turn a supplied
waveform/ASR pair into a quality row.  It has no detector imports, no scorer
argument, and no detector-score output column.  A caller must quality-freeze
the retained rows before a separately implemented scorer is allowed to run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.audio_features import FEATURE_NAMES, compute_features, waveform_views
from src.h2_asr_wer import WordErrorResult, normalized_word_error
from src.h2_waveform_transforms import TransformResult, apply_gain, apply_polarity, compress_crest_factor


PRE_SCORE_MANIFEST_VERSION = "h2_pre_score_pairs_v1"
QUALITY_ROW_VERSION = "h2_quality_row_v1"
REQUIRED_INPUT_COLUMNS = ("dataset", "sample_id", "label")
_FORBIDDEN_INPUT_MARKERS = ("score", "logit", "probability", "prediction")


def _canonical_json(value: Any) -> str:
    """Encode compact, deterministic JSON suitable for content addressing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash an explicit source file without reading a detector artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_column_name(column: object) -> str:
    return "".join(character for character in str(column).casefold() if character.isalnum())


def assert_score_independent_columns(columns: Sequence[object]) -> None:
    """Reject inputs/outputs that could carry detector responses.

    The check is intentionally conservative: a H2 input panel has no reason to
    contain score, logit, probability, or prediction columns.  This prevents a
    future convenience join from silently making sample selection score-aware.
    """
    forbidden = [
        str(column)
        for column in columns
        if any(marker in _normal_column_name(column) for marker in _FORBIDDEN_INPUT_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"Pre-score H2 artifacts must not contain detector-response columns: {forbidden}")


def _required_nonempty_strings(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"Input manifest column {column!r} contains a missing/empty value")
    return values.astype(str).str.strip()


def validate_score_independent_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one explicit, label-bearing H2 input panel before selection."""
    if frame.empty:
        raise ValueError("Explicit H2 input CSV is empty")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Explicit H2 input CSV has duplicate columns: {duplicates}")
    assert_score_independent_columns(frame.columns)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Explicit H2 input CSV is missing required columns: {missing}")
    result = frame.copy()
    result["dataset"] = _required_nonempty_strings(result, "dataset")
    result["sample_id"] = _required_nonempty_strings(result, "sample_id")
    if result["label"].isna().any() or result["label"].astype(str).str.strip().eq("").any():
        raise ValueError("Input manifest column 'label' contains a missing/empty value")
    if result.duplicated(["dataset", "sample_id"]).any():
        examples = result.loc[result.duplicated(["dataset", "sample_id"], keep=False), ["dataset", "sample_id"]]
        raise ValueError(f"Input manifest has duplicate dataset/sample_id identities: {examples.head(8).to_dict('records')}")
    return result


def _selection_key(seed: int, dataset: str, label: object, sample_id: str) -> str:
    """Return a cross-platform stable pseudo-random key, never Python's hash()."""
    payload = {"dataset": dataset, "label": str(label), "sample_id": sample_id, "seed": int(seed)}
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class FrozenInputManifest:
    """Frozen score-independent input rows plus reproducible provenance."""

    rows: pd.DataFrame
    provenance: Mapping[str, Any]


def freeze_input_manifest(
    input_rows: pd.DataFrame,
    *,
    per_label: int,
    seed: int,
    input_csv_sha256: str | None = None,
) -> FrozenInputManifest:
    """Freeze exactly ``per_label`` examples per dataset/label without scores.

    Selection is a seeded SHA-256 permutation within each dataset/label group.
    Thus reruns are invariant to CSV row order and Python/hash-platform state.
    Every group must be sufficiently populated; silent undersampling would make
    the planned paired panel unbalanced.
    """
    if isinstance(per_label, bool) or int(per_label) <= 0:
        raise ValueError("per_label must be a positive integer")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer, not bool")
    per_label, seed = int(per_label), int(seed)
    input_rows = validate_score_independent_input(input_rows)
    group_sizes = input_rows.groupby(["dataset", "label"], sort=True).size()
    undersized = group_sizes[group_sizes < per_label]
    if not undersized.empty:
        details = {f"{dataset}/{label}": int(size) for (dataset, label), size in undersized.items()}
        raise ValueError(f"Cannot freeze {per_label} examples per dataset/label; undersized groups: {details}")

    result = input_rows.copy()
    result["selection_key_sha256"] = [
        _selection_key(seed, dataset, label, sample_id)
        for dataset, label, sample_id in result[["dataset", "label", "sample_id"]].itertuples(index=False, name=None)
    ]
    result = result.sort_values(
        ["dataset", "label", "selection_key_sha256", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    result["selection_rank"] = result.groupby(["dataset", "label"], sort=False).cumcount() + 1
    result = result.loc[result["selection_rank"] <= per_label].copy()
    result["manifest_version"] = PRE_SCORE_MANIFEST_VERSION
    result["selection_seed"] = seed
    result["selection_method"] = "sha256(seed,dataset,label,sample_id)_within_dataset_label"
    result["source_input_csv_sha256"] = input_csv_sha256
    result = result.sort_values(["dataset", "label", "selection_rank", "sample_id"], kind="stable").reset_index(drop=True)
    if len(result) != len(group_sizes) * per_label:
        raise RuntimeError("Internal error: frozen H2 panel does not contain the required balanced count")

    canonical_rows = result.to_dict(orient="records")
    provenance: dict[str, Any] = {
        "artifact_kind": "h2_score_independent_input_freeze",
        "manifest_version": PRE_SCORE_MANIFEST_VERSION,
        "claim_guard": "Input selection only; it contains no detector outputs and establishes no H2 result.",
        "detector_scores_read": False,
        "detector_scoring_allowed": False,
        "selection_seed": seed,
        "per_dataset_label": per_label,
        "selection_method": "sha256(seed,dataset,label,sample_id)_within_dataset_label",
        "input_csv_sha256": input_csv_sha256,
        "n_input_rows": int(len(input_rows)),
        "n_frozen_rows": int(len(result)),
        "dataset_label_counts": {
            f"{dataset}/{label}": int(size) for (dataset, label), size in group_sizes.sort_index().items()
        },
        "frozen_rows_sha256": _sha256_bytes(_canonical_json(canonical_rows).encode("utf-8")),
    }
    return FrozenInputManifest(rows=result, provenance=provenance)


def freeze_input_manifest_from_csv(path: str | Path, *, per_label: int, seed: int) -> FrozenInputManifest:
    """Read only the caller-supplied CSV and freeze it under its file hash."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Explicit H2 input CSV is missing: {source}")
    if source.suffix.casefold() != ".csv":
        raise ValueError(f"Explicit H2 input must be CSV, got: {source}")
    return freeze_input_manifest(pd.read_csv(source), per_label=per_label, seed=seed, input_csv_sha256=sha256_file(source))


@dataclass(frozen=True)
class ArmDefinition:
    """A pre-registered waveform arm with an explicit target-feature condition."""

    arm_id: str
    transform: str
    parameters: Mapping[str, float]
    negative_control: bool
    target_feature: str
    target_direction: str
    target_tolerance: float

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["parameters"] = dict(sorted(self.parameters.items()))
        record["definition_sha256"] = _sha256_bytes(_canonical_json(record).encode("utf-8"))
        return record


def registered_crest_factor_arms() -> tuple[ArmDefinition, ...]:
    """Return the fixed H2 crest arms and amplitude-invariance controls.

    The 0.1-dB gain control is deliberately fixed before H2 data/model results:
    it is nonzero while remaining within the protocol's +/-0.2-LU gate absent
    clipping.  Controls target invariance of crest factor; neither chooses an
    arm based on waveform measurements or detector responses.
    """
    return (
        ArmDefinition(
            arm_id="drc_cf3",
            transform="drc_crest_factor",
            parameters={"target_reduction_db": 3.0},
            negative_control=False,
            target_feature="crest_factor_db",
            target_direction="decrease",
            target_tolerance=0.0,
        ),
        ArmDefinition(
            arm_id="drc_cf6",
            transform="drc_crest_factor",
            parameters={"target_reduction_db": 6.0},
            negative_control=False,
            target_feature="crest_factor_db",
            target_direction="decrease",
            target_tolerance=0.0,
        ),
        ArmDefinition(
            arm_id="small_gain_plus_0p1db",
            transform="gain",
            parameters={"gain_db": 0.1},
            negative_control=True,
            target_feature="crest_factor_db",
            target_direction="invariant",
            target_tolerance=1e-6,
        ),
        ArmDefinition(
            arm_id="polarity",
            transform="polarity",
            parameters={},
            negative_control=True,
            target_feature="crest_factor_db",
            target_direction="invariant",
            target_tolerance=1e-6,
        ),
    )


def arm_definitions_frame(arms: Sequence[ArmDefinition] | None = None) -> pd.DataFrame:
    """Materialize the exact arms and content hashes for a pre-score artifact."""
    definitions = tuple(registered_crest_factor_arms() if arms is None else arms)
    if not definitions:
        raise ValueError("At least one pre-registered H2 arm is required")
    identifiers = [arm.arm_id for arm in definitions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate H2 arm identifiers: {identifiers}")
    return pd.DataFrame([arm.as_record() for arm in definitions]).sort_values("arm_id", kind="stable").reset_index(drop=True)


def apply_registered_arm(audio: np.ndarray, sample_rate: int, arm: ArmDefinition) -> TransformResult:
    """Apply one fixed arm, without loading ASR or a detector."""
    if arm.transform == "drc_crest_factor":
        return compress_crest_factor(audio, sample_rate, target_reduction_db=float(arm.parameters["target_reduction_db"]))
    if arm.transform == "gain":
        return apply_gain(audio, sample_rate, gain_db=float(arm.parameters["gain_db"]))
    if arm.transform == "polarity":
        return apply_polarity(audio, sample_rate)
    raise ValueError(f"Unsupported pre-registered H2 transform: {arm.transform}")


@dataclass(frozen=True)
class QualityGateConfig:
    """The protocol gates, represented as values rather than implicit code paths."""

    stoi_minimum: float = 0.95
    wer_maximum: float = 0.05
    loudness_delta_maximum_lu: float = 0.2
    clipping_fraction_maximum: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.stoi_minimum <= 1.0:
            raise ValueError("stoi_minimum must be within [0, 1]")
        if self.wer_maximum < 0.0 or self.loudness_delta_maximum_lu < 0.0 or self.clipping_fraction_maximum < 0.0:
            raise ValueError("WER, loudness, and clipping thresholds must be non-negative")


Transcriber = Callable[[np.ndarray, int], str]
StoiMeasure = Callable[[np.ndarray, np.ndarray, int], float]
FeatureExtractor = Callable[[np.ndarray, int], Mapping[str, float]]


def _default_stoi(original: np.ndarray, transformed: np.ndarray, sample_rate: int) -> float:
    from pystoi import stoi

    return float(stoi(original.astype(np.float64), transformed.astype(np.float64), sample_rate, extended=False))


def _frozen_features(audio: np.ndarray, sample_rate: int) -> Mapping[str, float]:
    """Measure the registered full-waveform view, including deterministic resampling."""
    return compute_features(waveform_views(audio, sample_rate)["full_waveform"])


def waveform_sha256(audio: np.ndarray) -> str:
    """Hash canonical contiguous float32 samples (sample rate is recorded separately)."""
    canonical = np.ascontiguousarray(np.asarray(audio, dtype="<f4").reshape(-1))
    return _sha256_bytes(canonical.tobytes())


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _target_direction_pass(before: float | None, after: float | None, arm: ArmDefinition) -> bool:
    if before is None or after is None:
        return False
    if arm.target_direction == "decrease":
        return after < before
    if arm.target_direction == "increase":
        return after > before
    if arm.target_direction == "invariant":
        return abs(after - before) <= arm.target_tolerance
    raise ValueError(f"Unsupported H2 target direction: {arm.target_direction}")


def _base_quality_row(manifest_row: Mapping[str, Any], arm: ArmDefinition) -> dict[str, Any]:
    assert_score_independent_columns(list(manifest_row))
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in manifest_row]
    if missing:
        raise ValueError(f"Frozen H2 input row lacks required identity fields: {missing}")
    result: dict[str, Any] = {
        "quality_row_version": QUALITY_ROW_VERSION,
        "dataset": str(manifest_row["dataset"]),
        "sample_id": str(manifest_row["sample_id"]),
        "label": manifest_row["label"],
        "source_id": manifest_row.get("source_id"),
        "speaker_id": manifest_row.get("speaker_id"),
        "input_manifest_version": manifest_row.get("manifest_version"),
        "source_input_csv_sha256": manifest_row.get("source_input_csv_sha256"),
        "selection_seed": manifest_row.get("selection_seed"),
        "selection_method": manifest_row.get("selection_method"),
        "selection_key_sha256": manifest_row.get("selection_key_sha256"),
        "arm": arm.arm_id,
        "arm_definition_sha256": arm.as_record()["definition_sha256"],
        "arm_parameters_json": _canonical_json(dict(sorted(arm.parameters.items()))),
        "target_feature": arm.target_feature,
        "target_direction": arm.target_direction,
        "negative_control": arm.negative_control,
        # A separate scoring stage must refuse rows until this field is true.
        "detector_stage": "blocked_pending_quality_freeze",
        "retained": False,
        "failure_reasons_json": "[]",
        "pass_stoi": False,
        "pass_wer": False,
        "pass_loudness": False,
        "pass_clipping": False,
        "pass_target_direction": False,
    }
    result["pair_id"] = _sha256_bytes(
        _canonical_json({key: result[key] for key in ("dataset", "sample_id", "label", "selection_key_sha256", "arm")}).encode("utf-8")
    )
    return result


def evaluate_quality_pair(
    manifest_row: Mapping[str, Any],
    *,
    audio: np.ndarray,
    sample_rate: int,
    arm: ArmDefinition,
    transcribe: Transcriber,
    gates: QualityGateConfig = QualityGateConfig(),
    stoi_measure: StoiMeasure | None = None,
    feature_extractor: FeatureExtractor | None = None,
) -> dict[str, Any]:
    """Create one retained-or-failed quality row before any detector execution.

    The caller supplies the decoded waveform and a pinned ASR callable.  Every
    transform/metric/transcript exception is retained as an explicit failed row
    instead of deleting the pair.  No detector object, detector callback, or
    detector-response field exists in this API.
    """
    gates.validate()
    row = _base_quality_row(manifest_row, arm)
    errors: list[str] = []
    original = np.asarray(audio, dtype=np.float32).reshape(-1)
    if original.size == 0 or not np.isfinite(original).all() or int(sample_rate) <= 0:
        errors.append("invalid_original_waveform_or_sample_rate")
        row["failure_reasons_json"] = _canonical_json(errors)
        return row
    rate = int(sample_rate)
    row.update(
        {
            "sample_rate_hz": rate,
            "original_samples": int(original.size),
            "original_wave_sha256": waveform_sha256(original),
        }
    )
    try:
        transformed_result = apply_registered_arm(original, rate, arm)
        transformed = np.asarray(transformed_result.waveform, dtype=np.float32).reshape(-1)
        diagnostics = dict(transformed_result.diagnostics)
    except Exception as error:  # failure must remain represented in the manifest
        errors.append(f"transform_error:{type(error).__name__}")
        row["failure_reasons_json"] = _canonical_json(errors)
        return row
    row.update({f"transform_{key}": value for key, value in diagnostics.items()})
    row.update(
        {
            "transformed_samples": int(transformed.size),
            "transformed_wave_sha256": waveform_sha256(transformed),
            "original_lufs": _finite_number(diagnostics.get("input_integrated_lufs")),
            "transformed_lufs": _finite_number(diagnostics.get("output_integrated_lufs")),
            "loudness_delta_lu": _finite_number(diagnostics.get("loudness_delta_lu")),
            "original_clipping_fraction": _finite_number(diagnostics.get("input_clipping_fraction")),
            "transformed_clipping_fraction": _finite_number(diagnostics.get("output_clipping_fraction")),
        }
    )

    extractor = _frozen_features if feature_extractor is None else feature_extractor
    original_features: Mapping[str, float] | None = None
    transformed_features: Mapping[str, float] | None = None
    try:
        original_features = extractor(original, rate)
        transformed_features = extractor(transformed, rate)
        missing_features = set(FEATURE_NAMES).difference(original_features) | set(FEATURE_NAMES).difference(transformed_features)
        if missing_features:
            raise ValueError(f"Feature extractor omitted registered features: {sorted(missing_features)}")
        deltas = {
            feature: (
                _finite_number(transformed_features[feature]) - _finite_number(original_features[feature])
                if _finite_number(transformed_features[feature]) is not None and _finite_number(original_features[feature]) is not None
                else None
            )
            for feature in FEATURE_NAMES
        }
        before = _finite_number(original_features[arm.target_feature])
        after = _finite_number(transformed_features[arm.target_feature])
        row["target_feature_before"] = before
        row["target_feature_after"] = after
        row["all_feature_deltas_json"] = _canonical_json(deltas)
        row["pass_target_direction"] = _target_direction_pass(before, after, arm)
        if not row["pass_target_direction"]:
            errors.append("target_direction_failed")
    except Exception as error:
        errors.append(f"feature_error:{type(error).__name__}")
        row["all_feature_deltas_json"] = None

    loudness_delta = row.get("loudness_delta_lu")
    row["pass_loudness"] = bool(loudness_delta is not None and abs(loudness_delta) <= gates.loudness_delta_maximum_lu)
    if not row["pass_loudness"]:
        errors.append("loudness_gate_failed_or_unavailable")
    original_clipping = row.get("original_clipping_fraction")
    transformed_clipping = row.get("transformed_clipping_fraction")
    row["pass_clipping"] = bool(
        original_clipping is not None
        and transformed_clipping is not None
        and original_clipping <= gates.clipping_fraction_maximum
        and transformed_clipping <= gates.clipping_fraction_maximum
    )
    if not row["pass_clipping"]:
        errors.append("clipping_gate_failed_or_unavailable")

    measure_stoi = _default_stoi if stoi_measure is None else stoi_measure
    try:
        row["stoi"] = _finite_number(measure_stoi(original, transformed, rate))
        row["pass_stoi"] = bool(row["stoi"] is not None and row["stoi"] >= gates.stoi_minimum)
        if not row["pass_stoi"]:
            errors.append("stoi_gate_failed")
    except Exception as error:
        errors.append(f"stoi_error:{type(error).__name__}")

    try:
        original_transcript = transcribe(original, rate)
        transformed_transcript = transcribe(transformed, rate)
        if not isinstance(original_transcript, str) or not isinstance(transformed_transcript, str):
            raise TypeError("ASR callable must return transcript strings")
        wer: WordErrorResult = normalized_word_error(original_transcript, transformed_transcript)
        row.update(
            {
                "original_transcript": original_transcript,
                "transformed_transcript": transformed_transcript,
                "original_words_json": _canonical_json(list(wer.reference_words)),
                "transformed_words_json": _canonical_json(list(wer.hypothesis_words)),
                "wer_errors": wer.errors,
                "wer": wer.word_error_rate,
                "pass_wer": bool(wer.word_error_rate <= gates.wer_maximum),
            }
        )
        if not row["pass_wer"]:
            errors.append("wer_gate_failed")
    except Exception as error:
        errors.append(f"asr_or_wer_error:{type(error).__name__}")

    retained = all(
        bool(row[name]) for name in ("pass_stoi", "pass_wer", "pass_loudness", "pass_clipping", "pass_target_direction")
    )
    row["retained"] = retained
    row["quality_status"] = "passed" if retained else "failed"
    row["failure_reasons_json"] = _canonical_json(errors)
    return row


def quality_rows_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Validate a pre-score quality table before a caller persists it.

    Failure rows are deliberately retained.  Any response-like column triggers
    a hard failure so this layer cannot become a covert detector-output store.
    """
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Refusing to write an empty H2 quality table")
    assert_score_independent_columns(result.columns)
    required = {"pair_id", "detector_stage", "retained", "failure_reasons_json", "pass_stoi", "pass_wer", "pass_loudness", "pass_clipping", "pass_target_direction"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"H2 quality rows lack required transparency fields: {missing}")
    if not result["detector_stage"].eq("blocked_pending_quality_freeze").all():
        raise ValueError("Pre-score quality rows must remain blocked from detector execution")
    if result["pair_id"].duplicated().any():
        raise ValueError("H2 quality table has duplicate pair_id values")
    return result.sort_values(["dataset", "label", "sample_id", "arm"], kind="stable").reset_index(drop=True)
