"""Input freezing utilities for the H8-SF robust score-fusion study.

The first H8 stage is intentionally source-only.  It establishes one raw-score
polarity per frozen system and materializes a deterministic corpus-by-class
source manifest.  Target labels and target metrics are not part of this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import rankdata
from scipy.special import ndtri


H8_VERSION = "h8sf_source_freeze_v2"
SOURCE_DATASETS = ("ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF")
TARGET_DATASETS = ("CFAD", "CVoiceFake_small", "DECRO", "LibriSeVoc", "XMAD")
MODELS = (
    "Spectra-AASIST",
    "AASIST",
    "Res2TCNGuard",
    "RawTFNet",
    "WhisperMFCCMesoNet",
    "W2V2-AASIST",
    "XLSR-SLS",
    "RawBMamba",
)
LABELS = (0, 1)
SOURCE_CAP_PER_DATASET_LABEL = 10_000
SOURCE_SELECTION_SEED = 2608
MIN_ABSOLUTE_ORIENTATION_CORRELATION = 0.02


@dataclass(frozen=True)
class SourceFreezeArtifacts:
    """Byte-identifiable source-only H8 input outputs."""

    output_dir: Path
    manifest_path: Path
    provenance_path: Path
    orientation_path: Path


@dataclass(frozen=True)
class TargetFeatureArtifacts:
    """Locations for a label-free H8 target score representation."""

    output_dir: Path
    features_path: Path
    provenance_path: Path


def canonical_json(value: object) -> str:
    """Return stable pretty JSON for an H8 compact provenance record."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    """Hash a file without depending on its semantic content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sample_id(value: object) -> str:
    """Normalize only a terminal audio suffix, preserving meaningful dots.

    CVoiceFake_small has literal score IDs such as
    ``...multi_band_melgan.v2_generated...``.  ``Path.stem`` would collapse
    that family to a non-unique prefix, so H8 intentionally removes only a
    terminal audio extension after discarding any directory prefix.
    """
    name = Path(str(value)).name
    for suffix in (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _load_index(index_path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read H8 Arena index: {index_path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), dict) or not isinstance(value.get("models"), dict):
        raise ValueError("H8 Arena index must contain dataset and model mappings")
    return value


def _require_records(index: Mapping[str, Any], datasets: Sequence[str], models: Sequence[str]) -> None:
    dataset_records = index["datasets"]
    model_records = index["models"]
    missing_datasets = [dataset for dataset in datasets if dataset not in dataset_records]
    missing_models = [model for model in models if model not in model_records]
    if missing_datasets or missing_models:
        raise ValueError(f"H8 index missing datasets={missing_datasets} models={missing_models}")
    for model in models:
        artifacts = model_records[model].get("score_artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"H8 model lacks score artifacts: {model}")
        missing_scores = [dataset for dataset in datasets if not isinstance(artifacts.get(dataset), Mapping) or not isinstance(artifacts[dataset].get("scores"), Mapping)]
        if missing_scores:
            raise ValueError(f"H8 model {model} lacks source score artifacts: {missing_scores}")


def _labels_path(dataset_record: Mapping[str, Any]) -> Path:
    files = dataset_record.get("files")
    if not isinstance(files, Mapping) or not isinstance(dataset_record.get("local_dir"), str) or not isinstance(files.get("labels"), str):
        raise ValueError("H8 dataset record lacks local_dir/files.labels")
    path = Path(dataset_record["local_dir"]) / str(files["labels"])
    if not path.is_file():
        raise FileNotFoundError(f"H8 source label table unavailable: {path}")
    return path


def _score_path(model_record: Mapping[str, Any], dataset: str) -> Path:
    artifacts = model_record.get("score_artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(model_record.get("local_dir"), str):
        raise ValueError("H8 model record lacks local_dir/score_artifacts")
    artifact = artifacts.get(dataset)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("scores"), Mapping):
        raise ValueError(f"H8 missing score artifact record for {dataset}")
    relative_path = artifact["scores"].get("path")
    if not isinstance(relative_path, str):
        raise ValueError(f"H8 score path absent for {dataset}")
    path = Path(model_record["local_dir"]) / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"H8 source score artifact unavailable: {path}")
    expected_size = artifact["scores"].get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"H8 source score size mismatch for {path}: {path.stat().st_size} != {expected_size}")
    return path


def _read_labels(path: Path) -> pd.DataFrame:
    schema = pq.ParquetFile(path).schema_arrow
    names = list(schema.names)
    id_column = next((name for name in ("utterance_id", "sample_id", "path") if name in names), None)
    if id_column is None or "label" not in names:
        raise ValueError(f"H8 source labels have unsupported schema: {path}")
    table = pq.read_table(path, columns=[id_column, "label"]).to_pandas()
    labels = pd.DataFrame(
        {
            "sample_id": table[id_column].map(normalize_sample_id),
            "label": pd.to_numeric(table["label"], errors="raise").astype("int8"),
        }
    )
    if labels.empty or labels["sample_id"].duplicated().any() or not labels["label"].isin(LABELS).all():
        raise ValueError(f"H8 source labels are invalid or non-binary: {path}")
    return labels


def parse_scores(path: Path) -> pd.DataFrame:
    """Parse the standard two-column Arena score format with strict IDs."""
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"H8 expected two score fields at {path}:{line_number}")
            try:
                score = float(fields[1])
            except ValueError as error:
                raise ValueError(f"H8 non-numeric score at {path}:{line_number}") from error
            if not np.isfinite(score):
                raise ValueError(f"H8 non-finite score at {path}:{line_number}")
            rows.append((normalize_sample_id(fields[0]), score))
    frame = pd.DataFrame(rows, columns=["sample_id", "raw_score"])
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError(f"H8 scores are empty or have duplicate IDs: {path}")
    return frame


def _rank_correlation_with_labels(frame: pd.DataFrame) -> float:
    rank = rankdata(frame["raw_score"].to_numpy(dtype=float), method="average")
    labels = frame["label"].to_numpy(dtype=float)
    correlation = float(np.corrcoef(rank, labels)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("H8 source score orientation is undefined")
    return correlation


def _selection_key(dataset: str, label: int, sample_id: str) -> str:
    payload = f"H8SF|{SOURCE_SELECTION_SEED}|{dataset}|{label}|{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_rows(dataset: str, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        ids = sorted(labels.loc[labels["label"].eq(label), "sample_id"].astype(str))
        ranked = sorted((_selection_key(dataset, label, sample_id), sample_id) for sample_id in ids)
        if not ranked:
            raise ValueError(f"H8 source {dataset} has no label {label} rows")
        for rank, (selection_key, sample_id) in enumerate(ranked[:SOURCE_CAP_PER_DATASET_LABEL], start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "sample_id": sample_id,
                    "selection_rank": rank,
                    "selection_key_sha256": selection_key,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "label", "selection_rank"]).reset_index(drop=True)


def _directory_for_new_outputs(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"H8 output directory must be absolute: {path}")
    if path.exists():
        raise FileExistsError(f"H8 refuses to overwrite an existing output directory: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def freeze_source_inputs(
    *,
    index_path: Path,
    output_dir: Path,
    datasets: Sequence[str] = SOURCE_DATASETS,
    models: Sequence[str] = MODELS,
) -> SourceFreezeArtifacts:
    """Freeze source IDs and per-model orientation using sources only.

    The caller is deliberately unable to pass a target dataset here.  A source
    file with incomplete score coverage or inconsistent orientation stops the
    run before it can write a potentially misleading manifest.
    """
    index_path = Path(index_path).resolve()
    index = _load_index(index_path)
    datasets = tuple(datasets)
    models = tuple(models)
    if not datasets or not models or len(set(datasets)) != len(datasets) or len(set(models)) != len(models):
        raise ValueError("H8 source datasets/models must be non-empty and unique")
    _require_records(index, datasets, models)

    labels_by_dataset: dict[str, pd.DataFrame] = {}
    label_provenance: dict[str, object] = {}
    per_model: dict[str, list[dict[str, object]]] = {model: [] for model in models}
    score_provenance: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        label_path = _labels_path(index["datasets"][dataset])
        labels = _read_labels(label_path)
        labels_by_dataset[dataset] = labels
        label_provenance[dataset] = {
            "path": str(label_path),
            "sha256": sha256_file(label_path),
            "size_bytes": label_path.stat().st_size,
            "label_counts": {str(label): int(labels["label"].eq(label).sum()) for label in LABELS},
            "dataset_revision": index["datasets"][dataset].get("revision"),
        }
        for model in models:
            score_path = _score_path(index["models"][model], dataset)
            scores = parse_scores(score_path)
            joined = scores.merge(labels, on="sample_id", how="inner", validate="one_to_one")
            if len(joined) != len(labels) or len(joined) != len(scores):
                raise ValueError(
                    f"H8 source coverage mismatch for {model}/{dataset}: "
                    f"labels={len(labels)} scores={len(scores)} joined={len(joined)}"
                )
            correlation = _rank_correlation_with_labels(joined)
            if abs(correlation) < MIN_ABSOLUTE_ORIENTATION_CORRELATION:
                raise ValueError(f"H8 ambiguous source orientation for {model}/{dataset}: {correlation:.6f}")
            score_provenance.setdefault(model, {})[dataset] = {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
                "size_bytes": score_path.stat().st_size,
                "model_revision": index["models"][model].get("revision"),
                "n_rows": len(scores),
                "n_joined": len(joined),
                "rank_label_correlation": correlation,
            }
            per_model[model].append({"dataset": dataset, "rank_label_correlation": correlation})

    orientations: dict[str, object] = {}
    for model, rows in per_model.items():
        correlations = [float(row["rank_label_correlation"]) for row in rows]
        signs = {int(np.sign(value)) for value in correlations}
        if len(signs) != 1 or 0 in signs:
            raise ValueError(f"H8 source orientation changes across source datasets for {model}: {correlations}")
        multiplier = int(next(iter(signs)))
        orientations[model] = {
            "orientation_multiplier": multiplier,
            "orientation": "raw_is_spoof" if multiplier > 0 else "negated_raw_is_spoof",
            "source_correlation_by_dataset": rows,
            "source_correlation_mean": float(np.mean(correlations)),
        }

    manifest = pd.concat([_select_rows(dataset, labels_by_dataset[dataset]) for dataset in datasets], ignore_index=True)
    if manifest.duplicated(["dataset", "sample_id"]).any() or not manifest["label"].isin(LABELS).all():
        raise ValueError("H8 internal source manifest invariant failed")

    output_dir = _directory_for_new_outputs(Path(output_dir))
    manifest_path = output_dir / "source_manifest.csv"
    orientation_path = output_dir / "source_orientation.json"
    provenance_path = output_dir / "source_freeze_provenance.json"
    manifest.to_csv(manifest_path, index=False)
    orientation_path.write_text(canonical_json({"version": H8_VERSION, "models": orientations}), encoding="utf-8")
    provenance = {
        "artifact_kind": "h8sf_source_only_input_freeze",
        "version": H8_VERSION,
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "source_datasets": list(datasets),
        "models": list(models),
        "selection": {
            "seed": SOURCE_SELECTION_SEED,
            "cap_per_dataset_label": SOURCE_CAP_PER_DATASET_LABEL,
            "key_template": "sha256('H8SF|2608|dataset|label|sample_id')",
            "n_rows": int(len(manifest)),
            "counts": {
                f"{dataset}|{label}": int(
                    ((manifest["dataset"] == dataset) & (manifest["label"] == label)).sum()
                )
                for dataset in datasets
                for label in LABELS
            },
        },
        "source_label_artifacts": label_provenance,
        "source_score_artifacts": score_provenance,
        "orientation": orientations,
        "target_labels_read": False,
        "target_scores_read": False,
        "target_metrics_read": False,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_orientation_sha256": sha256_file(orientation_path),
    }
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return SourceFreezeArtifacts(output_dir, manifest_path, provenance_path, orientation_path)


def _read_orientation(path: Path, models: Sequence[str]) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"H8 source orientation is unavailable: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H8 source orientation is not valid JSON: {path}") from error
    if payload.get("version") != H8_VERSION or not isinstance(payload.get("models"), Mapping):
        raise ValueError("H8 source orientation has an unexpected contract")
    output: dict[str, int] = {}
    for model in models:
        row = payload["models"].get(model)
        if not isinstance(row, Mapping) or row.get("orientation_multiplier") not in (-1, 1):
            raise ValueError(f"H8 source orientation is missing a valid multiplier for {model}")
        output[model] = int(row["orientation_multiplier"])
    return output


def _rank_probit(raw_scores: pd.Series) -> np.ndarray:
    values = raw_scores.to_numpy(dtype=np.float64, copy=False)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("H8 rank/probit input must be non-empty and finite")
    ranks = rankdata(values, method="average")
    probabilities = np.clip((ranks - 0.5) / len(ranks), 1e-4, 1.0 - 1e-4)
    transformed = ndtri(probabilities)
    if not np.isfinite(transformed).all():
        raise ValueError("H8 rank/probit transform emitted a non-finite value")
    return np.asarray(transformed, dtype=np.float32)


def feature_column(model: str) -> str:
    """Return the literal, deterministic score feature name for one model."""
    return f"rank_probit__{model}"


def materialize_label_free_target_features(
    *,
    index_path: Path,
    orientation_path: Path,
    output_dir: Path,
    datasets: Sequence[str] = TARGET_DATASETS,
    models: Sequence[str] = MODELS,
) -> TargetFeatureArtifacts:
    """Write only common-ID rank/probit target features, never target labels.

    Every target model's empirical CDF is calculated from its full unlabeled
    score batch before restricting to the eight-model common intersection.
    This is the predeclared transductive, label-free adaptation step.
    """
    index_path = Path(index_path).resolve()
    orientation_path = Path(orientation_path).resolve()
    index = _load_index(index_path)
    datasets = tuple(datasets)
    models = tuple(models)
    if not datasets or not models or len(set(datasets)) != len(datasets) or len(set(models)) != len(models):
        raise ValueError("H8 target datasets/models must be non-empty and unique")
    _require_records(index, datasets, models)
    orientations = _read_orientation(orientation_path, models)
    output_rows: list[pd.DataFrame] = []
    artifact_ledger: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        per_model: list[pd.DataFrame] = []
        artifact_ledger[dataset] = {}
        for model in models:
            score_path = _score_path(index["models"][model], dataset)
            scores = parse_scores(score_path)
            column = feature_column(model)
            transformed = pd.DataFrame(
                {
                    "sample_id": scores["sample_id"].astype(str),
                    column: _rank_probit(scores["raw_score"] * orientations[model]),
                }
            )
            per_model.append(transformed)
            artifact_ledger[dataset][model] = {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
                "size_bytes": score_path.stat().st_size,
                "n_score_rows": int(len(scores)),
                "model_revision": index["models"][model].get("revision"),
                "orientation_multiplier": orientations[model],
            }
        common = per_model[0]
        for next_frame in per_model[1:]:
            common = common.merge(next_frame, on="sample_id", how="inner", validate="one_to_one")
        if common.empty or common["sample_id"].duplicated().any():
            raise ValueError(f"H8 target {dataset} lacks a valid eight-model common score panel")
        common.insert(0, "dataset", dataset)
        expected_trials = index["datasets"][dataset].get("n_trials")
        artifact_ledger[dataset]["common_panel"] = {
            "n_common_rows": int(len(common)),
            "declared_n_trials": int(expected_trials) if expected_trials is not None else None,
            "coverage_vs_declared_trials": float(len(common) / int(expected_trials)) if expected_trials else None,
        }
        output_rows.append(common.sort_values("sample_id").reset_index(drop=True))
    features = pd.concat(output_rows, ignore_index=True)
    if features.duplicated(["dataset", "sample_id"]).any() or not np.isfinite(features[[feature_column(model) for model in models]].to_numpy(dtype=float)).all():
        raise ValueError("H8 target feature output violates unique/finite invariants")
    output_dir = _directory_for_new_outputs(Path(output_dir))
    features_path = output_dir / "target_rank_probit_features.parquet"
    provenance_path = output_dir / "target_feature_provenance.json"
    features.to_parquet(features_path, index=False)
    provenance = {
        "artifact_kind": "h8sf_label_free_target_rank_probit_features",
        "version": H8_VERSION,
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "source_orientation_path": str(orientation_path),
        "source_orientation_sha256": sha256_file(orientation_path),
        "target_datasets": list(datasets),
        "models": list(models),
        "feature_columns": [feature_column(model) for model in models],
        "rank_transform": "Phi^-1(clip((average_rank-0.5)/n,1e-4,1-1e-4))",
        "target_cdf_uses_labels": False,
        "target_labels_read": False,
        "target_metrics_read": False,
        "target_score_artifacts": artifact_ledger,
        "n_rows": int(len(features)),
        "features_sha256": sha256_file(features_path),
    }
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return TargetFeatureArtifacts(output_dir, features_path, provenance_path)
