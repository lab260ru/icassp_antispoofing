"""Build one score-free H2 input CSV from a pinned Arena labels table.

The builder is deliberately narrower than a dataset reader: it resolves one
named entry from ``data/arena-index.yaml`` and invokes ``arena_io.load_labels``
only. It does not enumerate audio shards, compute features, access published
score artifacts, load ASR, or load a detector.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.arena_io import load_labels


INPUT_CSV_VERSION = "h2_score_free_input_csv_v1"
REQUIRED_LABEL_COLUMNS = ("sample_id", "source_id", "label")
RESPONSE_MARKERS = ("score", "logit", "probability", "prediction")


def sha256_file(path: str | Path) -> str:
    """Hash a source artifact in chunks without parsing a second data table."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normal_column_name(column: object) -> str:
    return "".join(character for character in str(column).casefold() if character.isalnum())


def assert_score_free_columns(columns: list[object] | pd.Index) -> None:
    """Prevent response-like columns from entering the pre-score CSV."""
    response_columns = [
        str(column)
        for column in columns
        if any(marker in _normal_column_name(column) for marker in RESPONSE_MARKERS)
    ]
    if response_columns:
        raise ValueError(f"Score-free H2 input cannot contain response-like columns: {response_columns}")


def _pinned_file_hash(dataset: Mapping[str, Any], relative_path: str) -> str | None:
    for entry in dataset.get("pinned_files", []):
        if entry.get("path") == relative_path:
            digest = entry.get("sha256")
            return str(digest) if digest is not None else None
    return None


def _validate_label_rows(labels: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"arena_io.load_labels returned incompatible rows; missing {missing}")
    if labels.empty:
        raise ValueError("arena_io.load_labels returned no labels")
    result = labels.copy()
    for column in ("sample_id", "source_id"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Loaded label rows contain an empty {column}")
        result[column] = result[column].astype(str).str.strip()
    if result["label"].isna().any() or result["label"].astype(str).str.strip().eq("").any():
        raise ValueError("Loaded label rows contain an empty label")
    if result["sample_id"].duplicated().any():
        duplicates = result.loc[result["sample_id"].duplicated(keep=False), "sample_id"].head(8).tolist()
        raise ValueError(f"Loaded label rows have duplicate sample IDs: {duplicates}")
    return result


def build_score_free_input_rows(
    index: Mapping[str, Any],
    dataset_name: str,
    *,
    labels_loader: Callable[[dict[str, Any]], pd.DataFrame] = load_labels,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the complete score-free input table for one explicit dataset.

    ``labels_loader`` defaults to :func:`src.arena_io.load_labels`; it is an
    injection point for synthetic unit tests, not an alternate data path for
    production. Any optional ``speaker_id`` returned by that loader is
    preserved, but the standard Arena label loader currently exposes only the
    three required label fields.
    """
    datasets = index.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("Arena index lacks a 'datasets' mapping")
    if dataset_name not in datasets:
        raise ValueError(f"Unknown Arena dataset {dataset_name!r}; known datasets: {sorted(datasets)}")
    dataset = datasets[dataset_name]
    if not isinstance(dataset, dict):
        raise ValueError(f"Arena dataset entry {dataset_name!r} is not a mapping")
    revision = dataset.get("revision")
    if revision is None or not str(revision).strip():
        raise ValueError(f"Arena dataset entry {dataset_name!r} lacks a pinned revision")
    files = dataset.get("files")
    if not isinstance(files, Mapping) or not files.get("labels"):
        raise ValueError(f"Arena dataset entry {dataset_name!r} lacks files.labels")
    if not dataset.get("local_dir"):
        raise ValueError(f"Arena dataset entry {dataset_name!r} lacks local_dir")

    labels_path = Path(str(dataset["local_dir"])) / str(files["labels"])
    if not labels_path.is_file():
        raise FileNotFoundError(f"Pinned Arena labels table is missing: {labels_path}")
    observed_hash = sha256_file(labels_path)
    expected_hash = _pinned_file_hash(dataset, str(files["labels"]))
    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError(
            f"Pinned labels SHA-256 mismatch for {dataset_name}: expected {expected_hash}, observed {observed_hash}"
        )

    labels = _validate_label_rows(labels_loader(dataset))
    result = labels.loc[:, list(REQUIRED_LABEL_COLUMNS)].copy()
    if "speaker_id" in labels.columns:
        result["speaker_id"] = labels["speaker_id"]
    result.insert(0, "dataset", dataset_name)
    result["dataset_revision"] = str(revision)
    assert_score_free_columns(result.columns)
    result = result.sort_values(["dataset", "label", "sample_id"], kind="stable").reset_index(drop=True)

    counts = result.groupby("label", sort=True).size()
    provenance: dict[str, Any] = {
        "artifact_kind": "h2_score_free_input_csv",
        "version": INPUT_CSV_VERSION,
        "claim_guard": "Label-derived input rows only; no audio, feature, score, ASR, or detector was read.",
        "dataset": {
            "name": dataset_name,
            "repo_id": dataset.get("repo_id"),
            "revision": str(revision),
            "source_revision_resolved": dataset.get("source_revision_resolved"),
        },
        "labels": {
            "relative_path": str(files["labels"]),
            "local_path": str(labels_path),
            "sha256": observed_hash,
            "pinned_sha256": expected_hash,
            "pinned_sha256_verified": expected_hash is not None,
            "loader": "src.arena_io.load_labels",
        },
        "source_access": {
            "audio_read": False,
            "feature_read": False,
            "score_read": False,
            "asr_loaded": False,
            "detector_loaded": False,
        },
        "n_rows": int(len(result)),
        "label_counts": {str(label): int(count) for label, count in counts.items()},
        "columns": result.columns.tolist(),
        "rows_sha256": hashlib.sha256(_canonical_json(result.to_dict(orient="records")).encode("utf-8")).hexdigest(),
    }
    return result, provenance


def build_score_free_input_from_index(
    index_path: str | Path,
    dataset_name: str,
    *,
    labels_loader: Callable[[dict[str, Any]], pd.DataFrame] = load_labels,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve one explicit dataset name against the supplied Arena index YAML."""
    source = Path(index_path)
    if not source.is_file():
        raise FileNotFoundError(f"Arena index is missing: {source}")
    index = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(index, Mapping):
        raise ValueError(f"Arena index is not a YAML mapping: {source}")
    rows, provenance = build_score_free_input_rows(index, dataset_name, labels_loader=labels_loader)
    provenance["arena_index"] = {"path": str(source.resolve()), "sha256": sha256_file(source)}
    return rows, provenance


def write_score_free_input_artifacts(
    rows: pd.DataFrame,
    provenance: Mapping[str, Any],
    *,
    output_csv: str | Path,
    output_provenance: str | Path,
) -> None:
    """Write distinct, non-overwriting CSV/JSON artifacts after final checks."""
    assert_score_free_columns(rows.columns)
    csv_path = Path(output_csv).resolve()
    provenance_path = Path(output_provenance).resolve()
    if csv_path == provenance_path:
        raise ValueError("H2 input CSV and provenance JSON paths must differ")
    existing = [str(path) for path in (csv_path, provenance_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite H2 input artifact(s): {existing}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False)
    finalized_provenance = dict(provenance)
    # Bind the provenance JSON to the exact emitted CSV bytes.  This is needed
    # by downstream score-blind panel freezes; a logical row hash alone cannot
    # detect a later textual CSV substitution.
    finalized_provenance["output_csv"] = {
        "path": str(csv_path),
        "sha256": sha256_file(csv_path),
        "n_rows": int(len(rows)),
    }
    provenance_path.write_text(json.dumps(finalized_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
