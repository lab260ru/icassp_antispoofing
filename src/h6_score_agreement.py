"""Locked H6 within-class agreement audit over published raw-score files.

H6 is intentionally isolated from the waveform and feature research loops.
Its first phase reads labels plus score-file *byte identities* only, selects a
label-stratified panel with its own seed, and seals every input.  Its second
phase can read only those sealed original score text artifacts.  It reports an
exhaustive, descriptive matrix; it never emits raw scores or selects a model,
feature, or intervention.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiments/h6_score_agreement/protocol.md"
ARENA_INDEX_PATH = REPO_ROOT / "data/arena-index.yaml"
H6_VERSION = "h6_within_class_published_score_agreement_v1"
SEED = 2611
SAMPLE_CAP_PER_DATASET_LABEL = 5_000
BOOTSTRAP_REPLICATES = 200
CONFIDENCE = 0.95

DATASETS = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
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
MODEL_PAIRS = tuple(combinations(MODELS, 2))
LABEL_ID_COLUMNS = ("utterance_id", "sample_id", "path")
OPTIONAL_CLUSTER_COLUMNS = ("source_id", "speaker_id", "speaker")


@dataclass(frozen=True)
class FreezeArtifacts:
    """Locations and identity of a completed non-overwritable H6 freeze."""

    output_dir: Path
    manifest_path: Path
    provenance_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class RawScoreRead:
    """A raw score parse result that can be represented as an unavailable cell."""

    table: pd.DataFrame
    failure: str | None


def canonical_json(value: object) -> str:
    """Return a stable, human-readable representation for integrity artifacts."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    """Hash bytes in bounded memory without interpreting a score artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sample_id(value: object) -> str:
    """Use the existing Arena convention: basename with one suffix removed."""
    return Path(str(value)).stem


def _valid_identifier(value: object) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    text = str(value).strip()
    return bool(text) and "\x00" not in text


def _absolute_existing(path: Path | str, *, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"H6 {description} path must be absolute: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"H6 {description} is not a regular file: {candidate}")
    return candidate


def _require_locked_index(index_path: Path | str, allowed_index_path: Path | str) -> Path:
    supplied = _absolute_existing(index_path, description="Arena index")
    allowed = _absolute_existing(allowed_index_path, description="locked Arena index")
    if supplied != allowed or supplied.resolve() != allowed.resolve():
        raise ValueError(f"H6 index firewall rejected {supplied}; locked index is {allowed}")
    return supplied


def _load_index(index_path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read H6 Arena index: {index_path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), dict) or not isinstance(value.get("models"), dict):
        raise ValueError("H6 Arena index must contain dataset and model mappings")
    if tuple(value["datasets"].keys()) and not set(DATASETS).issubset(value["datasets"]):
        raise ValueError("H6 Arena index is missing a locked dataset")
    if not set(MODELS).issubset(value["models"]):
        raise ValueError("H6 Arena index is missing a locked model")
    return value


def _label_path(dataset_record: Mapping[str, Any]) -> Path:
    local_dir = dataset_record.get("local_dir")
    files = dataset_record.get("files")
    if not isinstance(local_dir, str) or not isinstance(files, Mapping) or not isinstance(files.get("labels"), str):
        raise ValueError("H6 dataset record lacks local_dir/files.labels")
    return _absolute_existing(Path(local_dir) / str(files["labels"]), description="label table")


def _score_path(model_record: Mapping[str, Any], dataset: str) -> tuple[Path, Mapping[str, Any]]:
    local_dir = model_record.get("local_dir")
    artifacts = model_record.get("score_artifacts")
    if not isinstance(local_dir, str) or not isinstance(artifacts, Mapping):
        raise ValueError("H6 model record lacks local_dir/score_artifacts")
    artifact = artifacts.get(dataset)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("scores"), Mapping):
        raise ValueError(f"H6 model record lacks a raw score artifact for {dataset}")
    scores = artifact["scores"]
    if not isinstance(scores.get("path"), str):
        raise ValueError(f"H6 score artifact lacks a path for {dataset}")
    return _absolute_existing(Path(local_dir) / str(scores["path"]), description="raw score artifact"), scores


def _pinned_file_record(dataset_record: Mapping[str, Any], relative_path: str) -> Mapping[str, Any] | None:
    pinned = dataset_record.get("pinned_files")
    if not isinstance(pinned, list):
        return None
    for record in pinned:
        if isinstance(record, Mapping) and record.get("path") == relative_path:
            return record
    return None


def _label_schema(path: Path) -> tuple[str, str, str | None, list[str]]:
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    if len(columns) != len(set(columns)):
        raise ValueError(f"H6 labels have duplicate schema columns: {path}")
    id_column = next((column for column in LABEL_ID_COLUMNS if column in columns), None)
    if id_column is None or "label" not in columns:
        raise ValueError(f"H6 unsupported label schema: {path}")
    cluster_column = next((column for column in OPTIONAL_CLUSTER_COLUMNS if column in columns), None)
    return id_column, "label", cluster_column, columns


def _read_label_table(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the exact label projection; this function never opens score text."""
    id_column, label_column, cluster_column, schema = _label_schema(path)
    projection = [id_column, label_column]
    if cluster_column is not None and cluster_column not in projection:
        projection.append(cluster_column)
    table = pq.read_table(path, columns=projection).to_pandas()
    raw_ids = table[id_column]
    normalized_ids = raw_ids.map(lambda value: normalize_sample_id(value) if _valid_identifier(value) else "")
    output = pd.DataFrame(
        {
            "sample_id": normalized_ids,
            "label": pd.to_numeric(table[label_column], errors="coerce"),
            "bootstrap_cluster_id": (
                table[cluster_column].map(lambda value: str(value) if _valid_identifier(value) else "")
                if cluster_column is not None
                else normalized_ids
            ),
        }
    )
    source_kind = f"label_column:{cluster_column}" if cluster_column is not None else "stable_sample_id"
    return output, {
        "label_schema_columns": schema,
        "label_projection_columns": projection,
        "sample_id_column": id_column,
        "bootstrap_cluster_source": source_kind,
    }


def _label_validation(table: pd.DataFrame) -> dict[str, object]:
    binary = table["label"].isin(LABELS) & np.isfinite(pd.to_numeric(table["label"], errors="coerce"))
    sample_valid = table["sample_id"].map(_valid_identifier)
    cluster_valid = table["bootstrap_cluster_id"].map(_valid_identifier)
    duplicate_sample_rows = int(table.duplicated("sample_id", keep=False).sum())
    return {
        "invalid_label_rows": int((~binary).sum()),
        "malformed_sample_id_rows": int((~sample_valid).sum()),
        "malformed_bootstrap_cluster_rows": int((~cluster_valid).sum()),
        "duplicate_normalized_sample_id_rows": duplicate_sample_rows,
        "binary_label_counts": {str(label): int((binary & table["label"].eq(label)).sum()) for label in LABELS},
    }


def _label_failures(validation: Mapping[str, object]) -> list[str]:
    fields = (
        ("invalid_label_rows", "invalid_label"),
        ("malformed_sample_id_rows", "malformed_sample_id"),
        ("malformed_bootstrap_cluster_rows", "malformed_bootstrap_cluster"),
        ("duplicate_normalized_sample_id_rows", "duplicate_normalized_sample_id"),
    )
    return [name for field, name in fields if int(validation.get(field, 0))]


def _selection_rows(dataset: str, labels: pd.DataFrame) -> pd.DataFrame:
    valid = (
        labels["label"].isin(LABELS)
        & labels["sample_id"].map(_valid_identifier)
        & labels["bootstrap_cluster_id"].map(_valid_identifier)
        & ~labels.duplicated("sample_id", keep=False)
    )
    rows: list[dict[str, object]] = []
    for label in LABELS:
        sample_ids = sorted(set(labels.loc[valid & labels["label"].eq(label), "sample_id"].astype(str)))
        ranked = sorted(
            (
                hashlib.sha256(f"{SEED}|{dataset}|{label}|{sample_id}".encode("utf-8")).hexdigest(),
                sample_id,
            )
            for sample_id in sample_ids
        )
        clusters = labels.loc[valid & labels["label"].eq(label), ["sample_id", "bootstrap_cluster_id"]].set_index("sample_id")["bootstrap_cluster_id"]
        for rank, (selection_key, sample_id) in enumerate(ranked[:SAMPLE_CAP_PER_DATASET_LABEL], start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "sample_id": sample_id,
                    "bootstrap_cluster_id": str(clusters.at[sample_id]),
                    "selection_rank": rank,
                    "selection_key_sha256": selection_key,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["dataset", "label", "sample_id", "bootstrap_cluster_id", "selection_rank", "selection_key_sha256"],
    )


def _score_artifact_record(dataset: str, model: str, model_record: Mapping[str, Any]) -> tuple[dict[str, object], list[str]]:
    """Record raw-score byte identity only; score text is not parsed in a freeze."""
    failures: list[str] = []
    try:
        path, pinned = _score_path(model_record, dataset)
    except (ValueError, FileNotFoundError) as error:
        return {
            "dataset": dataset,
            "model": model,
            "available": False,
            "failure": "missing_or_invalid_score_artifact",
            "detail": str(error),
        }, ["missing_or_invalid_score_artifact"]
    observed_size = int(path.stat().st_size)
    observed_sha = sha256_file(path)
    expected_size = pinned.get("size_bytes")
    expected_sha = pinned.get("sha256")
    if not isinstance(model_record.get("revision"), str) or not model_record.get("revision"):
        failures.append("missing_model_revision")
    if expected_size is not None and int(expected_size) != observed_size:
        failures.append("score_size_mismatch")
    if expected_sha is not None and str(expected_sha) != observed_sha:
        failures.append("score_hash_mismatch")
    return {
        "dataset": dataset,
        "model": model,
        "available": True,
        "absolute_path": str(path),
        "relative_path": str(pinned.get("path")),
        "model_repo_id": model_record.get("repo_id"),
        "model_revision": model_record.get("revision"),
        "pinned_byte_size": expected_size,
        "pinned_sha256": expected_sha,
        "byte_size": observed_size,
        "sha256": observed_sha,
        "validation_failures": failures,
    }, failures


def _label_input_record(dataset: str, dataset_record: Mapping[str, Any]) -> tuple[dict[str, object], pd.DataFrame]:
    path = _label_path(dataset_record)
    labels, schema = _read_label_table(path)
    validation = _label_validation(labels)
    selection = _selection_rows(dataset, labels)
    failures = _label_failures(validation)
    if any(int(selection["label"].eq(label).sum()) < 1 for label in LABELS):
        failures.append("missing_selected_label")
    relative_path = str(dataset_record["files"]["labels"])
    pinned = _pinned_file_record(dataset_record, relative_path)
    observed_size = int(path.stat().st_size)
    observed_sha = sha256_file(path)
    expected_size = pinned.get("size_bytes") if pinned is not None else None
    expected_sha = pinned.get("sha256") if pinned is not None else None
    if not isinstance(dataset_record.get("revision"), str) or not dataset_record.get("revision"):
        failures.append("missing_dataset_revision")
    if pinned is None:
        failures.append("missing_pinned_label_record")
    elif expected_size is not None and int(expected_size) != observed_size:
        failures.append("label_size_mismatch")
    if expected_sha is not None and str(expected_sha) != observed_sha:
        failures.append("label_hash_mismatch")
    record = {
        "dataset": dataset,
        "dataset_repo_id": dataset_record.get("repo_id"),
        "dataset_revision": dataset_record.get("revision"),
        "absolute_path": str(path),
        "relative_path": relative_path,
        "pinned_byte_size": expected_size,
        "pinned_sha256": expected_sha,
        "byte_size": observed_size,
        "sha256": observed_sha,
        **schema,
        "n_label_rows": int(len(labels)),
        "n_selected_by_label": {str(label): int(selection["label"].eq(label).sum()) for label in LABELS},
        "validation": validation,
        "validation_failures": sorted(set(failures)),
    }
    return record, selection


def _require_new_output_dir(output_dir: Path | str) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"H6 artifacts are non-overwritable; output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def freeze_inputs(
    output_dir: Path | str,
    *,
    index_path: Path | str = ARENA_INDEX_PATH,
    allowed_index_path: Path | str = ARENA_INDEX_PATH,
    frozen_at_utc: str | None = None,
) -> FreezeArtifacts:
    """Seal a strictly label-selected H6 panel before raw score values are read.

    The freeze hashes each original score file so later input substitution is
    impossible, but it never parses, ranks, joins, or materialises any score
    value.  ``allowed_index_path`` is injectable only for isolated synthetic
    tests; the public CLI never accepts an alternate index.
    """
    locked_index_path = _require_locked_index(index_path, allowed_index_path)
    index = _load_index(locked_index_path)
    output = _require_new_output_dir(output_dir)
    frozen_at = frozen_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    labels: list[dict[str, object]] = []
    selections: list[pd.DataFrame] = []
    score_artifacts: list[dict[str, object]] = []
    failed_datasets: set[str] = set()
    for dataset in DATASETS:
        label_record, selection = _label_input_record(dataset, index["datasets"][dataset])
        labels.append(label_record)
        selections.append(selection)
        if label_record["validation_failures"] or not len(selection):
            failed_datasets.add(dataset)
        for model in MODELS:
            score_record, failures = _score_artifact_record(dataset, model, index["models"][model])
            score_artifacts.append(score_record)
            if failures:
                failed_datasets.add(dataset)
    selection_table = pd.concat(selections, ignore_index=True)
    if selection_table.duplicated(["dataset", "sample_id"], keep=False).any():
        raise RuntimeError("H6 label freeze generated duplicate selected sample IDs")
    if selection_table.duplicated(["dataset", "label", "selection_rank"], keep=False).any():
        raise RuntimeError("H6 label freeze generated duplicate selection ranks")
    if (selection_table.groupby(["dataset", "label"], sort=True).size() > SAMPLE_CAP_PER_DATASET_LABEL).any():
        raise RuntimeError("H6 label freeze exceeded the locked per-class cap")

    output.mkdir()
    manifest_path = output / "h6_label_selection_manifest.parquet"
    selection_table.to_parquet(manifest_path, index=False)
    manifest_sha = sha256_file(manifest_path)
    provenance = {
        "h6_version": H6_VERSION,
        "protocol_path": "experiments/h6_score_agreement/protocol.md",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "frozen_at_utc": frozen_at,
        "analysis_scope": "within_class_published_raw_score_agreement_only",
        "raw_score_values_read_during_freeze": False,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "labels": list(LABELS),
        "model_pairs_lexicographic": [list(pair) for pair in MODEL_PAIRS],
        "sample_cap_per_dataset_label": SAMPLE_CAP_PER_DATASET_LABEL,
        "selection_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": CONFIDENCE,
        "sample_id_normalization": "Path(value).stem",
        "arena_index": {
            "absolute_path": str(locked_index_path),
            "sha256": sha256_file(locked_index_path),
            "byte_size": int(locked_index_path.stat().st_size),
        },
        "freeze_status": "complete" if not failed_datasets else "failed_input_integrity",
        "failed_datasets": sorted(failed_datasets),
        "label_inputs": labels,
        "score_artifacts": score_artifacts,
        "selection_manifest": {
            "absolute_path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "byte_size": int(manifest_path.stat().st_size),
            "n_rows": int(len(selection_table)),
            "columns": selection_table.columns.tolist(),
        },
    }
    provenance_path = output / "h6_input_freeze_provenance.json"
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return FreezeArtifacts(output, manifest_path, provenance_path, manifest_sha)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read H6 provenance JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("H6 provenance must be a JSON object")
    return value


def _index_records(index: Mapping[str, Any], dataset: str, model: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    datasets = index.get("datasets")
    models = index.get("models")
    if not isinstance(datasets, Mapping) or not isinstance(models, Mapping):
        raise ValueError("H6 index mappings are unavailable")
    dataset_record = datasets.get(dataset)
    model_record = models.get(model)
    if not isinstance(dataset_record, Mapping) or not isinstance(model_record, Mapping):
        raise ValueError(f"H6 index scope mismatch for {dataset}/{model}")
    return dataset_record, model_record


def _record_maps(provenance: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    label_records = provenance.get("label_inputs")
    score_records = provenance.get("score_artifacts")
    if not isinstance(label_records, list) or len(label_records) != len(DATASETS):
        raise ValueError("H6 provenance lacks exactly five label input records")
    if not isinstance(score_records, list) or len(score_records) != len(DATASETS) * len(MODELS):
        raise ValueError("H6 provenance lacks exactly forty score artifact records")
    labels: dict[str, Mapping[str, Any]] = {}
    scores: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in label_records:
        if not isinstance(record, Mapping) or record.get("dataset") not in DATASETS or record["dataset"] in labels:
            raise ValueError("H6 label input records have invalid scope or duplicates")
        labels[str(record["dataset"])] = record
    for record in score_records:
        key = (str(record.get("dataset")), str(record.get("model")))
        if not isinstance(record, Mapping) or key[0] not in DATASETS or key[1] not in MODELS or key in scores:
            raise ValueError("H6 score artifact records have invalid scope or duplicates")
        scores[key] = record
    if set(labels) != set(DATASETS) or set(scores) != {(d, m) for d in DATASETS for m in MODELS}:
        raise ValueError("H6 provenance artifact scope is not the exact locked matrix")
    return labels, scores


def _validate_manifest(provenance_file: Path, provenance: Mapping[str, Any], expected_selection: list[pd.DataFrame]) -> pd.DataFrame:
    metadata = provenance.get("selection_manifest")
    if not isinstance(metadata, Mapping):
        raise ValueError("H6 provenance lacks selection manifest metadata")
    manifest_path = Path(str(metadata.get("absolute_path", "")))
    if not manifest_path.is_absolute() or manifest_path.parent != provenance_file.parent.resolve():
        raise ValueError("H6 selection manifest path is outside its freeze directory")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"H6 selection manifest missing: {manifest_path}")
    if int(metadata.get("byte_size", -1)) != manifest_path.stat().st_size or metadata.get("sha256") != sha256_file(manifest_path):
        raise ValueError("H6 selection manifest changed after freeze")
    manifest = pd.read_parquet(manifest_path)
    columns = ["dataset", "label", "sample_id", "bootstrap_cluster_id", "selection_rank", "selection_key_sha256"]
    if manifest.columns.tolist() != columns or len(manifest) != int(metadata.get("n_rows", -1)):
        raise ValueError("H6 selection manifest schema or row count mismatch")
    if manifest.duplicated(["dataset", "sample_id"], keep=False).any():
        raise ValueError("H6 selection manifest contains duplicate sample IDs")
    if manifest.duplicated(["dataset", "label", "selection_rank"], keep=False).any():
        raise ValueError("H6 selection manifest contains duplicate selection ranks")
    if not manifest["sample_id"].map(_valid_identifier).all() or not manifest["bootstrap_cluster_id"].map(_valid_identifier).all():
        raise ValueError("H6 selection manifest has malformed identity keys")
    if not manifest["label"].isin(LABELS).all() or set(manifest["dataset"].astype(str)) != set(DATASETS):
        raise ValueError("H6 selection manifest has an invalid label/dataset scope")
    if (manifest.groupby(["dataset", "label"], sort=True).size() > SAMPLE_CAP_PER_DATASET_LABEL).any():
        raise ValueError("H6 selection manifest exceeds the locked cap")
    expected = pd.concat(expected_selection, ignore_index=True).sort_values(columns).reset_index(drop=True)
    observed = manifest.sort_values(columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("H6 selection manifest no longer matches the independent label-only selection") from error
    return manifest


def validate_frozen_provenance(
    provenance_path: Path | str,
    *,
    allowed_index_path: Path | str = ARENA_INDEX_PATH,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Revalidate labels and byte identities before an analyzer opens score text."""
    provenance_file = Path(provenance_path)
    provenance = _load_json(provenance_file)
    if provenance.get("h6_version") != H6_VERSION:
        raise ValueError("Unsupported or missing H6 provenance version")
    if provenance.get("datasets") != list(DATASETS) or provenance.get("models") != list(MODELS) or provenance.get("labels") != list(LABELS):
        raise ValueError("H6 provenance scope/order does not match the locked protocol")
    if provenance.get("model_pairs_lexicographic") != [list(pair) for pair in MODEL_PAIRS]:
        raise ValueError("H6 provenance model-pair registry drifted")
    if provenance.get("sample_cap_per_dataset_label") != SAMPLE_CAP_PER_DATASET_LABEL or provenance.get("selection_seed") != SEED:
        raise ValueError("H6 provenance cap or selection seed drifted")
    if provenance.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES or provenance.get("bootstrap_confidence") != CONFIDENCE:
        raise ValueError("H6 provenance bootstrap configuration drifted")
    if provenance.get("raw_score_values_read_during_freeze") is not False:
        raise ValueError("H6 provenance does not establish a label-only freeze")
    if provenance.get("sample_id_normalization") != "Path(value).stem":
        raise ValueError("H6 provenance sample-ID normalization drifted")
    if sha256_file(PROTOCOL_PATH) != provenance.get("protocol_sha256"):
        raise ValueError("H6 protocol changed after freeze; create a new label-only freeze")

    index_meta = provenance.get("arena_index")
    if not isinstance(index_meta, Mapping):
        raise ValueError("H6 provenance lacks Arena index metadata")
    supplied_index = _require_locked_index(Path(str(index_meta.get("absolute_path", ""))), allowed_index_path)
    if int(index_meta.get("byte_size", -1)) != supplied_index.stat().st_size or index_meta.get("sha256") != sha256_file(supplied_index):
        raise ValueError("H6 Arena index changed after freeze")
    index = _load_index(supplied_index)
    label_records, score_records = _record_maps(provenance)
    expected_selection: list[pd.DataFrame] = []
    for dataset in DATASETS:
        dataset_record, _ = _index_records(index, dataset, MODELS[0])
        current_record, current_selection = _label_input_record(dataset, dataset_record)
        if dict(current_record) != dict(label_records[dataset]):
            raise ValueError(f"H6 label input changed after freeze for {dataset}")
        expected_selection.append(current_selection)
    manifest = _validate_manifest(provenance_file, provenance, expected_selection)

    for dataset in DATASETS:
        for model in MODELS:
            _, model_record = _index_records(index, dataset, model)
            observed, _ = _score_artifact_record(dataset, model, model_record)
            frozen = score_records[(dataset, model)]
            # JSON equality intentionally includes byte identity/revision/path
            # but no score values. A failure record must also be immutable.
            if dict(observed) != dict(frozen):
                raise ValueError(f"H6 raw score artifact changed after freeze for {dataset}/{model}")
    expected_failed_datasets = sorted(
        {
            dataset
            for dataset in DATASETS
            if label_records[dataset].get("validation_failures")
            or any(score_records[(dataset, model)].get("validation_failures") for model in MODELS)
        }
    )
    expected_status = "complete" if not expected_failed_datasets else "failed_input_integrity"
    if provenance.get("failed_datasets") != expected_failed_datasets or provenance.get("freeze_status") != expected_status:
        raise ValueError("H6 freeze status does not match revalidated input records")
    return provenance, manifest, index


def _read_raw_scores(path: Path) -> RawScoreRead:
    """Parse only the original two-column raw score text, with no repair rule."""
    rows: list[tuple[str, float]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                fields = line.strip().split()
                if not fields:
                    continue
                if len(fields) != 2:
                    return RawScoreRead(pd.DataFrame(columns=["sample_id", "raw_score"]), f"malformed_raw_score_line_{number}")
                sample_id = normalize_sample_id(fields[0])
                if not _valid_identifier(sample_id):
                    return RawScoreRead(pd.DataFrame(columns=["sample_id", "raw_score"]), f"malformed_raw_score_id_{number}")
                try:
                    score = float(fields[1])
                except ValueError:
                    return RawScoreRead(pd.DataFrame(columns=["sample_id", "raw_score"]), f"non_numeric_raw_score_{number}")
                if not np.isfinite(score):
                    return RawScoreRead(pd.DataFrame(columns=["sample_id", "raw_score"]), f"nonfinite_raw_score_{number}")
                rows.append((sample_id, score))
    except UnicodeDecodeError:
        return RawScoreRead(pd.DataFrame(columns=["sample_id", "raw_score"]), "raw_score_not_utf8_text")
    table = pd.DataFrame(rows, columns=["sample_id", "raw_score"])
    if table.empty:
        return RawScoreRead(table, "empty_raw_score_artifact")
    if table.duplicated("sample_id", keep=False).any():
        return RawScoreRead(table, "duplicate_model_sample_id")
    return RawScoreRead(table, None)


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    ranks_a = rankdata(values_a, method="average")
    ranks_b = rankdata(values_b, method="average")
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.sqrt(np.dot(centered_a, centered_a) * np.dot(centered_b, centered_b)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centered_a, centered_b) / denominator)


def _orientation_and_coverage(raw: RawScoreRead, selected: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    """Join to the immutable panel and apply the registered spoof orientation."""
    if raw.failure is not None:
        return pd.DataFrame(columns=["sample_id", "label", "bootstrap_cluster_id", "score_spoof"]), raw.failure, None
    joined = selected.merge(raw.table, on="sample_id", how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        return pd.DataFrame(columns=["sample_id", "label", "bootstrap_cluster_id", "score_spoof"]), "missing_selected_raw_score", None
    values = joined["raw_score"].to_numpy(dtype=float)
    labels = joined["label"].to_numpy(dtype=float)
    if len(np.unique(labels)) != len(LABELS):
        return pd.DataFrame(columns=["sample_id", "label", "bootstrap_cluster_id", "score_spoof"]), "orientation_requires_both_labels", None
    orientation_correlation = _spearman(values, labels)
    if not np.isfinite(orientation_correlation) or abs(orientation_correlation) < 0.02:
        return pd.DataFrame(columns=["sample_id", "label", "bootstrap_cluster_id", "score_spoof"]), "ambiguous_spoof_orientation", None
    orientation = "raw_is_spoof" if orientation_correlation > 0.0 else "negated_raw_is_spoof"
    joined["score_spoof"] = joined["raw_score"] if orientation == "raw_is_spoof" else -joined["raw_score"]
    return joined[["sample_id", "label", "bootstrap_cluster_id", "score_spoof"]], None, orientation


def _cell_seed(dataset: str, label: int, model_a: str, model_b: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{SEED}|{dataset}|{label}|{model_a}|{model_b}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), digest


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = (1.0 - CONFIDENCE) / 2.0
    return tuple(float(value) for value in np.quantile(finite, [alpha, 1.0 - alpha]))


def _cluster_bootstrap(values_a: np.ndarray, values_b: np.ndarray, clusters: np.ndarray, *, seed: int) -> dict[str, object]:
    unique_clusters = np.array(sorted(set(clusters.astype(str))))
    positions = [np.flatnonzero(clusters.astype(str) == cluster) for cluster in unique_clusters]
    correlations = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_clusters = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        sampled_positions = np.concatenate([positions[index] for index in selected_clusters])
        correlations[replicate] = _spearman(values_a[sampled_positions], values_b[sampled_positions])
    low, high = _percentile_interval(correlations)
    return {
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid": int(np.isfinite(correlations).sum()),
        "bootstrap_replicates_invalid": int((~np.isfinite(correlations)).sum()),
        "bootstrap_confidence": CONFIDENCE,
        "spearman_ci_low": low,
        "spearman_ci_high": high,
        "bootstrap_cluster_count": int(len(unique_clusters)),
    }


def _failed_cell(dataset: str, label: int, model_a: str, model_b: str, status: str, *, selected_count: int = 0, a_count: int = 0, b_count: int = 0) -> dict[str, object]:
    seed, seed_hash = _cell_seed(dataset, label, model_a, model_b)
    return {
        "dataset": dataset,
        "label": label,
        "model_a": model_a,
        "model_b": model_b,
        "cell_status": status,
        "selected_sample_count": selected_count,
        "model_a_available_score_count": a_count,
        "model_b_available_score_count": b_count,
        "exact_joined_sample_count": 0,
        "spearman_agreement": float("nan"),
        "bootstrap_cluster_count": 0,
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid": 0,
        "bootstrap_replicates_invalid": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": CONFIDENCE,
        "spearman_ci_low": float("nan"),
        "spearman_ci_high": float("nan"),
        "bootstrap_seed": seed,
        "bootstrap_seed_sha256": seed_hash,
    }


def _cell_row(dataset: str, label: int, model_a: str, model_b: str, selected: pd.DataFrame, score_tables: Mapping[tuple[str, str], tuple[pd.DataFrame, str | None, str | None]]) -> dict[str, object]:
    panel = selected.loc[(selected["dataset"].eq(dataset)) & (selected["label"].eq(label))].copy()
    expected_count = len(panel)
    scores_a, failure_a, _ = score_tables[(dataset, model_a)]
    scores_b, failure_b, _ = score_tables[(dataset, model_b)]
    a_class = scores_a.loc[scores_a["label"].eq(label)] if not scores_a.empty else scores_a
    b_class = scores_b.loc[scores_b["label"].eq(label)] if not scores_b.empty else scores_b
    a_count, b_count = len(a_class), len(b_class)
    if expected_count < 2:
        return _failed_cell(dataset, label, model_a, model_b, "failed_insufficient_label_selected_samples", selected_count=expected_count, a_count=a_count, b_count=b_count)
    if failure_a is not None or failure_b is not None:
        reasons = sorted(set(reason for reason in (failure_a, failure_b) if reason is not None))
        return _failed_cell(dataset, label, model_a, model_b, "failed_score_validation_" + "_and_".join(reasons), selected_count=expected_count, a_count=a_count, b_count=b_count)
    if a_count != expected_count or b_count != expected_count:
        return _failed_cell(dataset, label, model_a, model_b, "failed_missing_score_without_intersection_reselection", selected_count=expected_count, a_count=a_count, b_count=b_count)
    wide = panel[["sample_id", "bootstrap_cluster_id"]].merge(
        a_class[["sample_id", "score_spoof"]].rename(columns={"score_spoof": "score_a"}), on="sample_id", how="left", validate="one_to_one"
    ).merge(
        b_class[["sample_id", "score_spoof"]].rename(columns={"score_spoof": "score_b"}), on="sample_id", how="left", validate="one_to_one")
    if len(wide) != expected_count or wide[["score_a", "score_b"]].isna().any().any():
        return _failed_cell(dataset, label, model_a, model_b, "failed_nonexact_sample_id_join", selected_count=expected_count, a_count=a_count, b_count=b_count)
    values_a = wide["score_a"].to_numpy(dtype=float)
    values_b = wide["score_b"].to_numpy(dtype=float)
    agreement = _spearman(values_a, values_b)
    if not np.isfinite(agreement):
        return _failed_cell(dataset, label, model_a, model_b, "failed_nonfinite_within_class_spearman", selected_count=expected_count, a_count=a_count, b_count=b_count)
    seed, seed_hash = _cell_seed(dataset, label, model_a, model_b)
    bootstrap = _cluster_bootstrap(values_a, values_b, wide["bootstrap_cluster_id"].to_numpy(), seed=seed)
    return {
        "dataset": dataset,
        "label": label,
        "model_a": model_a,
        "model_b": model_b,
        "cell_status": "ok",
        "selected_sample_count": expected_count,
        "model_a_available_score_count": a_count,
        "model_b_available_score_count": b_count,
        "exact_joined_sample_count": expected_count,
        "spearman_agreement": agreement,
        **bootstrap,
        "bootstrap_seed": seed,
        "bootstrap_seed_sha256": seed_hash,
    }


def aggregate_cells(matrix: pd.DataFrame) -> pd.DataFrame:
    """Materialise all 28 registered pair summaries over their ten fixed cells."""
    expected = len(DATASETS) * len(LABELS) * len(MODEL_PAIRS)
    key = ["dataset", "label", "model_a", "model_b"]
    if len(matrix) != expected or matrix.duplicated(key, keep=False).any():
        raise ValueError("H6 aggregation requires exactly one exhaustive 280-cell matrix")
    rows: list[dict[str, object]] = []
    for model_a, model_b in MODEL_PAIRS:
        unit = matrix.loc[(matrix["model_a"].eq(model_a)) & (matrix["model_b"].eq(model_b))].copy()
        observed = {(str(row.dataset), int(row.label)) for row in unit.itertuples()}
        expected_cells = {(dataset, label) for dataset in DATASETS for label in LABELS}
        ok = unit.loc[unit["cell_status"].eq("ok")]
        values = pd.to_numeric(ok["spearman_agreement"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "expected_dataset_class_cells": len(expected_cells),
                "observed_dataset_class_cells": len(observed),
                "ok_cell_count": int(len(ok)),
                "failed_cell_count": int(len(unit) - len(ok)),
                "all_ten_cells_ok": bool(observed == expected_cells and len(ok) == len(expected_cells) and len(finite) == len(expected_cells)),
                "median_spearman_agreement": float(np.median(finite)) if len(finite) else float("nan"),
                "minimum_spearman_agreement": float(np.min(finite)) if len(finite) else float("nan"),
                "maximum_spearman_agreement": float(np.max(finite)) if len(finite) else float("nan"),
                "cross_cell_spearman_range": float(np.max(finite) - np.min(finite)) if len(finite) else float("nan"),
                "cell_statuses": ";".join(sorted(set(unit["cell_status"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def _all_failed_matrix(status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [_failed_cell(dataset, label, model_a, model_b, status) for dataset in DATASETS for label in LABELS for model_a, model_b in MODEL_PAIRS]
    )


def analyze_frozen_manifest(
    provenance_path: Path | str,
    output_dir: Path | str,
    *,
    allowed_index_path: Path | str = ARENA_INDEX_PATH,
) -> tuple[Path, Path, Path]:
    """Write the exhaustive H6 raw-score agreement atlas from a validated freeze.

    The analyzer deliberately has no score-path, waveform-path, feature-path,
    model-path, or result-summary option.  The frozen provenance is its only
    input locator.  It writes an explicit 280-row failure matrix rather than
    silently intersecting or replacing unavailable score inputs.
    """
    provenance, selected, _ = validate_frozen_provenance(provenance_path, allowed_index_path=allowed_index_path)
    output = _require_new_output_dir(output_dir)
    output.mkdir()
    freeze_complete = provenance.get("freeze_status") == "complete"
    orientations: list[dict[str, object]] = []
    if not freeze_complete:
        matrix = _all_failed_matrix("failed_label_or_score_input_freeze")
        stop_reason = "input_freeze_not_complete"
    else:
        score_records = {(str(record["dataset"]), str(record["model"])): record for record in provenance["score_artifacts"]}
        score_tables: dict[tuple[str, str], tuple[pd.DataFrame, str | None, str | None]] = {}
        for dataset in DATASETS:
            dataset_selected = selected.loc[selected["dataset"].eq(dataset), ["sample_id", "label", "bootstrap_cluster_id"]].copy()
            for model in MODELS:
                record = score_records[(dataset, model)]
                path = Path(str(record.get("absolute_path", "")))
                raw = _read_raw_scores(path)
                oriented, failure, orientation = _orientation_and_coverage(raw, dataset_selected)
                score_tables[(dataset, model)] = (oriented, failure, orientation)
                orientations.append({"dataset": dataset, "model": model, "orientation": orientation, "validation_status": "ok" if failure is None else failure})
        matrix = pd.DataFrame(
            [_cell_row(dataset, label, model_a, model_b, selected, score_tables) for dataset in DATASETS for label in LABELS for model_a, model_b in MODEL_PAIRS]
        )
        stop_reason = "complete" if matrix["cell_status"].eq("ok").all() else "raw_score_validation_or_unavailable_cell"
    expected = len(DATASETS) * len(LABELS) * len(MODEL_PAIRS)
    if len(matrix) != expected or matrix.duplicated(["dataset", "label", "model_a", "model_b"], keep=False).any():
        raise RuntimeError("H6 failed to materialise its exact 280-cell registry")
    aggregation = aggregate_cells(matrix)
    matrix_path = output / "h6_within_class_score_agreement_matrix.csv"
    aggregation_path = output / "h6_model_pair_agreement_summary.csv"
    matrix.to_csv(matrix_path, index=False)
    aggregation.to_csv(aggregation_path, index=False)
    report = {
        "h6_version": H6_VERSION,
        "protocol_sha256": provenance["protocol_sha256"],
        "input_freeze_provenance_absolute_path": str(Path(provenance_path).resolve()),
        "selection_manifest_sha256": provenance["selection_manifest"]["sha256"],
        "raw_score_only": True,
        "score_values_emitted": False,
        "analysis_status": "complete" if stop_reason == "complete" else "stopped_with_explicit_unavailable_cells",
        "stop_reason": stop_reason,
        "n_expected_dataset_class_model_pair_cells": expected,
        "n_written_dataset_class_model_pair_cells": int(len(matrix)),
        "complete_280_cell_matrix": bool(len(matrix) == expected),
        "ok_cell_count": int(matrix["cell_status"].eq("ok").sum()),
        "failed_or_unavailable_cell_count": int((~matrix["cell_status"].eq("ok")).sum()),
        "n_model_pair_summaries": int(len(aggregation)),
        "orientation_records": orientations,
        "outputs": {
            "matrix": {"absolute_path": str(matrix_path.resolve()), "sha256": sha256_file(matrix_path), "byte_size": int(matrix_path.stat().st_size)},
            "aggregation": {"absolute_path": str(aggregation_path.resolve()), "sha256": sha256_file(aggregation_path), "byte_size": int(aggregation_path.stat().st_size)},
        },
    }
    report_path = output / "h6_analysis_provenance.json"
    report_path.write_text(canonical_json(report), encoding="utf-8")
    return matrix_path, aggregation_path, report_path
