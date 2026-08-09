"""Score-free input freezing and descriptive analysis for the locked H4 atlas.

H4 intentionally consumes only fixed feature Parquet products.  This module
does not decode audio, import model code, load score artifacts, or train a
classifier.  The only statistic is a rank-based binary-label AUROC calculated
directly from one registered waveform feature at a time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.audio_features import FEATURE_NAMES, FEATURE_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiments/h4_label_transportability/protocol.md"
H4_VERSION = "h4_label_cue_transportability_v1"
SEED = 2609
SOURCE_CAP_PER_DATASET_LABEL = 5_000
BOOTSTRAP_REPLICATES = 500
CONFIDENCE = 0.95

DATASETS = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
HELD_OUT_DATASETS = ("InTheWild", "ASVspoof5")
VIEWS = ("full_waveform", "deterministic_crop", "preemphasized_crop")
BASE_COLUMNS = ("sample_id", "source_id", "label", "view")
ALLOWED_COLUMNS = (*BASE_COLUMNS, *FEATURE_NAMES)
FORBIDDEN_TERMS = ("score", "logit", "detector", "model", "eer")
FORBIDDEN_PATH_TERMS = (*FORBIDDEN_TERMS, "arena")

H4_ALLOWED_INPUT_PATHS: dict[str, Path] = {
    dataset: Path(
        "/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/"
        f"{FEATURE_VERSION}/{dataset}/features_wide.parquet"
    )
    for dataset in DATASETS
}


@dataclass(frozen=True)
class FreezeArtifacts:
    """Paths and hashes emitted by a completed non-overwritable H4 freeze."""

    output_dir: Path
    manifest_path: Path
    provenance_path: Path
    manifest_sha256: str


def canonical_json(value: object) -> str:
    """Return a stable JSON representation suitable for integrity records."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_forbidden_term(value: str, terms: tuple[str, ...] = FORBIDDEN_TERMS) -> bool:
    lower = value.casefold()
    return any(term in lower for term in terms)


def _path_is_safe(path: Path) -> bool:
    """Reject score/model/Arena-looking paths before any file access."""
    return not _has_forbidden_term(str(path), FORBIDDEN_PATH_TERMS)


def _as_absolute_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"H4 input path must be absolute: {candidate}")
    if not _path_is_safe(candidate):
        raise ValueError(f"H4 path firewall rejected forbidden path: {candidate}")
    return candidate


def validate_explicit_input_paths(
    input_paths: Mapping[str, Path | str],
    *,
    allowed_paths: Mapping[str, Path | str] = H4_ALLOWED_INPUT_PATHS,
) -> dict[str, Path]:
    """Require exactly the five named, protocol-allowed local inputs.

    ``allowed_paths`` is injectable only for synthetic tests.  The public CLI
    always uses ``H4_ALLOWED_INPUT_PATHS`` and thus cannot be redirected to an
    Arena, score, model, or arbitrary feature artifact.
    """
    expected = tuple(allowed_paths)
    if set(input_paths) != set(expected):
        missing = sorted(set(expected) - set(input_paths))
        extra = sorted(set(input_paths) - set(expected))
        raise ValueError(f"H4 requires exactly the five locked datasets; missing={missing}, extra={extra}")
    resolved: dict[str, Path] = {}
    for dataset in expected:
        supplied = _as_absolute_path(input_paths[dataset])
        allowed = _as_absolute_path(allowed_paths[dataset])
        # Require the literal protocol path as well as its resolved target.
        # The first check prohibits path aliases; the second rejects a symlink
        # that unexpectedly escapes the declared input location.
        if supplied != allowed or supplied.resolve() != allowed.resolve():
            raise ValueError(
                f"H4 path firewall rejected {dataset}: supplied={supplied}, locked={allowed}"
            )
        if not supplied.is_file():
            raise FileNotFoundError(f"H4 locked input is not a regular file: {supplied}")
        resolved[dataset] = supplied
    return resolved


def _schema_columns(path: Path) -> list[str]:
    """Inspect Parquet schema names without reading feature values."""
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    if len(columns) != len(set(columns)):
        raise ValueError(f"H4 input has duplicate Parquet column names: {path}")
    missing = sorted(set(ALLOWED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"H4 input missing required allowed columns at {path}: {missing}")
    forbidden = sorted(column for column in columns if _has_forbidden_term(column))
    if forbidden:
        raise ValueError(f"H4 allowed-column firewall rejected response-like columns at {path}: {forbidden}")
    return columns


def _read_allowed_table(path: Path) -> pd.DataFrame:
    """Read exactly H4's registered feature/label fields, never metadata extras."""
    _schema_columns(path)
    table = pd.read_parquet(path, columns=list(ALLOWED_COLUMNS))
    if tuple(table.columns) != ALLOWED_COLUMNS:
        # Pandas/Arrow is expected to honour the projection order.  Enforce it
        # so a dependency change cannot silently alter the frozen schema.
        table = table.loc[:, list(ALLOWED_COLUMNS)].copy()
    return table


def _string_valid(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _binary_label_mask(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.isin([0, 1]) & np.isfinite(numeric)


def _normalised_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.isin([0, 1]), other=np.nan).astype("Float64")


def _count_by_view_and_label(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    labels = _normalised_labels(table["label"])
    output: dict[str, dict[str, int]] = {}
    for view in VIEWS:
        mask = table["view"].eq(view)
        output[view] = {
            "label_0": int((mask & labels.eq(0)).sum()),
            "label_1": int((mask & labels.eq(1)).sum()),
            "all_rows": int(mask.sum()),
        }
    return output


def _validation_record(table: pd.DataFrame) -> dict[str, object]:
    """Record all integrity conditions that make H4 cells explicitly fail."""
    label_valid = _binary_label_mask(table["label"])
    source_valid = table["source_id"].map(_string_valid)
    sample_valid = table["sample_id"].map(_string_valid)
    view_valid = table["view"].map(_string_valid)
    duplicate_key_rows = int(table.duplicated(["sample_id", "view"], keep=False).sum())
    observed_views = sorted(
        value for value in table.loc[view_valid, "view"].astype(str).unique().tolist() if value in VIEWS
    )
    unexpected_views = sorted(
        value for value in table.loc[view_valid, "view"].astype(str).unique().tolist() if value not in VIEWS
    )

    valid_key_rows = table.loc[label_valid & source_valid & sample_valid & view_valid & table["view"].isin(VIEWS)].copy()
    valid_key_rows["_label"] = _normalised_labels(valid_key_rows["label"]).astype(int)
    incomplete_source_sample_pairs = 0
    source_label_conflicts = 0
    if not valid_key_rows.empty:
        per_pair_views = valid_key_rows.groupby(["_label", "source_id", "sample_id"], sort=True)["view"].agg(
            lambda values: set(values)
        )
        incomplete_source_sample_pairs = int(sum(set(VIEWS) != set(values) for values in per_pair_views))
        labels_per_source = valid_key_rows.groupby("source_id", sort=True)["_label"].nunique()
        source_label_conflicts = int((labels_per_source > 1).sum())
    return {
        "malformed_source_id_rows": int((~source_valid).sum()),
        "malformed_sample_id_rows": int((~sample_valid).sum()),
        "malformed_view_rows": int((~view_valid).sum()),
        "invalid_label_rows": int((~label_valid).sum()),
        "duplicate_sample_view_rows": duplicate_key_rows,
        "missing_required_views": [view for view in VIEWS if view not in observed_views],
        "unexpected_views": unexpected_views,
        "incomplete_source_sample_pairs": incomplete_source_sample_pairs,
        "source_label_conflicts": source_label_conflicts,
    }


def _integrity_failures(validation: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    count_fields = (
        ("malformed_source_id_rows", "malformed_source_id"),
        ("malformed_sample_id_rows", "malformed_sample_id"),
        ("malformed_view_rows", "malformed_view"),
        ("invalid_label_rows", "invalid_label"),
        ("duplicate_sample_view_rows", "duplicate_key"),
        ("incomplete_source_sample_pairs", "missing_view"),
        ("source_label_conflicts", "source_label_conflict"),
    )
    for field, name in count_fields:
        if int(validation.get(field, 0)):
            failures.append(name)
    if validation.get("missing_required_views"):
        failures.append("missing_required_view")
    if validation.get("unexpected_views"):
        failures.append("unexpected_view")
    return failures


def _selection_rows(dataset: str, table: pd.DataFrame, *, cap: int, seed: int) -> pd.DataFrame:
    """Choose source IDs deterministically, retaining no sample-level response data."""
    labels = _normalised_labels(table["label"])
    valid = (
        labels.notna()
        & table["source_id"].map(_string_valid)
        & table["sample_id"].map(_string_valid)
        & table["view"].map(_string_valid)
        & table["view"].isin(VIEWS)
    )
    rows: list[dict[str, object]] = []
    for label in (0, 1):
        sources = sorted(set(table.loc[valid & labels.eq(label), "source_id"].astype(str).tolist()))
        ranked = sorted(
            (
                hashlib.sha256(f"{seed}|{dataset}|{label}|{source_id}".encode("utf-8")).hexdigest(),
                source_id,
            )
            for source_id in sources
        )
        for rank, (selection_key, source_id) in enumerate(ranked[:cap], start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "source_id": source_id,
                    "selection_rank": rank,
                    "selection_key_sha256": selection_key,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["dataset", "label", "source_id", "selection_rank", "selection_key_sha256"],
    )


def _input_record(dataset: str, path: Path, table: pd.DataFrame, *, cap: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    schema_columns = _schema_columns(path)
    validation = _validation_record(table)
    selection = _selection_rows(dataset, table, cap=cap, seed=seed)
    labels = _normalised_labels(table["label"])
    record: dict[str, object] = {
        "dataset": dataset,
        "absolute_path": str(path),
        "byte_size": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "source_parquet_schema_columns": schema_columns,
        "allowed_column_schema": list(ALLOWED_COLUMNS),
        "projected_columns_only": True,
        "feature_version": FEATURE_VERSION,
        "n_rows": int(len(table)),
        "binary_label_counts": {
            "label_0": int(labels.eq(0).sum()),
            "label_1": int(labels.eq(1).sum()),
        },
        "per_view_row_counts": _count_by_view_and_label(table),
        "n_distinct_valid_sources_by_label": {
            "label_0": int(table.loc[labels.eq(0) & table["source_id"].map(_string_valid), "source_id"].nunique()),
            "label_1": int(table.loc[labels.eq(1) & table["source_id"].map(_string_valid), "source_id"].nunique()),
        },
        "n_selected_sources_by_label": {
            "label_0": int((selection["label"] == 0).sum()),
            "label_1": int((selection["label"] == 1).sum()),
        },
        "validation": validation,
        "validation_failures": _integrity_failures(validation),
    }
    return record, selection


def _require_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"H4 artifacts are non-overwritable; output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def freeze_inputs(
    input_paths: Mapping[str, Path | str],
    output_dir: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H4_ALLOWED_INPUT_PATHS,
    source_cap: int = SOURCE_CAP_PER_DATASET_LABEL,
    seed: int = SEED,
    frozen_at_utc: str | None = None,
) -> FreezeArtifacts:
    """Create a hash-sealed source-selection manifest before H4 analysis.

    This function performs no statistic.  It records structural input problems
    rather than replacing rows, so the later analyzer can emit explicit failed
    cells for every affected dataset/view/feature combination.
    """
    if source_cap < 1:
        raise ValueError("H4 source cap must be positive")
    locked_inputs = validate_explicit_input_paths(input_paths, allowed_paths=allowed_paths)
    output = Path(output_dir)
    _require_new_output_dir(output)
    frozen_at = frozen_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    records: list[dict[str, object]] = []
    selections: list[pd.DataFrame] = []
    for dataset in locked_inputs:
        table = _read_allowed_table(locked_inputs[dataset])
        record, selection = _input_record(dataset, locked_inputs[dataset], table, cap=source_cap, seed=seed)
        records.append(record)
        selections.append(selection)
    selection_table = pd.concat(selections, ignore_index=True)
    if selection_table.duplicated(["dataset", "label", "source_id"]).any():
        raise RuntimeError("H4 freeze generated duplicate selected source keys")

    output.mkdir()
    manifest_path = output / "h4_input_selection_manifest.parquet"
    selection_table.to_parquet(manifest_path, index=False)
    manifest_sha = sha256_file(manifest_path)
    provenance = {
        "h4_version": H4_VERSION,
        "protocol_path": "experiments/h4_label_transportability/protocol.md",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "frozen_at_utc": frozen_at,
        "score_free_classifier_free": True,
        "feature_version": FEATURE_VERSION,
        "datasets": list(locked_inputs),
        "views": list(VIEWS),
        "features": list(FEATURE_NAMES),
        "allowed_columns": list(ALLOWED_COLUMNS),
        "forbidden_terms": list(FORBIDDEN_TERMS),
        "source_cap_per_dataset_label": int(source_cap),
        "selection_seed": int(seed),
        "selection_manifest": {
            "absolute_path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "byte_size": int(manifest_path.stat().st_size),
            "n_rows": int(len(selection_table)),
            "columns": selection_table.columns.tolist(),
        },
        "inputs": records,
    }
    provenance_path = output / "h4_input_freeze_provenance.json"
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return FreezeArtifacts(
        output_dir=output,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        manifest_sha256=manifest_sha,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read H4 provenance JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("H4 provenance must be a JSON object")
    return value


def validate_frozen_provenance(
    provenance_path: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H4_ALLOWED_INPUT_PATHS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate freeze immutability and return its selected-source manifest.

    Analysis never accepts a feature path directly.  The freeze provenance is
    the sole route to a feature table, and every current file hash/size/path is
    checked before a label--feature value is read.
    """
    provenance_file = Path(provenance_path)
    provenance = _load_json(provenance_file)
    if provenance.get("h4_version") != H4_VERSION:
        raise ValueError("Unsupported or missing H4 provenance version")
    if provenance.get("feature_version") != FEATURE_VERSION:
        raise ValueError("H4 provenance has an unexpected feature registry version")
    if provenance.get("datasets") != list(allowed_paths):
        raise ValueError("H4 provenance dataset order/scope does not match the locked protocol")
    if provenance.get("views") != list(VIEWS) or provenance.get("features") != list(FEATURE_NAMES):
        raise ValueError("H4 provenance views/features do not match the locked protocol")
    if provenance.get("allowed_columns") != list(ALLOWED_COLUMNS):
        raise ValueError("H4 provenance allowed-column schema does not match the locked protocol")
    if provenance.get("source_cap_per_dataset_label") != SOURCE_CAP_PER_DATASET_LABEL:
        raise ValueError("H4 provenance source cap does not match the locked protocol")
    if provenance.get("selection_seed") != SEED:
        raise ValueError("H4 provenance selection seed does not match the locked protocol")
    if sha256_file(PROTOCOL_PATH) != provenance.get("protocol_sha256"):
        raise ValueError("H4 protocol changed after the input freeze; create a new freeze")

    records = provenance.get("inputs")
    if not isinstance(records, list) or len(records) != len(allowed_paths):
        raise ValueError("H4 provenance must contain exactly five input records")
    record_by_dataset: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("dataset") in record_by_dataset:
            raise ValueError("H4 provenance input records must have unique datasets")
        record_by_dataset[str(record["dataset"])] = record
    if set(record_by_dataset) != set(allowed_paths):
        raise ValueError("H4 provenance input records are outside the locked five-dataset scope")

    # Reuse the exact same firewall as the freeze, then verify the current
    # bytes.  A moved, changed, or substituted table is a hard stop, not a
    # new analysis with silently different data.
    current_paths = {dataset: Path(record_by_dataset[dataset].get("absolute_path", "")) for dataset in allowed_paths}
    locked_paths = validate_explicit_input_paths(current_paths, allowed_paths=allowed_paths)
    for dataset, path in locked_paths.items():
        record = record_by_dataset[dataset]
        if record.get("allowed_column_schema") != list(ALLOWED_COLUMNS):
            raise ValueError(f"H4 input record allowed schema mismatch for {dataset}")
        if int(record.get("byte_size", -1)) != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"H4 input changed after freeze for {dataset}; create a new freeze")
        _schema_columns(path)

    selection_meta = provenance.get("selection_manifest")
    if not isinstance(selection_meta, dict):
        raise ValueError("H4 provenance lacks a selection manifest record")
    manifest_path = Path(selection_meta.get("absolute_path", ""))
    if not manifest_path.is_absolute() or manifest_path.parent != provenance_file.parent.resolve():
        raise ValueError("H4 selected-source manifest path is outside its freeze directory")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"H4 selected-source manifest missing: {manifest_path}")
    if int(selection_meta.get("byte_size", -1)) != manifest_path.stat().st_size or selection_meta.get("sha256") != sha256_file(manifest_path):
        raise ValueError("H4 selected-source manifest changed after freeze")
    manifest = pd.read_parquet(manifest_path)
    expected_columns = ["dataset", "label", "source_id", "selection_rank", "selection_key_sha256"]
    if manifest.columns.tolist() != expected_columns:
        raise ValueError("H4 selected-source manifest schema mismatch")
    if len(manifest) != int(selection_meta.get("n_rows", -1)):
        raise ValueError("H4 selected-source manifest row count mismatch")
    if manifest.duplicated(["dataset", "label", "source_id"]).any():
        raise ValueError("H4 selected-source manifest has duplicate source keys")
    if set(manifest["dataset"].astype(str)) - set(allowed_paths):
        raise ValueError("H4 selected-source manifest includes an out-of-scope dataset")
    if not manifest["label"].isin([0, 1]).all() or not manifest["source_id"].map(_string_valid).all():
        raise ValueError("H4 selected-source manifest contains malformed label/source data")
    if (manifest.groupby(["dataset", "label"], sort=True).size() > SOURCE_CAP_PER_DATASET_LABEL).any():
        raise ValueError("H4 selected-source manifest exceeds the locked source cap")
    return provenance, manifest


def _status_from_validation(record: Mapping[str, Any]) -> str | None:
    failures = record.get("validation_failures", [])
    if not isinstance(failures, list):
        return "failed_invalid_freeze_validation"
    if failures:
        return "failed_input_" + "_and_".join(str(item) for item in sorted(failures))
    selected = record.get("n_selected_sources_by_label", {})
    if not isinstance(selected, Mapping):
        return "failed_invalid_freeze_selection"
    for label in (0, 1):
        if int(selected.get(f"label_{label}", 0)) < 1:
            return f"failed_no_selected_source_label_{label}"
    return None


def _source_selected_rows(table: pd.DataFrame, selection: pd.DataFrame, dataset: str) -> pd.DataFrame:
    selected = selection.loc[selection["dataset"].astype(str).eq(dataset), ["label", "source_id"]].copy()
    selected["label"] = selected["label"].astype(int)
    selected["source_id"] = selected["source_id"].astype(str)
    frame = table.copy()
    labels = _normalised_labels(frame["label"])
    frame["_label"] = labels
    frame = frame.loc[frame["_label"].notna()].copy()
    frame["_label"] = frame["_label"].astype(int)
    frame["source_id"] = frame["source_id"].astype(str)
    return frame.merge(selected, how="inner", left_on=["_label", "source_id"], right_on=["label", "source_id"], validate="many_to_one")


def _rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for label 1 using stable ranks; no classifier is fit."""
    value0 = values[labels == 0]
    value1 = values[labels == 1]
    if not len(value0) or not len(value1):
        return float("nan")
    all_values = np.concatenate((value0, value1))
    order = np.argsort(all_values, kind="mergesort")
    sorted_values = all_values[order]
    ranks = np.empty(len(all_values), dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
    ends = np.r_[starts[1:], len(sorted_values)]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    rank_sum_positive = ranks[len(value0) :].sum()
    return float((rank_sum_positive - len(value1) * (len(value1) + 1) / 2.0) / (len(value0) * len(value1)))


def _bootstrap_delta_auc(
    finite: pd.DataFrame,
    *,
    feature: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Exact source-cluster percentile bootstrap, vectorised by replicate batch.

    The weighted Mann--Whitney form preserves each sampled source cluster's
    complete finite rows.  It avoids materialising 500 expanded DataFrames per
    cell, which matters for a 420-cell, 5,000-source-per-label audit.
    """
    values = finite[feature].to_numpy(dtype=float)
    labels = finite["label"].to_numpy(dtype=int)
    sources = finite["source_id"].astype(str).to_numpy()
    indices_by_label: dict[int, np.ndarray] = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    if not len(indices_by_label[0]) or not len(indices_by_label[1]):
        return {
            "bootstrap_replicates_requested": int(replicates),
            "bootstrap_replicates_valid": 0,
            "bootstrap_replicates_invalid": int(replicates),
            "bootstrap_confidence": CONFIDENCE,
            "bootstrap_n_input_rows": int(len(finite)),
            "bootstrap_n_strata": 2,
            "bootstrap_n_clusters": 0,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_excludes_zero": False,
        }

    source_ids: dict[int, np.ndarray] = {}
    source_row_index: dict[int, np.ndarray] = {}
    for label in (0, 1):
        source_ids[label], source_row_index[label] = np.unique(sources[indices_by_label[label]], return_inverse=True)
        if not len(source_ids[label]):
            raise RuntimeError("H4 bootstrap lost a nonempty source stratum")

    # Sort once and use source multiplicity weights for every resample.  This
    # yields the same tie-aware AUROC as resampling rows within each selected
    # source cluster and reranking, but is substantially less allocation-heavy.
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    batch_size = 32
    position = 0
    rows0 = indices_by_label[0]
    rows1 = indices_by_label[1]
    # Maps original finite-row position to source index within each label.
    source_index_for_row = np.full(len(finite), -1, dtype=int)
    source_index_for_row[rows0] = source_row_index[0]
    source_index_for_row[rows1] = source_row_index[1]
    for batch_start in range(0, replicates, batch_size):
        width = min(batch_size, replicates - batch_start)
        multiplicities: dict[int, np.ndarray] = {}
        for label in (0, 1):
            n_clusters = len(source_ids[label])
            draws = rng.integers(0, n_clusters, size=(width, n_clusters))
            counts = np.zeros((width, n_clusters), dtype=np.int32)
            np.add.at(counts, (np.arange(width)[:, None], draws), 1)
            multiplicities[label] = counts
        weights = np.zeros((width, len(finite)), dtype=np.int32)
        weights[:, rows0] = multiplicities[0][:, source_index_for_row[rows0]]
        weights[:, rows1] = multiplicities[1][:, source_index_for_row[rows1]]
        sorted_weights = weights[:, order]
        group0 = np.add.reduceat(sorted_weights * (sorted_labels == 0), starts, axis=1)
        group1 = np.add.reduceat(sorted_weights * (sorted_labels == 1), starts, axis=1)
        earlier0 = np.cumsum(group0, axis=1) - group0
        numerator = np.sum(group1 * (earlier0 + 0.5 * group0), axis=1, dtype=float)
        denominator = group0.sum(axis=1, dtype=float) * group1.sum(axis=1, dtype=float)
        batch_auc = numerator / denominator
        estimates[position : position + width] = 2.0 * batch_auc - 1.0
        position += width
    valid = estimates[np.isfinite(estimates)]
    alpha = (1.0 - CONFIDENCE) / 2.0
    low, high = (float("nan"), float("nan"))
    if len(valid):
        low, high = (float(value) for value in np.quantile(valid, [alpha, 1.0 - alpha]))
    return {
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid": int(len(valid)),
        "bootstrap_replicates_invalid": int(replicates - len(valid)),
        "bootstrap_confidence": CONFIDENCE,
        "bootstrap_n_input_rows": int(len(finite)),
        "bootstrap_n_strata": 2,
        "bootstrap_n_clusters": int(len(source_ids[0]) + len(source_ids[1])),
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(np.isfinite(low) and np.isfinite(high) and (low > 0.0 or high < 0.0)),
    }


def _failed_cell(dataset: str, view: str, feature: str, status: str, *, selected: pd.DataFrame | None = None) -> dict[str, object]:
    selected_counts = {label: 0 for label in (0, 1)}
    if selected is not None and not selected.empty:
        selected_counts = {label: int((selected["_label"] == label).sum()) for label in (0, 1)}
    return {
        "dataset": dataset,
        "view": view,
        "feature": feature,
        "cell_status": status,
        "selected_rows_label_0": selected_counts[0],
        "selected_rows_label_1": selected_counts[1],
        "finite_count_label_0": 0,
        "finite_count_label_1": 0,
        "finite_availability_label_0": float("nan"),
        "finite_availability_label_1": float("nan"),
        "label_auroc": float("nan"),
        "delta_auc": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_excludes_zero": False,
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid": 0,
        "bootstrap_replicates_invalid": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": CONFIDENCE,
        "bootstrap_n_input_rows": 0,
        "bootstrap_n_strata": 0,
        "bootstrap_n_clusters": 0,
        "bootstrap_seed": SEED,
    }


def _cell_row(dataset: str, view: str, feature: str, selected: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, object]:
    view_rows = selected.loc[selected["view"].eq(view)].copy()
    selected_counts = {label: int((view_rows["_label"] == label).sum()) for label in (0, 1)}
    finite = pd.to_numeric(view_rows[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    view_rows[feature] = finite
    finite_rows = view_rows.loc[view_rows[feature].notna()].copy()
    finite_counts = {label: int((finite_rows["_label"] == label).sum()) for label in (0, 1)}
    availability = {
        label: (finite_counts[label] / selected_counts[label] if selected_counts[label] else float("nan"))
        for label in (0, 1)
    }
    if not selected_counts[0] or not selected_counts[1]:
        return _failed_cell(dataset, view, feature, "failed_unavailable_selected_label", selected=view_rows)
    if not finite_counts[0] or not finite_counts[1]:
        row = _failed_cell(dataset, view, feature, "failed_unavailable_finite_label", selected=view_rows)
        row.update(
            {
                "finite_count_label_0": finite_counts[0],
                "finite_count_label_1": finite_counts[1],
                "finite_availability_label_0": availability[0],
                "finite_availability_label_1": availability[1],
            }
        )
        return row
    values = finite_rows[feature].to_numpy(dtype=float)
    labels = finite_rows["_label"].to_numpy(dtype=int)
    auc = _rank_auc(values, labels)
    bootstrap_input = finite_rows.rename(columns={"_label": "label"})[[feature, "label", "source_id"]]
    bootstrap = _bootstrap_delta_auc(bootstrap_input, feature=feature, replicates=replicates, seed=seed)
    return {
        "dataset": dataset,
        "view": view,
        "feature": feature,
        "cell_status": "ok",
        "selected_rows_label_0": selected_counts[0],
        "selected_rows_label_1": selected_counts[1],
        "finite_count_label_0": finite_counts[0],
        "finite_count_label_1": finite_counts[1],
        "finite_availability_label_0": availability[0],
        "finite_availability_label_1": availability[1],
        "label_auroc": auc,
        "delta_auc": 2.0 * auc - 1.0,
        "bootstrap_seed": seed,
        **bootstrap,
    }


def _direction(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "none"
    if not np.isfinite(numeric) or numeric == 0.0:
        return "none"
    return "positive" if numeric > 0.0 else "negative"


def aggregate_cells(matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply the locked H4 atlas-only rule to all 84 view--feature units."""
    expected = len(DATASETS) * len(VIEWS) * len(FEATURE_NAMES)
    if len(matrix) != expected or matrix.duplicated(["dataset", "view", "feature"]).any():
        raise ValueError("H4 aggregation requires exactly one exhaustive 420-cell matrix")
    rows: list[dict[str, object]] = []
    for view in VIEWS:
        for feature in FEATURE_NAMES:
            unit = matrix.loc[(matrix["view"] == view) & (matrix["feature"] == feature)].set_index("dataset")
            missing = [dataset for dataset in DATASETS if dataset not in unit.index]
            cells_ok = [dataset for dataset in DATASETS if dataset in unit.index and unit.at[dataset, "cell_status"] == "ok"]
            directions = {dataset: _direction(unit.at[dataset, "delta_auc"]) if dataset in unit.index else "none" for dataset in DATASETS}
            positive = sum(direction == "positive" for direction in directions.values())
            negative = sum(direction == "negative" for direction in directions.values())
            stable_direction = "positive" if positive >= 4 else ("negative" if negative >= 4 else "none")
            held_out_interval_same = bool(stable_direction != "none")
            held_out_abs_delta = bool(stable_direction != "none")
            availability_everywhere = True
            for dataset in DATASETS:
                if dataset not in unit.index:
                    availability_everywhere = False
                    continue
                availability_everywhere = availability_everywhere and all(
                    np.isfinite(float(unit.at[dataset, column])) and float(unit.at[dataset, column]) >= 0.80
                    for column in ("finite_availability_label_0", "finite_availability_label_1")
                )
            for dataset in HELD_OUT_DATASETS:
                if dataset not in unit.index or unit.at[dataset, "cell_status"] != "ok":
                    held_out_interval_same = False
                    held_out_abs_delta = False
                    continue
                low = float(unit.at[dataset, "ci_low"])
                high = float(unit.at[dataset, "ci_high"])
                same_interval = low > 0.0 if stable_direction == "positive" else high < 0.0
                held_out_interval_same = held_out_interval_same and same_interval
                held_out_abs_delta = held_out_abs_delta and abs(float(unit.at[dataset, "delta_auc"])) >= 0.10
            complete = not missing and len(cells_ok) == len(DATASETS)
            stable = bool(
                complete
                and stable_direction != "none"
                and held_out_interval_same
                and held_out_abs_delta
                and availability_everywhere
            )
            rows.append(
                {
                    "view": view,
                    "feature": feature,
                    "datasets_expected": len(DATASETS),
                    "datasets_with_ok_cells": len(cells_ok),
                    "failed_or_missing_datasets": ";".join(
                        dataset for dataset in DATASETS if dataset not in cells_ok
                    ),
                    "positive_delta_auc_datasets": positive,
                    "negative_delta_auc_datasets": negative,
                    "stable_direction": stable_direction,
                    "same_nonzero_sign_at_least_4_of_5": stable_direction != "none",
                    "held_out_intervals_exclude_zero_in_same_sign": held_out_interval_same,
                    "held_out_abs_delta_auc_at_least_0p10": held_out_abs_delta,
                    "finite_availability_at_least_80pct_all_labels_all_corpora": availability_everywhere,
                    "atlas_only_stable_label_association": stable,
                    "aggregation_status": (
                        "atlas_only_stable_label_association"
                        if stable
                        else "does_not_meet_atlas_only_stable_label_association_rule"
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze_frozen_manifest(
    provenance_path: Path | str,
    output_dir: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H4_ALLOWED_INPUT_PATHS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> tuple[Path, Path, Path]:
    """Materialise the exhaustive H4 matrix from one validated input freeze."""
    if bootstrap_replicates < 1:
        raise ValueError("H4 bootstrap replicates must be positive")
    if seed != SEED:
        raise ValueError("H4 analysis seed is locked to 2609")
    provenance, selection = validate_frozen_provenance(provenance_path, allowed_paths=allowed_paths)
    output = Path(output_dir)
    _require_new_output_dir(output)
    records = {str(record["dataset"]): record for record in provenance["inputs"]}
    cells: list[dict[str, object]] = []
    for dataset in allowed_paths:
        record = records[dataset]
        input_status = _status_from_validation(record)
        if input_status is None:
            table = _read_allowed_table(Path(record["absolute_path"]))
            selected = _source_selected_rows(table, selection, dataset)
            # The input is hash-verified, but this explicit count protects
            # against a malformed or hand-edited selection manifest.
            expected_sources = selection.loc[selection["dataset"].eq(dataset), ["label", "source_id"]]
            observed_sources = selected[["_label", "source_id"]].drop_duplicates()
            if len(observed_sources) != len(expected_sources):
                input_status = "failed_selected_source_absent_from_input"
        else:
            selected = None
        for view in VIEWS:
            for feature in FEATURE_NAMES:
                if input_status is not None:
                    cells.append(_failed_cell(dataset, view, feature, input_status, selected=selected))
                else:
                    assert selected is not None
                    cells.append(_cell_row(dataset, view, feature, selected, replicates=bootstrap_replicates, seed=seed))
    matrix = pd.DataFrame(cells)
    if len(matrix) != len(DATASETS) * len(VIEWS) * len(FEATURE_NAMES):
        raise RuntimeError("H4 did not materialise the locked exhaustive 420-cell matrix")
    aggregation = aggregate_cells(matrix)
    if len(aggregation) != len(VIEWS) * len(FEATURE_NAMES):
        raise RuntimeError("H4 did not materialise the locked 84-unit aggregation")
    output.mkdir()
    matrix_path = output / "h4_label_cue_matrix.csv"
    aggregation_path = output / "h4_label_cue_aggregation.csv"
    matrix.to_csv(matrix_path, index=False)
    aggregation.to_csv(aggregation_path, index=False)
    report_path = output / "h4_analysis_provenance.json"
    report = {
        "h4_version": H4_VERSION,
        "score_free_classifier_free": True,
        "input_freeze_provenance_absolute_path": str(Path(provenance_path).resolve()),
        "input_freeze_provenance_sha256": sha256_file(Path(provenance_path)),
        "selection_manifest_sha256": provenance["selection_manifest"]["sha256"],
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_confidence": CONFIDENCE,
        "n_dataset_view_feature_cells": int(len(matrix)),
        "n_view_feature_aggregation_units": int(len(aggregation)),
        "n_ok_cells": int((matrix["cell_status"] == "ok").sum()),
        "n_failed_cells": int((matrix["cell_status"] != "ok").sum()),
        "n_atlas_only_stable_units": int(aggregation["atlas_only_stable_label_association"].sum()),
        "matrix": {
            "filename": matrix_path.name,
            "sha256": sha256_file(matrix_path),
            "byte_size": int(matrix_path.stat().st_size),
        },
        "aggregation": {
            "filename": aggregation_path.name,
            "sha256": sha256_file(aggregation_path),
            "byte_size": int(aggregation_path.stat().st_size),
        },
    }
    report_path.write_text(canonical_json(report), encoding="utf-8")
    return matrix_path, aggregation_path, report_path
