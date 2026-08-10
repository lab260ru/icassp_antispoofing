"""Strictly score- and label-free infrastructure for the locked H5 atlas.

H5 consumes only the immutable ``v1_28`` feature products named in its
protocol.  Its freeze reads *only* ``sample_id`` and ``view``; its analyzer
adds the registered feature columns, and no other field.  In particular, a
feature container may physically carry label metadata for another experiment,
but H5 never projects, materialises, validates against, or writes that field.
It does not decode audio, import a detector/ASR/model, train, or access prior
result artifacts.
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
from scipy.stats import rankdata

from src.audio_features import FEATURE_NAMES, FEATURE_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiments/h5_view_invariance/protocol.md"
H5_VERSION = "h5_paired_view_invariance_v1"
SEED = 2610
SAMPLE_CAP_PER_DATASET = 10_000
BOOTSTRAP_REPLICATES = 200
CONFIDENCE = 0.95
BOOTSTRAP_BATCH_SIZE = 16

DATASETS = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
VIEWS = ("full_waveform", "deterministic_crop", "preemphasized_crop")
VIEW_PAIRS = tuple(combinations(tuple(sorted(VIEWS)), 2))
IDENTITY_COLUMNS = ("sample_id", "view")
ALLOWED_COLUMNS = (*IDENTITY_COLUMNS, *FEATURE_NAMES)

# These terms are prohibited in an H5 path and in a source header.  Labels are
# separately prohibited from H5's exact projection (``ALLOWED_COLUMNS``), not
# from the physical feature container: the pre-existing v1_28 tables contain a
# label metadata column for other locked studies.  Reading its *name* from a
# Parquet schema would not read its values, but rejecting the container would
# make every protocol-declared H5 input impossible to use.  The strict boundary
# is therefore the exact Arrow/Pandas projection below.
FORBIDDEN_RESPONSE_TERMS = (
    "score",
    "logit",
    "detector",
    "model",
    "eer",
    "arena",
    "audio",
    "asr",
    "h1",
    "h2",
    "h2b",
    "h4",
    "result",
    "summary",
)
FORBIDDEN_OUTPUT_TERMS = ("label", *FORBIDDEN_RESPONSE_TERMS)

H5_ALLOWED_INPUT_PATHS: dict[str, Path] = {
    dataset: Path(
        "/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/"
        f"{FEATURE_VERSION}/{dataset}/features_wide.parquet"
    )
    for dataset in DATASETS
}


@dataclass(frozen=True)
class FreezeArtifacts:
    """Paths and hash emitted by a non-overwritable H5 input freeze."""

    output_dir: Path
    manifest_path: Path
    provenance_path: Path
    manifest_sha256: str


def canonical_json(value: object) -> str:
    """Return stable, human-readable JSON for an integrity record."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_forbidden_term(value: str, terms: tuple[str, ...] = FORBIDDEN_RESPONSE_TERMS) -> bool:
    lowered = value.casefold()
    return any(term in lowered for term in terms)


def _path_is_safe(path: Path) -> bool:
    return not _has_forbidden_term(str(path))


def _as_absolute_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"H5 input path must be absolute: {candidate}")
    if not _path_is_safe(candidate):
        raise ValueError(f"H5 path firewall rejected forbidden path: {candidate}")
    return candidate


def validate_explicit_input_paths(
    input_paths: Mapping[str, Path | str],
    *,
    allowed_paths: Mapping[str, Path | str] = H5_ALLOWED_INPUT_PATHS,
) -> dict[str, Path]:
    """Require exactly the five literal, protocol-allowed feature products.

    ``allowed_paths`` is injectable solely for synthetic tests.  The public
    command-line tools do not expose an alternate source path and always use
    ``H5_ALLOWED_INPUT_PATHS``.
    """
    expected = tuple(allowed_paths)
    if set(input_paths) != set(expected):
        missing = sorted(set(expected) - set(input_paths))
        extra = sorted(set(input_paths) - set(expected))
        raise ValueError(f"H5 requires exactly the five locked datasets; missing={missing}, extra={extra}")
    resolved: dict[str, Path] = {}
    for dataset in expected:
        supplied = _as_absolute_path(input_paths[dataset])
        allowed = _as_absolute_path(allowed_paths[dataset])
        if supplied != allowed or supplied.resolve() != allowed.resolve():
            raise ValueError(f"H5 path firewall rejected {dataset}: supplied={supplied}, locked={allowed}")
        if not supplied.is_file():
            raise FileNotFoundError(f"H5 locked input is not a regular file: {supplied}")
        resolved[dataset] = supplied
    return resolved


def _schema_columns(path: Path) -> list[str]:
    """Validate Parquet headers before reading identity or feature values."""
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    if len(columns) != len(set(columns)):
        raise ValueError(f"H5 input has duplicate Parquet column names: {path}")
    missing = sorted(set(ALLOWED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"H5 input missing required projected columns at {path}: {missing}")
    response_like = sorted(column for column in columns if _has_forbidden_term(column))
    if response_like:
        raise ValueError(f"H5 source-column firewall rejected response-like columns at {path}: {response_like}")
    if any(_has_forbidden_term(column, FORBIDDEN_OUTPUT_TERMS) for column in ALLOWED_COLUMNS):
        raise RuntimeError("H5 locked projection unexpectedly contains a forbidden term")
    return columns


def _read_exact_projection(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read an exact, schema-validated projection and nothing else."""
    _schema_columns(path)
    table = pd.read_parquet(path, columns=list(columns))
    if tuple(table.columns) != columns:
        table = table.loc[:, list(columns)].copy()
    if tuple(table.columns) != columns:
        raise ValueError("H5 projection order drifted from the locked schema")
    return table


def _read_identity_table(path: Path) -> pd.DataFrame:
    """Read only ``sample_id`` and ``view`` for the pre-feature input freeze."""
    return _read_exact_projection(path, IDENTITY_COLUMNS)


def _read_feature_table(path: Path) -> pd.DataFrame:
    """Read identities plus the 28 registered features for a validated freeze."""
    return _read_exact_projection(path, ALLOWED_COLUMNS)


def _string_valid(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _validation_record(table: pd.DataFrame) -> dict[str, object]:
    """Record identity/view integrity without inspecting feature values."""
    sample_valid = table["sample_id"].map(_string_valid)
    view_valid = table["view"].map(_string_valid)
    duplicate_rows = int(table.duplicated(["sample_id", "view"], keep=False).sum())
    observed = sorted(value for value in table.loc[view_valid, "view"].astype(str).unique() if value in VIEWS)
    unexpected = sorted(value for value in table.loc[view_valid, "view"].astype(str).unique() if value not in VIEWS)
    complete_sample_count = 0
    incomplete_sample_count = 0
    valid = table.loc[sample_valid & view_valid & table["view"].isin(VIEWS), list(IDENTITY_COLUMNS)]
    if not valid.empty:
        per_sample = valid.groupby("sample_id", sort=True)["view"].agg(lambda values: set(values))
        complete_sample_count = int(sum(set(VIEWS) == set(values) for values in per_sample))
        incomplete_sample_count = int(sum(set(VIEWS) != set(values) for values in per_sample))
    return {
        "malformed_sample_id_rows": int((~sample_valid).sum()),
        "malformed_view_rows": int((~view_valid).sum()),
        "duplicate_sample_view_rows": duplicate_rows,
        "missing_required_views": [view for view in VIEWS if view not in observed],
        "unexpected_views": unexpected,
        "incomplete_sample_view_sets": incomplete_sample_count,
        "complete_sample_view_sets": complete_sample_count,
    }


def _integrity_failures(validation: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    for field, name in (
        ("malformed_sample_id_rows", "malformed_sample_id"),
        ("malformed_view_rows", "malformed_view"),
        ("duplicate_sample_view_rows", "duplicate_sample_view"),
        ("incomplete_sample_view_sets", "incomplete_sample_view_set"),
    ):
        if int(validation.get(field, 0)):
            failures.append(name)
    if validation.get("missing_required_views"):
        failures.append("missing_required_view")
    if validation.get("unexpected_views"):
        failures.append("unexpected_view")
    return failures


def _selection_rows(dataset: str, identities: pd.DataFrame) -> pd.DataFrame:
    """Select H5 sample IDs by its independent SHA-256 seed only."""
    valid = (
        identities["sample_id"].map(_string_valid)
        & identities["view"].map(_string_valid)
        & identities["view"].isin(VIEWS)
    )
    sample_ids = sorted(set(identities.loc[valid, "sample_id"].astype(str)))
    ranked = sorted(
        (
            hashlib.sha256(f"{SEED}|{dataset}|{sample_id}".encode("utf-8")).hexdigest(),
            sample_id,
        )
        for sample_id in sample_ids
    )
    rows = [
        {
            "dataset": dataset,
            "sample_id": sample_id,
            "selection_rank": rank,
            "selection_key_sha256": selection_key,
        }
        for rank, (selection_key, sample_id) in enumerate(ranked[:SAMPLE_CAP_PER_DATASET], start=1)
    ]
    return pd.DataFrame(rows, columns=["dataset", "sample_id", "selection_rank", "selection_key_sha256"])


def _input_record(dataset: str, path: Path, identities: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    _schema_columns(path)
    validation = _validation_record(identities)
    selection = _selection_rows(dataset, identities)
    record: dict[str, object] = {
        "dataset": dataset,
        "absolute_path": str(path),
        "byte_size": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "projected_columns": list(IDENTITY_COLUMNS),
        "analysis_projection_columns": list(ALLOWED_COLUMNS),
        "projected_columns_only": True,
        "feature_version": FEATURE_VERSION,
        "n_identity_rows": int(len(identities)),
        "per_view_row_counts": {view: int(identities["view"].eq(view).sum()) for view in VIEWS},
        "n_distinct_valid_samples": int(
            identities.loc[
                identities["sample_id"].map(_string_valid)
                & identities["view"].map(_string_valid)
                & identities["view"].isin(VIEWS),
                "sample_id",
            ].nunique()
        ),
        "n_selected_samples": int(len(selection)),
        "validation": validation,
        "validation_failures": _integrity_failures(validation),
    }
    return record, selection


def _require_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"H5 artifacts are non-overwritable; output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def freeze_inputs(
    input_paths: Mapping[str, Path | str],
    output_dir: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H5_ALLOWED_INPUT_PATHS,
    frozen_at_utc: str | None = None,
) -> FreezeArtifacts:
    """Hash-seal an independent H5 sample panel before any feature statistic.

    This phase reads no feature values.  It is intentionally separate from all
    earlier study manifests and uses the protocol-locked seed 2610.
    """
    locked_inputs = validate_explicit_input_paths(input_paths, allowed_paths=allowed_paths)
    output = Path(output_dir)
    _require_new_output_dir(output)
    frozen_at = frozen_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    records: list[dict[str, object]] = []
    selections: list[pd.DataFrame] = []
    for dataset, path in locked_inputs.items():
        identities = _read_identity_table(path)
        record, selection = _input_record(dataset, path, identities)
        records.append(record)
        selections.append(selection)
    selection_table = pd.concat(selections, ignore_index=True)
    if selection_table.duplicated(["dataset", "sample_id"]).any():
        raise RuntimeError("H5 freeze generated duplicate selected sample keys")
    if (selection_table.groupby("dataset", sort=True).size() > SAMPLE_CAP_PER_DATASET).any():
        raise RuntimeError("H5 freeze exceeded its locked sample cap")

    output.mkdir()
    manifest_path = output / "h5_input_selection_manifest.parquet"
    selection_table.to_parquet(manifest_path, index=False)
    manifest_sha = sha256_file(manifest_path)
    failed_datasets = [record["dataset"] for record in records if record["validation_failures"] or not record["n_selected_samples"]]
    provenance = {
        "h5_version": H5_VERSION,
        "protocol_path": "experiments/h5_view_invariance/protocol.md",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "frozen_at_utc": frozen_at,
        "score_and_label_free": True,
        "feature_version": FEATURE_VERSION,
        "datasets": list(locked_inputs),
        "views": list(VIEWS),
        "view_pairs_lexicographic": [list(pair) for pair in VIEW_PAIRS],
        "features": list(FEATURE_NAMES),
        "freeze_projection_columns": list(IDENTITY_COLUMNS),
        "analysis_projection_columns": list(ALLOWED_COLUMNS),
        "forbidden_response_terms": list(FORBIDDEN_RESPONSE_TERMS),
        "sample_cap_per_dataset": SAMPLE_CAP_PER_DATASET,
        "selection_seed": SEED,
        "freeze_status": "complete" if not failed_datasets else "failed_input_integrity",
        "failed_datasets": failed_datasets,
        "selection_manifest": {
            "absolute_path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "byte_size": int(manifest_path.stat().st_size),
            "n_rows": int(len(selection_table)),
            "columns": selection_table.columns.tolist(),
        },
        "inputs": records,
    }
    provenance_path = output / "h5_input_freeze_provenance.json"
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return FreezeArtifacts(output, manifest_path, provenance_path, manifest_sha)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read H5 provenance JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("H5 provenance must be a JSON object")
    return value


def _records_by_dataset(provenance: Mapping[str, Any], allowed_paths: Mapping[str, Path | str]) -> dict[str, dict[str, Any]]:
    records = provenance.get("inputs")
    if not isinstance(records, list) or len(records) != len(allowed_paths):
        raise ValueError("H5 provenance must contain exactly five input records")
    by_dataset: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("dataset") in by_dataset:
            raise ValueError("H5 provenance input records must have unique datasets")
        by_dataset[str(record["dataset"])] = record
    if set(by_dataset) != set(allowed_paths):
        raise ValueError("H5 provenance input records are outside the locked five-dataset scope")
    return by_dataset


def validate_frozen_provenance(
    provenance_path: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H5_ALLOWED_INPUT_PATHS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Revalidate a sealed H5 freeze before reading any feature values."""
    provenance_file = Path(provenance_path)
    provenance = _load_json(provenance_file)
    if provenance.get("h5_version") != H5_VERSION:
        raise ValueError("Unsupported or missing H5 provenance version")
    if provenance.get("feature_version") != FEATURE_VERSION:
        raise ValueError("H5 provenance has an unexpected feature registry version")
    if provenance.get("datasets") != list(allowed_paths):
        raise ValueError("H5 provenance dataset order/scope does not match the locked protocol")
    if provenance.get("views") != list(VIEWS) or provenance.get("view_pairs_lexicographic") != [list(pair) for pair in VIEW_PAIRS]:
        raise ValueError("H5 provenance views/view-pair order does not match the locked protocol")
    if provenance.get("features") != list(FEATURE_NAMES):
        raise ValueError("H5 provenance feature registry does not match the locked protocol")
    if provenance.get("freeze_projection_columns") != list(IDENTITY_COLUMNS) or provenance.get("analysis_projection_columns") != list(ALLOWED_COLUMNS):
        raise ValueError("H5 provenance projection schema does not match the locked protocol")
    if provenance.get("sample_cap_per_dataset") != SAMPLE_CAP_PER_DATASET:
        raise ValueError("H5 provenance sample cap does not match the locked protocol")
    if provenance.get("selection_seed") != SEED:
        raise ValueError("H5 provenance selection seed does not match the locked protocol")
    if sha256_file(PROTOCOL_PATH) != provenance.get("protocol_sha256"):
        raise ValueError("H5 protocol changed after the input freeze; create a new freeze")

    records = _records_by_dataset(provenance, allowed_paths)
    current_paths = {dataset: Path(records[dataset].get("absolute_path", "")) for dataset in allowed_paths}
    locked_paths = validate_explicit_input_paths(current_paths, allowed_paths=allowed_paths)
    expected_selections: list[pd.DataFrame] = []
    for dataset, path in locked_paths.items():
        record = records[dataset]
        if record.get("projected_columns") != list(IDENTITY_COLUMNS) or record.get("analysis_projection_columns") != list(ALLOWED_COLUMNS):
            raise ValueError(f"H5 input record projection mismatch for {dataset}")
        if int(record.get("byte_size", -1)) != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"H5 input changed after freeze for {dataset}; create a new freeze")
        identities = _read_identity_table(path)
        if record.get("validation") != _validation_record(identities):
            raise ValueError(f"H5 identity/view validation changed after freeze for {dataset}")
        expected_selections.append(_selection_rows(dataset, identities))

    selection_meta = provenance.get("selection_manifest")
    if not isinstance(selection_meta, dict):
        raise ValueError("H5 provenance lacks a selection manifest record")
    manifest_path = Path(selection_meta.get("absolute_path", ""))
    if not manifest_path.is_absolute() or manifest_path.parent != provenance_file.parent.resolve():
        raise ValueError("H5 selected-sample manifest path is outside its freeze directory")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"H5 selected-sample manifest missing: {manifest_path}")
    if int(selection_meta.get("byte_size", -1)) != manifest_path.stat().st_size or selection_meta.get("sha256") != sha256_file(manifest_path):
        raise ValueError("H5 selected-sample manifest changed after freeze")
    manifest = pd.read_parquet(manifest_path)
    expected_columns = ["dataset", "sample_id", "selection_rank", "selection_key_sha256"]
    if manifest.columns.tolist() != expected_columns or len(manifest) != int(selection_meta.get("n_rows", -1)):
        raise ValueError("H5 selected-sample manifest schema or row count mismatch")
    if manifest.duplicated(["dataset", "sample_id"]).any() or not manifest["sample_id"].map(_string_valid).all():
        raise ValueError("H5 selected-sample manifest contains duplicate or malformed sample keys")
    if set(manifest["dataset"].astype(str)) != set(allowed_paths):
        raise ValueError("H5 selected-sample manifest does not cover exactly the locked datasets")
    if (manifest.groupby("dataset", sort=True).size() > SAMPLE_CAP_PER_DATASET).any():
        raise ValueError("H5 selected-sample manifest exceeds the locked cap")
    expected = pd.concat(expected_selections, ignore_index=True).sort_values(expected_columns).reset_index(drop=True)
    observed = manifest.sort_values(expected_columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("H5 selected-sample manifest no longer matches the independent locked selection") from error
    return provenance, manifest


def _input_status(record: Mapping[str, Any]) -> str | None:
    failures = record.get("validation_failures")
    if not isinstance(failures, list):
        return "failed_invalid_freeze_validation"
    if failures:
        return "failed_input_" + "_and_".join(str(item) for item in sorted(failures))
    if int(record.get("n_selected_samples", 0)) < 1:
        return "failed_no_selected_samples"
    return None


def _selected_rows(table: pd.DataFrame, selection: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str | None]:
    selected_ids = selection.loc[selection["dataset"].astype(str).eq(dataset), "sample_id"].astype(str).tolist()
    selected_set = set(selected_ids)
    rows = table.loc[table["sample_id"].astype(str).isin(selected_set)].copy()
    if len(selected_set) != len(selected_ids):
        return rows, "failed_duplicate_selected_sample"
    if rows.duplicated(["sample_id", "view"], keep=False).any():
        return rows, "failed_duplicate_sample_view"
    observed_ids = set(rows["sample_id"].astype(str))
    if observed_ids != selected_set:
        return rows, "failed_selected_sample_absent_from_input"
    per_sample_views = rows.groupby("sample_id", sort=True)["view"].agg(lambda values: set(values))
    if any(set(values) != set(VIEWS) for values in per_sample_views):
        return rows, "failed_selected_sample_missing_or_extra_view"
    return rows, None


def _cell_seed(dataset: str, view_a: str, view_b: str, feature: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{SEED}|{dataset}|{view_a}|{view_b}|{feature}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), digest


def _spearman_concordance(values_a: np.ndarray, values_b: np.ndarray) -> float:
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


def _paired_statistics(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    concordance = _spearman_concordance(values_a, values_b)
    pooled = np.concatenate((values_a, values_b))
    q1, q3 = np.quantile(pooled, [0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr == 0.0:
        return concordance, float("nan")
    return concordance, float(np.median(values_b - values_a) / iqr)


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = (1.0 - CONFIDENCE) / 2.0
    return tuple(float(value) for value in np.quantile(finite, [alpha, 1.0 - alpha]))


def _paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Exact paired-sample percentile bootstrap for both registered metrics."""
    n_pairs = len(values_a)
    concordances = np.empty(replicates, dtype=float)
    shifts = np.empty(replicates, dtype=float)
    rng = np.random.default_rng(seed)
    position = 0
    for start in range(0, replicates, BOOTSTRAP_BATCH_SIZE):
        width = min(BOOTSTRAP_BATCH_SIZE, replicates - start)
        indices = rng.integers(0, n_pairs, size=(width, n_pairs))
        sampled_a = values_a[indices]
        sampled_b = values_b[indices]
        ranks_a = rankdata(sampled_a, method="average", axis=1)
        ranks_b = rankdata(sampled_b, method="average", axis=1)
        centered_a = ranks_a - ranks_a.mean(axis=1, keepdims=True)
        centered_b = ranks_b - ranks_b.mean(axis=1, keepdims=True)
        denominators = np.sqrt((centered_a * centered_a).sum(axis=1) * (centered_b * centered_b).sum(axis=1))
        concordances[position : position + width] = np.divide(
            (centered_a * centered_b).sum(axis=1),
            denominators,
            out=np.full(width, np.nan),
            where=denominators > 0.0,
        )
        pooled = np.concatenate((sampled_a, sampled_b), axis=1)
        q1, q3 = np.quantile(pooled, [0.25, 0.75], axis=1)
        iqr = q3 - q1
        shifts[position : position + width] = np.divide(
            np.median(sampled_b - sampled_a, axis=1),
            iqr,
            out=np.full(width, np.nan),
            where=np.isfinite(iqr) & (iqr != 0.0),
        )
        position += width
    concordance_low, concordance_high = _percentile_interval(concordances)
    shift_low, shift_high = _percentile_interval(shifts)
    return {
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid_concordance": int(np.isfinite(concordances).sum()),
        "bootstrap_replicates_valid_shift": int(np.isfinite(shifts).sum()),
        "bootstrap_replicates_invalid_concordance": int((~np.isfinite(concordances)).sum()),
        "bootstrap_replicates_invalid_shift": int((~np.isfinite(shifts)).sum()),
        "bootstrap_confidence": CONFIDENCE,
        "concordance_ci_low": concordance_low,
        "concordance_ci_high": concordance_high,
        "normalized_shift_ci_low": shift_low,
        "normalized_shift_ci_high": shift_high,
    }


def _failed_cell(dataset: str, view_a: str, view_b: str, feature: str, status: str) -> dict[str, object]:
    seed, seed_hash = _cell_seed(dataset, view_a, view_b, feature)
    return {
        "dataset": dataset,
        "view_a": view_a,
        "view_b": view_b,
        "feature": feature,
        "cell_status": status,
        "paired_finite_sample_count": 0,
        "spearman_concordance": float("nan"),
        "iqr_normalized_median_shift": float("nan"),
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid_concordance": 0,
        "bootstrap_replicates_valid_shift": 0,
        "bootstrap_replicates_invalid_concordance": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_invalid_shift": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": CONFIDENCE,
        "concordance_ci_low": float("nan"),
        "concordance_ci_high": float("nan"),
        "normalized_shift_ci_low": float("nan"),
        "normalized_shift_ci_high": float("nan"),
        "bootstrap_seed": seed,
        "bootstrap_seed_sha256": seed_hash,
    }


def _cell_row(dataset: str, view_a: str, view_b: str, feature: str, selected: pd.DataFrame) -> dict[str, object]:
    wide = selected.pivot(index="sample_id", columns="view", values=feature)
    if view_a not in wide.columns or view_b not in wide.columns:
        return _failed_cell(dataset, view_a, view_b, feature, "failed_required_view_absent")
    values_a = pd.to_numeric(wide[view_a], errors="coerce").to_numpy(dtype=float)
    values_b = pd.to_numeric(wide[view_b], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = values_a[finite]
    values_b = values_b[finite]
    if len(values_a) < 2:
        return _failed_cell(dataset, view_a, view_b, feature, "failed_insufficient_finite_pairs")
    concordance, shift = _paired_statistics(values_a, values_b)
    if not np.isfinite(shift):
        return _failed_cell(dataset, view_a, view_b, feature, "failed_zero_or_nonfinite_pooled_iqr")
    if not np.isfinite(concordance):
        return _failed_cell(dataset, view_a, view_b, feature, "failed_nonfinite_rank_concordance")
    seed, seed_hash = _cell_seed(dataset, view_a, view_b, feature)
    bootstrap = _paired_bootstrap(values_a, values_b, replicates=BOOTSTRAP_REPLICATES, seed=seed)
    # A paired resample can contain only one tied observed value even when the
    # original pair has a well-defined rank concordance.  The protocol locks
    # 200 *attempted* resamples, not an unsupported rejection/redraw rule;
    # retain those invalid attempts explicitly and form percentile intervals
    # from the valid paired resamples.  Replacing them would be an unregistered
    # estimator change.  The aggregation separately requires finite intervals.
    return {
        "dataset": dataset,
        "view_a": view_a,
        "view_b": view_b,
        "feature": feature,
        "cell_status": "ok",
        "paired_finite_sample_count": int(len(values_a)),
        "spearman_concordance": concordance,
        "iqr_normalized_median_shift": shift,
        **bootstrap,
        "bootstrap_seed": seed,
        "bootstrap_seed_sha256": seed_hash,
    }


def aggregate_cells(matrix: pd.DataFrame) -> pd.DataFrame:
    """Materialise all 84 locked view-pair/feature descriptive summaries."""
    expected = len(DATASETS) * len(VIEW_PAIRS) * len(FEATURE_NAMES)
    if len(matrix) != expected or matrix.duplicated(["dataset", "view_a", "view_b", "feature"]).any():
        raise ValueError("H5 aggregation requires exactly one exhaustive 420-cell matrix")
    rows: list[dict[str, object]] = []
    for view_a, view_b in VIEW_PAIRS:
        for feature in FEATURE_NAMES:
            unit = matrix.loc[
                (matrix["view_a"] == view_a) & (matrix["view_b"] == view_b) & (matrix["feature"] == feature)
            ].set_index("dataset")
            missing = [dataset for dataset in DATASETS if dataset not in unit.index]
            cells_ok = [dataset for dataset in DATASETS if dataset in unit.index and unit.at[dataset, "cell_status"] == "ok"]
            all_ok = not missing and len(cells_ok) == len(DATASETS)
            concordances = pd.to_numeric(unit.loc[cells_ok, "spearman_concordance"], errors="coerce").to_numpy(dtype=float)
            shifts = pd.to_numeric(unit.loc[cells_ok, "iqr_normalized_median_shift"], errors="coerce").to_numpy(dtype=float)
            ci_lows = pd.to_numeric(unit.loc[cells_ok, "concordance_ci_low"], errors="coerce").to_numpy(dtype=float)
            shift_ci_lows = pd.to_numeric(unit.loc[cells_ok, "normalized_shift_ci_low"], errors="coerce").to_numpy(dtype=float)
            shift_ci_highs = pd.to_numeric(unit.loc[cells_ok, "normalized_shift_ci_high"], errors="coerce").to_numpy(dtype=float)
            stable = bool(
                all_ok
                and np.isfinite(concordances).all()
                and np.isfinite(ci_lows).all()
                and np.isfinite(shifts).all()
                and np.isfinite(shift_ci_lows).all()
                and np.isfinite(shift_ci_highs).all()
                and (concordances >= 0.90).all()
                and (ci_lows >= 0.80).all()
                and (shift_ci_lows >= -0.10).all()
                and (shift_ci_highs <= 0.10).all()
            )
            rows.append(
                {
                    "view_a": view_a,
                    "view_b": view_b,
                    "feature": feature,
                    "dataset_cell_count": int(len(unit)),
                    "successful_dataset_cell_count": int(len(cells_ok)),
                    "missing_datasets": ";".join(missing),
                    "all_dataset_cells_ok": all_ok,
                    "median_spearman_concordance": float(np.median(concordances)) if len(concordances) else float("nan"),
                    "median_absolute_iqr_normalized_shift": float(np.median(np.abs(shifts))) if len(shifts) else float("nan"),
                    "all_concordance_at_least_0p90": bool(all_ok and np.isfinite(concordances).all() and (concordances >= 0.90).all()),
                    "all_concordance_ci_lower_at_least_0p80": bool(all_ok and np.isfinite(ci_lows).all() and (ci_lows >= 0.80).all()),
                    "all_shift_intervals_within_plus_minus_0p10": bool(
                        all_ok
                        and np.isfinite(shift_ci_lows).all()
                        and np.isfinite(shift_ci_highs).all()
                        and (shift_ci_lows >= -0.10).all()
                        and (shift_ci_highs <= 0.10).all()
                    ),
                    "view_stable_terminal_descriptive": stable,
                    "aggregation_status": "view_stable_terminal_descriptive" if stable else "does_not_meet_view_stable_rule",
                }
            )
    return pd.DataFrame(rows)


def analyze_frozen_manifest(
    provenance_path: Path | str,
    output_dir: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H5_ALLOWED_INPUT_PATHS,
) -> tuple[Path, Path, Path]:
    """Materialise the H5 420-cell matrix only from a validated input freeze."""
    provenance, selection = validate_frozen_provenance(provenance_path, allowed_paths=allowed_paths)
    output = Path(output_dir)
    _require_new_output_dir(output)
    records = _records_by_dataset(provenance, allowed_paths)
    cells: list[dict[str, object]] = []
    for dataset in allowed_paths:
        status = _input_status(records[dataset])
        selected: pd.DataFrame | None = None
        if status is None:
            table = _read_feature_table(Path(records[dataset]["absolute_path"]))
            selected, status = _selected_rows(table, selection, dataset)
        for view_a, view_b in VIEW_PAIRS:
            for feature in FEATURE_NAMES:
                cells.append(
                    _failed_cell(dataset, view_a, view_b, feature, status)
                    if status is not None
                    else _cell_row(dataset, view_a, view_b, feature, selected)
                )
    matrix = pd.DataFrame(cells)
    expected_cells = len(DATASETS) * len(VIEW_PAIRS) * len(FEATURE_NAMES)
    if len(matrix) != expected_cells or matrix.duplicated(["dataset", "view_a", "view_b", "feature"]).any():
        raise RuntimeError("H5 did not materialise the locked exhaustive 420-cell matrix")
    aggregation = aggregate_cells(matrix)
    if len(aggregation) != len(VIEW_PAIRS) * len(FEATURE_NAMES):
        raise RuntimeError("H5 did not materialise the locked 84-unit aggregation")
    output.mkdir()
    matrix_path = output / "h5_view_invariance_matrix.csv"
    aggregation_path = output / "h5_view_invariance_aggregation.csv"
    matrix.to_csv(matrix_path, index=False)
    aggregation.to_csv(aggregation_path, index=False)
    report_path = output / "h5_analysis_provenance.json"
    report = {
        "h5_version": H5_VERSION,
        "score_and_label_free": True,
        "input_freeze_provenance_absolute_path": str(Path(provenance_path).resolve()),
        "input_freeze_provenance_sha256": sha256_file(Path(provenance_path)),
        "selection_manifest_sha256": provenance["selection_manifest"]["sha256"],
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": CONFIDENCE,
        "n_dataset_view_pair_feature_cells": int(len(matrix)),
        "n_view_pair_feature_aggregation_units": int(len(aggregation)),
        "failed_cell_count": int((matrix["cell_status"] != "ok").sum()),
        "matrix": {"absolute_path": str(matrix_path.resolve()), "sha256": sha256_file(matrix_path), "byte_size": int(matrix_path.stat().st_size)},
        "aggregation": {"absolute_path": str(aggregation_path.resolve()), "sha256": sha256_file(aggregation_path), "byte_size": int(aggregation_path.stat().st_size)},
    }
    report_path.write_text(canonical_json(report), encoding="utf-8")
    return matrix_path, aggregation_path, report_path
