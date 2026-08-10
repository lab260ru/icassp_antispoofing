"""Render only the sealed H6 compact agreement atlas as a supplementary figure.

The renderer is intentionally downstream-only.  It accepts the three literal,
hash-sealed compact products from ``h6_analysis_001`` and validates their
schema, registry, completeness, and internal aggregation identities before
displaying a fixed 28-by-10 matrix.  It never opens raw score artifacts,
model code or weights, audio, labels, feature products, or any other research
result.  The figure is descriptive: it is not a model ranking or selection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# A deterministic non-interactive backend is required on the research node.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from src.h6_score_agreement import (
    BOOTSTRAP_REPLICATES,
    CONFIDENCE,
    DATASETS,
    H6_VERSION,
    LABELS,
    MODEL_PAIRS,
    SEED,
)


H6_ANALYSIS_ROOT = Path(
    "/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001"
)
MATRIX_FILENAME = "h6_within_class_score_agreement_matrix.csv"
AGGREGATION_FILENAME = "h6_model_pair_agreement_summary.csv"
PROVENANCE_FILENAME = "h6_analysis_provenance.json"
PDF_FILENAME = "h6_within_class_score_agreement_heatmap.pdf"
PNG_FILENAME = "h6_within_class_score_agreement_heatmap.png"
METADATA_FILENAME = "h6_within_class_score_agreement_heatmap.metadata.json"

# These are the byte identities recorded at the completed H6 analysis boundary.
# They are not derived from, and never expose, any raw score values.
SEALED_MATRIX_SHA256 = "105e99fca829af8bf1600fa73374ddcde0f4121b6716a8beff0b1fd503eec2e5"
SEALED_AGGREGATION_SHA256 = "294faa27e8e9e7cce4292c0b5f9e05bbfb2667e88e1bb719113f81914a0326e8"
SEALED_PROVENANCE_SHA256 = "3e9d4f628fdffea5430859db589c32dce3ce5e9ce7c9b1b2448928ae4bd433d8"

MATRIX_COLUMNS = (
    "dataset",
    "label",
    "model_a",
    "model_b",
    "cell_status",
    "selected_sample_count",
    "model_a_available_score_count",
    "model_b_available_score_count",
    "exact_joined_sample_count",
    "spearman_agreement",
    "bootstrap_replicates_requested",
    "bootstrap_replicates_valid",
    "bootstrap_replicates_invalid",
    "bootstrap_confidence",
    "spearman_ci_low",
    "spearman_ci_high",
    "bootstrap_cluster_count",
    "bootstrap_seed",
    "bootstrap_seed_sha256",
)
AGGREGATION_COLUMNS = (
    "model_a",
    "model_b",
    "expected_dataset_class_cells",
    "observed_dataset_class_cells",
    "ok_cell_count",
    "failed_cell_count",
    "all_ten_cells_ok",
    "median_spearman_agreement",
    "minimum_spearman_agreement",
    "maximum_spearman_agreement",
    "cross_cell_spearman_range",
    "cell_statuses",
)

# These compact response-like fields are deliberately not part of either locked
# schema.  The exact-header check is the primary firewall; this explicit list
# makes the rejection reason clear if a raw/model-response substitute appears.
FORBIDDEN_RAW_RESPONSE_COLUMNS = (
    "raw_score",
    "score_a",
    "score_b",
    "logit",
    "probability",
    "embedding",
)

DATASET_DISPLAY_LABELS = {
    "ASVspoof2019_LA": "ASV19 LA",
    "ASVspoof2021_LA": "ASV21 LA",
    "ASVspoof2021_DF": "ASV21 DF",
    "InTheWild": "In the Wild",
    "ASVspoof5": "ASVspoof 5",
}
LABEL_DISPLAY_LABELS = {0: "bona fide", 1: "spoof"}


@dataclass(frozen=True)
class H6FigureInputContract:
    """Exact input locations and immutable byte identities for a display run."""

    matrix_path: Path
    aggregation_path: Path
    provenance_path: Path
    matrix_sha256: str
    aggregation_sha256: str
    provenance_sha256: str


SEALED_H6_FIGURE_INPUT_CONTRACT = H6FigureInputContract(
    matrix_path=H6_ANALYSIS_ROOT / MATRIX_FILENAME,
    aggregation_path=H6_ANALYSIS_ROOT / AGGREGATION_FILENAME,
    provenance_path=H6_ANALYSIS_ROOT / PROVENANCE_FILENAME,
    matrix_sha256=SEALED_MATRIX_SHA256,
    aggregation_sha256=SEALED_AGGREGATION_SHA256,
    provenance_sha256=SEALED_PROVENANCE_SHA256,
)


@dataclass(frozen=True)
class SealedH6FigureInputs:
    """Validated compact data only; raw model responses are never loaded."""

    matrix: pd.DataFrame
    aggregation: pd.DataFrame
    matrix_path: Path
    aggregation_path: Path
    provenance_path: Path
    matrix_sha256: str
    aggregation_sha256: str
    provenance_sha256: str


def canonical_json(value: object) -> str:
    """Return stable, human-readable metadata without raw response values."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    """Hash one compact artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_contract(contract: H6FigureInputContract) -> None:
    expected = {
        "matrix": (contract.matrix_path, MATRIX_FILENAME, contract.matrix_sha256),
        "aggregation": (contract.aggregation_path, AGGREGATION_FILENAME, contract.aggregation_sha256),
        "provenance": (contract.provenance_path, PROVENANCE_FILENAME, contract.provenance_sha256),
    }
    parents: set[Path] = set()
    for kind, (path, filename, expected_hash) in expected.items():
        if not path.is_absolute() or path.name != filename:
            raise ValueError(f"H6 figure contract has an impossible {kind} path: {path}")
        if not _is_sha256(expected_hash):
            raise ValueError(f"H6 figure contract has an invalid {kind} SHA-256")
        parents.add(path.parent)
    if len(parents) != 1:
        raise ValueError("H6 figure contract must keep all compact inputs in one analysis directory")


def _require_exact_input_path(path: Path | str, *, expected: Path, kind: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"H6 figure {kind} input must be an absolute path: {candidate}")
    if candidate != expected or candidate.resolve() != expected.resolve():
        raise ValueError(f"H6 figure requires the literal sealed h6_analysis_001 {kind} path")
    if candidate.name != expected.name or candidate.is_symlink():
        raise ValueError(f"H6 figure rejected an impossible or substituted {kind} path")
    if not candidate.is_file():
        raise FileNotFoundError(f"H6 figure {kind} input is not a regular file: {candidate}")
    return candidate


def _read_exact_csv(path: Path, expected_columns: tuple[str, ...], *, kind: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    raw_fields = [field for field in header if field.casefold() in FORBIDDEN_RAW_RESPONSE_COLUMNS]
    if raw_fields:
        raise ValueError(f"H6 figure rejected raw/model-response {kind} columns: {raw_fields}")
    if tuple(header) != expected_columns:
        raise ValueError(f"H6 figure {kind} schema does not exactly match the sealed compact schema")
    return pd.read_csv(path, usecols=list(expected_columns))


def _parse_boolean(values: pd.Series, *, field: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalised = values.astype("string").str.strip().str.casefold()
    if not normalised.isin(["true", "false"]).all():
        invalid = values.loc[~normalised.isin(["true", "false"])].head(5).tolist()
        raise ValueError(f"H6 figure {field} must contain only true/false values; saw {invalid}")
    return normalised.eq("true")


def _finite_numeric(table: pd.DataFrame, fields: tuple[str, ...], *, kind: str) -> None:
    for field in fields:
        values = pd.to_numeric(table[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"H6 figure {kind}.{field} must be finite numeric values")


def _seed_for_cell(dataset: str, label: int, model_a: str, model_b: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{SEED}|{dataset}|{label}|{model_a}|{model_b}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), digest


def _validate_matrix(matrix: pd.DataFrame) -> None:
    expected_rows = len(DATASETS) * len(LABELS) * len(MODEL_PAIRS)
    key = ["dataset", "label", "model_a", "model_b"]
    if len(matrix) != expected_rows:
        raise ValueError(f"H6 figure requires exactly {expected_rows} matrix cells; found {len(matrix)}")
    if matrix.duplicated(key, keep=False).any():
        raise ValueError("H6 figure matrix has duplicate dataset/class/model-pair cells")
    labels = pd.to_numeric(matrix["label"], errors="coerce")
    if not labels.isin(LABELS).all():
        raise ValueError("H6 figure matrix does not contain only the locked binary classes")
    matrix["label"] = labels.astype(int)
    expected_keys = {(dataset, label, model_a, model_b) for dataset in DATASETS for label in LABELS for model_a, model_b in MODEL_PAIRS}
    observed_keys = {
        (str(row.dataset), int(row.label), str(row.model_a), str(row.model_b)) for row in matrix.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise ValueError("H6 figure matrix does not exactly match the locked five-corpus model-pair registry")
    if set(matrix["cell_status"].astype(str)) != {"ok"}:
        raise ValueError("H6 figure requires the completed all-ok H6 matrix; unavailable cells are not displayable here")

    integer_fields = (
        "selected_sample_count",
        "model_a_available_score_count",
        "model_b_available_score_count",
        "exact_joined_sample_count",
        "bootstrap_replicates_requested",
        "bootstrap_replicates_valid",
        "bootstrap_replicates_invalid",
        "bootstrap_cluster_count",
        "bootstrap_seed",
    )
    _finite_numeric(matrix, integer_fields, kind="matrix")
    for field in integer_fields:
        numeric = pd.to_numeric(matrix[field], errors="raise")
        if not np.equal(numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float))).all():
            raise ValueError(f"H6 figure matrix.{field} must contain integers")
        # ``bootstrap_seed`` is a SHA-256-derived unsigned 64-bit value and
        # may exceed signed int64.  Preserve pandas' parsed unsigned dtype
        # rather than wrapping it through an ``int64`` cast.
        matrix[field] = numeric
    if not (
        matrix["selected_sample_count"].eq(5_000)
        & matrix["model_a_available_score_count"].eq(5_000)
        & matrix["model_b_available_score_count"].eq(5_000)
        & matrix["exact_joined_sample_count"].eq(5_000)
    ).all():
        raise ValueError("H6 figure requires exactly 5,000 selected/available/joined samples in every completed cell")
    if not matrix["bootstrap_replicates_requested"].eq(BOOTSTRAP_REPLICATES).all():
        raise ValueError("H6 figure bootstrap replicate count differs from the sealed H6 configuration")
    if not (matrix["bootstrap_replicates_valid"] > 0).all() or not (
        matrix["bootstrap_replicates_valid"] + matrix["bootstrap_replicates_invalid"]
    ).eq(BOOTSTRAP_REPLICATES).all():
        raise ValueError("H6 figure matrix bootstrap validity counts are inconsistent")
    _finite_numeric(matrix, ("bootstrap_confidence", "spearman_agreement", "spearman_ci_low", "spearman_ci_high"), kind="matrix")
    if not np.isclose(pd.to_numeric(matrix["bootstrap_confidence"], errors="raise"), CONFIDENCE, rtol=0.0, atol=1e-12).all():
        raise ValueError("H6 figure bootstrap confidence differs from the sealed H6 configuration")
    for field in ("spearman_agreement", "spearman_ci_low", "spearman_ci_high"):
        matrix[field] = pd.to_numeric(matrix[field], errors="raise")
        if not matrix[field].between(-1.0, 1.0).all():
            raise ValueError(f"H6 figure matrix.{field} must lie in [-1, 1]")
    if not ((matrix["spearman_ci_low"] <= matrix["spearman_agreement"]) & (matrix["spearman_agreement"] <= matrix["spearman_ci_high"])).all():
        raise ValueError("H6 figure matrix agreement must lie inside its percentile interval")
    if matrix["bootstrap_seed_sha256"].map(lambda value: _is_sha256(str(value))).eq(False).any():
        raise ValueError("H6 figure matrix has an invalid bootstrap seed SHA-256")
    for row in matrix.itertuples(index=False):
        seed, digest = _seed_for_cell(str(row.dataset), int(row.label), str(row.model_a), str(row.model_b))
        if int(row.bootstrap_seed) != seed or str(row.bootstrap_seed_sha256) != digest:
            raise ValueError("H6 figure matrix bootstrap seed does not match its locked cell identity")


def _expected_aggregation_row(matrix: pd.DataFrame, model_a: str, model_b: str) -> dict[str, object]:
    unit = matrix.loc[(matrix["model_a"].eq(model_a)) & (matrix["model_b"].eq(model_b))]
    values = pd.to_numeric(unit["spearman_agreement"], errors="raise").to_numpy(dtype=float)
    return {
        "expected_dataset_class_cells": len(DATASETS) * len(LABELS),
        "observed_dataset_class_cells": len(DATASETS) * len(LABELS),
        "ok_cell_count": len(DATASETS) * len(LABELS),
        "failed_cell_count": 0,
        "all_ten_cells_ok": True,
        "median_spearman_agreement": float(np.median(values)),
        "minimum_spearman_agreement": float(np.min(values)),
        "maximum_spearman_agreement": float(np.max(values)),
        "cross_cell_spearman_range": float(np.max(values) - np.min(values)),
        "cell_statuses": "ok",
    }


def _same_value(observed: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(observed), expected, rtol=1e-12, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return observed == expected


def _validate_aggregation(matrix: pd.DataFrame, aggregation: pd.DataFrame) -> None:
    if len(aggregation) != len(MODEL_PAIRS):
        raise ValueError(f"H6 figure requires exactly {len(MODEL_PAIRS)} model-pair summaries; found {len(aggregation)}")
    if aggregation.duplicated(["model_a", "model_b"], keep=False).any():
        raise ValueError("H6 figure aggregation has duplicate model-pair summaries")
    expected_pairs = set(MODEL_PAIRS)
    observed_pairs = {(str(row.model_a), str(row.model_b)) for row in aggregation.itertuples(index=False)}
    if observed_pairs != expected_pairs:
        raise ValueError("H6 figure aggregation does not exactly match the locked model-pair registry")
    aggregation["all_ten_cells_ok"] = _parse_boolean(aggregation["all_ten_cells_ok"], field="all_ten_cells_ok")
    for model_a, model_b in MODEL_PAIRS:
        row = aggregation.loc[(aggregation["model_a"].eq(model_a)) & (aggregation["model_b"].eq(model_b))]
        if len(row) != 1:  # Completeness is checked above; retain a local invariant.
            raise RuntimeError("H6 figure aggregation lookup was not unique")
        observed = row.iloc[0]
        expected = _expected_aggregation_row(matrix, model_a, model_b)
        for field, expected_value in expected.items():
            if not _same_value(observed[field], expected_value):
                raise ValueError(f"H6 figure aggregation does not match the sealed matrix for {model_a}/{model_b}: {field}")


def _load_provenance(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read H6 figure analysis provenance: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("H6 figure analysis provenance must be a JSON object")
    return value


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    matrix_path: Path,
    aggregation_path: Path,
    matrix_sha256: str,
    aggregation_sha256: str,
) -> None:
    scalar_expectations: dict[str, object] = {
        "h6_version": H6_VERSION,
        "raw_score_only": True,
        "score_values_emitted": False,
        "analysis_status": "complete",
        "stop_reason": "complete",
        "n_expected_dataset_class_model_pair_cells": len(DATASETS) * len(LABELS) * len(MODEL_PAIRS),
        "n_written_dataset_class_model_pair_cells": len(DATASETS) * len(LABELS) * len(MODEL_PAIRS),
        "complete_280_cell_matrix": True,
        "ok_cell_count": len(DATASETS) * len(LABELS) * len(MODEL_PAIRS),
        "failed_or_unavailable_cell_count": 0,
        "n_model_pair_summaries": len(MODEL_PAIRS),
    }
    for field, expected in scalar_expectations.items():
        if provenance.get(field) != expected:
            raise ValueError(f"H6 figure provenance has unexpected {field!r}; compact display refuses incomplete analysis")
    outputs = provenance.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("H6 figure provenance lacks compact output identities")
    expected_outputs = {
        "matrix": (matrix_path, matrix_sha256),
        "aggregation": (aggregation_path, aggregation_sha256),
    }
    for key, (path, digest) in expected_outputs.items():
        record = outputs.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"H6 figure provenance lacks {key} output identity")
        if record.get("absolute_path") != str(path.resolve()):
            raise ValueError(f"H6 figure provenance path mismatch for {key}")
        if record.get("sha256") != digest:
            raise ValueError(f"H6 figure provenance hash mismatch for {key}")
        if int(record.get("byte_size", -1)) != path.stat().st_size:
            raise ValueError(f"H6 figure provenance byte-size mismatch for {key}")


def _load_sealed_h6_figure_inputs(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    provenance_path: Path | str,
    *,
    contract: H6FigureInputContract,
) -> SealedH6FigureInputs:
    """Load a contract-sealed 280-cell/28-summary compact H6 result pair."""
    _validate_contract(contract)
    matrix_file = _require_exact_input_path(matrix_path, expected=contract.matrix_path, kind="matrix")
    aggregation_file = _require_exact_input_path(aggregation_path, expected=contract.aggregation_path, kind="aggregation")
    provenance_file = _require_exact_input_path(provenance_path, expected=contract.provenance_path, kind="provenance")
    if matrix_file.parent != aggregation_file.parent or matrix_file.parent != provenance_file.parent:
        raise ValueError("H6 figure compact inputs must share the one sealed analysis directory")
    matrix_sha256 = sha256_file(matrix_file)
    aggregation_sha256 = sha256_file(aggregation_file)
    provenance_sha256 = sha256_file(provenance_file)
    if matrix_sha256 != contract.matrix_sha256:
        raise ValueError("H6 figure matrix hash changed or is not the sealed H6 compact matrix")
    if aggregation_sha256 != contract.aggregation_sha256:
        raise ValueError("H6 figure aggregation hash changed or is not the sealed H6 compact aggregation")
    if provenance_sha256 != contract.provenance_sha256:
        raise ValueError("H6 figure provenance hash changed or is not the sealed H6 analysis provenance")
    matrix = _read_exact_csv(matrix_file, MATRIX_COLUMNS, kind="matrix")
    aggregation = _read_exact_csv(aggregation_file, AGGREGATION_COLUMNS, kind="aggregation")
    _validate_matrix(matrix)
    _validate_aggregation(matrix, aggregation)
    _validate_provenance(
        _load_provenance(provenance_file),
        matrix_path=matrix_file,
        aggregation_path=aggregation_file,
        matrix_sha256=matrix_sha256,
        aggregation_sha256=aggregation_sha256,
    )
    return SealedH6FigureInputs(
        matrix=matrix,
        aggregation=aggregation,
        matrix_path=matrix_file,
        aggregation_path=aggregation_file,
        provenance_path=provenance_file,
        matrix_sha256=matrix_sha256,
        aggregation_sha256=aggregation_sha256,
        provenance_sha256=provenance_sha256,
    )


def load_sealed_h6_figure_inputs(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    provenance_path: Path | str,
) -> SealedH6FigureInputs:
    """Load only the three literal, hash-sealed ``h6_analysis_001`` products."""
    return _load_sealed_h6_figure_inputs(
        matrix_path,
        aggregation_path,
        provenance_path,
        contract=SEALED_H6_FIGURE_INPUT_CONTRACT,
    )


def _model_pair_label(model_a: str, model_b: str) -> str:
    return f"{model_a} × {model_b}"


def _ordered_display_matrix(matrix: pd.DataFrame) -> np.ndarray:
    values = np.full((len(MODEL_PAIRS), len(DATASETS) * len(LABELS)), np.nan, dtype=float)
    for row_index, (model_a, model_b) in enumerate(MODEL_PAIRS):
        unit = matrix.loc[(matrix["model_a"].eq(model_a)) & (matrix["model_b"].eq(model_b))]
        for column_index, (dataset, label) in enumerate((item for item in ((dataset, label) for dataset in DATASETS for label in LABELS))):
            cell = unit.loc[(unit["dataset"].eq(dataset)) & (unit["label"].eq(label)), "spearman_agreement"]
            if len(cell) != 1:
                raise RuntimeError("H6 figure display matrix lookup was not unique")
            values[row_index, column_index] = float(cell.iloc[0])
    if not np.isfinite(values).all():
        raise RuntimeError("H6 figure display matrix unexpectedly contains a non-finite completed cell")
    return values


def _write_metadata(destination: Path, inputs: SealedH6FigureInputs) -> Path:
    pdf_path = destination / PDF_FILENAME
    png_path = destination / PNG_FILENAME
    metadata_path = destination / METADATA_FILENAME
    metadata = {
        "h6_figure_version": "h6_within_class_score_agreement_heatmap_v1",
        "input_scope": "only the literal, sealed h6_analysis_001 compact matrix, aggregation, and analysis provenance",
        "interpretation": "descriptive within-class rank agreement only; no model ranking, selection, quality, independence, cue, or causal claim",
        "matrix": {"absolute_path": str(inputs.matrix_path), "sha256": inputs.matrix_sha256, "byte_size": int(inputs.matrix_path.stat().st_size)},
        "aggregation": {"absolute_path": str(inputs.aggregation_path), "sha256": inputs.aggregation_sha256, "byte_size": int(inputs.aggregation_path.stat().st_size)},
        "analysis_provenance": {"absolute_path": str(inputs.provenance_path), "sha256": inputs.provenance_sha256, "byte_size": int(inputs.provenance_path.stat().st_size)},
        "fixed_model_pair_order": [list(pair) for pair in MODEL_PAIRS],
        "fixed_column_order": [[dataset, label] for dataset in DATASETS for label in LABELS],
        "column_labels": [f"{DATASET_DISPLAY_LABELS[dataset]} / {LABEL_DISPLAY_LABELS[label]}" for dataset in DATASETS for label in LABELS],
        "color_map": "h6_blue_neutral_orange",
        "color_scale": {"minimum": -1.0, "maximum": 1.0, "fixed": True},
        "cell_annotations": "none; all 280 values remain in the sealed compact matrix",
        "sorting_or_selection": "none; rows and columns use only the registered fixed orders",
        "outputs": {
            "pdf": {"absolute_path": str(pdf_path.resolve()), "sha256": sha256_file(pdf_path), "byte_size": int(pdf_path.stat().st_size)},
            "png": {"absolute_path": str(png_path.resolve()), "sha256": sha256_file(png_path), "byte_size": int(png_path.stat().st_size), "dpi": 300},
        },
    }
    metadata_path.write_text(canonical_json(metadata), encoding="utf-8")
    return metadata_path


def _render_h6_within_class_score_agreement_heatmap(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    provenance_path: Path | str,
    output_dir: Path | str,
    *,
    contract: H6FigureInputContract,
) -> tuple[Path, Path, Path]:
    """Write a fixed-order, fixed-scale, descriptive H6 supplementary heatmap."""
    inputs = _load_sealed_h6_figure_inputs(
        matrix_path,
        aggregation_path,
        provenance_path,
        contract=contract,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("H6 figure output directory is non-overwritable; choose a new directory")
    destination.mkdir(parents=True, exist_ok=False)
    pdf_path = destination / PDF_FILENAME
    png_path = destination / PNG_FILENAME

    # Okabe--Ito blue and vermillion around a neutral centre retain a
    # colorblind-safe signed reading.  The [-1, 1] range is never rescaled.
    cmap = LinearSegmentedColormap.from_list(
        "h6_blue_neutral_orange", ["#0072B2", "#F7F7F7", "#D55E00"], N=256
    )
    norm = Normalize(vmin=-1.0, vmax=1.0, clip=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    display = _ordered_display_matrix(inputs.matrix)
    fig, axis = plt.subplots(figsize=(11.8, 10.2), constrained_layout=False)
    mappable = axis.pcolormesh(
        np.arange(display.shape[1] + 1),
        np.arange(display.shape[0] + 1),
        display,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.38,
        shading="flat",
    )
    axis.set_xlim(0, display.shape[1])
    axis.set_ylim(display.shape[0], 0)
    axis.set_xticks(np.arange(display.shape[1]) + 0.5)
    axis.set_xticklabels(
        [f"{DATASET_DISPLAY_LABELS[dataset]}\n{LABEL_DISPLAY_LABELS[label]}" for dataset in DATASETS for label in LABELS],
        rotation=35,
        ha="right",
        fontsize=7.1,
    )
    axis.set_yticks(np.arange(display.shape[0]) + 0.5)
    axis.set_yticklabels([_model_pair_label(*pair) for pair in MODEL_PAIRS], fontsize=6.6)
    axis.tick_params(axis="both", length=0)
    for boundary in range(2, display.shape[1], 2):
        axis.axvline(boundary, color="#4D4D4D", linewidth=0.82, zorder=4)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel("Registered corpus × class cells (fixed order; no sorting or selection)", labelpad=10)
    axis.set_ylabel("Registered unordered model pairs (fixed order; no ranking)", labelpad=10)
    fig.suptitle("H6 within-class published-score rank agreement atlas (descriptive display)", y=0.985, fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.285, right=0.985, bottom=0.215, top=0.93)
    colorbar_axis = fig.add_axes([0.39, 0.087, 0.42, 0.017])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("Within-class Spearman rank agreement (fixed [−1, 1])", labelpad=2)
    fig.text(
        0.64,
        0.025,
        "Descriptive agreement only; not model quality, independence, cue reliance, or causal sensitivity.",
        ha="center",
        va="center",
        fontsize=7.25,
    )
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    metadata_path = _write_metadata(destination, inputs)
    return pdf_path, png_path, metadata_path


def render_h6_within_class_score_agreement_heatmap(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    provenance_path: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path, Path]:
    """Render only the literal, sealed H6 display contract once it is committed."""
    return _render_h6_within_class_score_agreement_heatmap(
        matrix_path,
        aggregation_path,
        provenance_path,
        output_dir,
        contract=SEALED_H6_FIGURE_INPUT_CONTRACT,
    )
