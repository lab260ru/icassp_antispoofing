"""Render a sealed H5 supplementary concordance heatmap and nothing else.

This module accepts only the compact matrix and aggregation from the literal
``h5_analysis_001`` run, after their sibling provenance has revalidated their
paths, sizes, and SHA-256 values.  It does not open a feature container,
waveform, label, score artifact, detector/model/ASR resource, or earlier study
result.  The plotting values are precomputed H5 aggregation quantities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Keep figures reproducible on a headless research node.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.audio_features import FEATURE_NAMES
from src.h5_view_invariance import DATASETS, H5_VERSION, VIEW_PAIRS


H5_ANALYSIS_ROOT = Path(
    "/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001"
)
MATRIX_FILENAME = "h5_view_invariance_matrix.csv"
AGGREGATION_FILENAME = "h5_view_invariance_aggregation.csv"
PROVENANCE_FILENAME = "h5_analysis_provenance.json"
PDF_FILENAME = "h5_view_invariance_median_concordance_heatmap.pdf"
PNG_FILENAME = "h5_view_invariance_median_concordance_heatmap.png"
METADATA_FILENAME = "h5_view_invariance_median_concordance_heatmap.metadata.json"

H5_FIGURE_ALLOWED_INPUTS: dict[str, Path] = {
    "matrix": H5_ANALYSIS_ROOT / MATRIX_FILENAME,
    "aggregation": H5_ANALYSIS_ROOT / AGGREGATION_FILENAME,
}

# The display boundary is intentionally stricter than the H5 source table:
# none of these identities may enter a compact figure input/header/path.
FORBIDDEN_TERMS = (
    "label",
    "source",
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
MATRIX_COLUMNS = (
    "dataset",
    "view_a",
    "view_b",
    "feature",
    "cell_status",
    "paired_finite_sample_count",
    "spearman_concordance",
    "iqr_normalized_median_shift",
    "bootstrap_replicates_requested",
    "bootstrap_replicates_valid_concordance",
    "bootstrap_replicates_valid_shift",
    "bootstrap_replicates_invalid_concordance",
    "bootstrap_replicates_invalid_shift",
    "bootstrap_confidence",
    "concordance_ci_low",
    "concordance_ci_high",
    "normalized_shift_ci_low",
    "normalized_shift_ci_high",
    "bootstrap_seed",
    "bootstrap_seed_sha256",
)
AGGREGATION_COLUMNS = (
    "view_a",
    "view_b",
    "feature",
    "dataset_cell_count",
    "successful_dataset_cell_count",
    "missing_datasets",
    "all_dataset_cells_ok",
    "median_spearman_concordance",
    "median_absolute_iqr_normalized_shift",
    "all_concordance_at_least_0p90",
    "all_concordance_ci_lower_at_least_0p80",
    "all_shift_intervals_within_plus_minus_0p10",
    "view_stable_terminal_descriptive",
    "aggregation_status",
)
PAIR_LABELS = {
    ("deterministic_crop", "full_waveform"): "Deterministic crop /\nfull waveform",
    ("deterministic_crop", "preemphasized_crop"): "Deterministic crop /\npre-emphasized crop",
    ("full_waveform", "preemphasized_crop"): "Full waveform /\npre-emphasized crop",
}


@dataclass(frozen=True)
class SealedH5FigureInputs:
    """Validated compact dataframes and their hash-bound input identities."""

    matrix: pd.DataFrame
    aggregation: pd.DataFrame
    matrix_path: Path
    aggregation_path: Path
    matrix_sha256: str
    aggregation_sha256: str
    analysis_provenance_path: Path
    analysis_provenance_sha256: str


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden(value: str) -> bool:
    lowered = value.casefold()
    return any(term in lowered for term in FORBIDDEN_TERMS)


def _parse_boolean(values: pd.Series, *, field: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalised = values.astype("string").str.strip().str.casefold()
    if not normalised.isin(["true", "false"]).all():
        invalid = values.loc[~normalised.isin(["true", "false"])].head(5).tolist()
        raise ValueError(f"H5 figure {field} must contain only true/false values; saw {invalid}")
    return normalised.eq("true")


def _safe_exact_path(path: Path | str, *, key: str, allowed_paths: Mapping[str, Path | str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"H5 figure {key} input must be an absolute path: {candidate}")
    if _forbidden(str(candidate)):
        raise ValueError(f"H5 figure path firewall rejected: {candidate}")
    if key not in allowed_paths:
        raise ValueError(f"H5 figure missing locked allowed path for {key}")
    allowed = Path(allowed_paths[key])
    if not allowed.is_absolute() or _forbidden(str(allowed)):
        raise ValueError(f"H5 figure locked path is invalid for {key}")
    if candidate != allowed or candidate.resolve() != allowed.resolve():
        raise ValueError(f"H5 figure requires the literal h5_analysis_001 {key} path")
    if not candidate.is_file():
        raise FileNotFoundError(f"H5 figure input is not a regular file: {candidate}")
    return candidate


def _load_provenance(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read H5 figure analysis provenance: {path}") from error
    if not isinstance(data, dict):
        raise ValueError("H5 figure analysis provenance must be a JSON object")
    return data


def _read_exact_csv(path: Path, expected_columns: tuple[str, ...], *, kind: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    forbidden = [column for column in header if _forbidden(column)]
    if forbidden:
        raise ValueError(f"H5 figure rejected response-like {kind} CSV columns: {forbidden}")
    if tuple(header) != expected_columns:
        raise ValueError(f"H5 figure {kind} CSV schema does not exactly match the sealed H5 schema")
    return pd.read_csv(path, usecols=list(expected_columns))


def _validate_provenance_artifact(
    provenance: Mapping[str, Any],
    *,
    key: str,
    path: Path,
) -> str:
    record = provenance.get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"H5 figure provenance lacks {key} identity")
    if record.get("absolute_path") != str(path.resolve()):
        raise ValueError(f"H5 figure provenance path mismatch for {key}")
    if int(record.get("byte_size", -1)) != path.stat().st_size:
        raise ValueError(f"H5 figure provenance byte-size mismatch for {key}")
    current_hash = sha256_file(path)
    if record.get("sha256") != current_hash:
        raise ValueError(f"H5 figure input hash changed for {key}")
    return current_hash


def _validate_matrix(matrix: pd.DataFrame) -> None:
    expected_count = len(DATASETS) * len(VIEW_PAIRS) * len(FEATURE_NAMES)
    if len(matrix) != expected_count:
        raise ValueError(f"H5 figure requires exactly {expected_count} matrix rows; found {len(matrix)}")
    keys = ["dataset", "view_a", "view_b", "feature"]
    if matrix.duplicated(keys).any():
        raise ValueError("H5 figure matrix has duplicate dataset/view-pair/feature cells")
    if set(matrix["dataset"].astype(str)) != set(DATASETS):
        raise ValueError("H5 figure matrix does not contain exactly the locked five datasets")
    observed_pairs = set(zip(matrix["view_a"].astype(str), matrix["view_b"].astype(str), strict=True))
    if observed_pairs != set(VIEW_PAIRS):
        raise ValueError("H5 figure matrix does not contain exactly the locked view pairs")
    if set(matrix["feature"].astype(str)) != set(FEATURE_NAMES):
        raise ValueError("H5 figure matrix does not contain exactly the locked 28 features")
    statuses = matrix["cell_status"].astype(str)
    if not statuses.map(lambda value: value == "ok" or value.startswith("failed_")).all():
        raise ValueError("H5 figure matrix has an unsupported cell status")
    for view_a, view_b in VIEW_PAIRS:
        pair_rows = matrix.loc[(matrix["view_a"] == view_a) & (matrix["view_b"] == view_b)]
        if len(pair_rows) != len(DATASETS) * len(FEATURE_NAMES):
            raise ValueError(f"H5 figure matrix has an incomplete grid for {view_a}/{view_b}")
        for dataset in DATASETS:
            features = frozenset(pair_rows.loc[pair_rows["dataset"] == dataset, "feature"].astype(str))
            if features != frozenset(FEATURE_NAMES):
                raise ValueError(f"H5 figure matrix has an incomplete feature grid for {dataset}/{view_a}/{view_b}")
    ok = matrix["cell_status"].astype(str).eq("ok")
    concordance = pd.to_numeric(matrix.loc[ok, "spearman_concordance"], errors="coerce")
    if not np.isfinite(concordance.to_numpy(dtype=float)).all() or not concordance.between(-1.0, 1.0).all():
        raise ValueError("H5 figure successful matrix cells require finite Spearman values in [-1, 1]")


def _expected_aggregation_row(matrix: pd.DataFrame, view_a: str, view_b: str, feature: str) -> dict[str, object]:
    unit = matrix.loc[
        (matrix["view_a"] == view_a) & (matrix["view_b"] == view_b) & (matrix["feature"] == feature)
    ].set_index("dataset")
    successful = unit.loc[unit["cell_status"].astype(str).eq("ok")]
    all_ok = len(successful) == len(DATASETS)
    concordance = pd.to_numeric(successful["spearman_concordance"], errors="coerce").to_numpy(dtype=float)
    shifts = pd.to_numeric(successful["iqr_normalized_median_shift"], errors="coerce").to_numpy(dtype=float)
    ci_lows = pd.to_numeric(successful["concordance_ci_low"], errors="coerce").to_numpy(dtype=float)
    shift_ci_lows = pd.to_numeric(successful["normalized_shift_ci_low"], errors="coerce").to_numpy(dtype=float)
    shift_ci_highs = pd.to_numeric(successful["normalized_shift_ci_high"], errors="coerce").to_numpy(dtype=float)
    stable = bool(
        all_ok
        and np.isfinite(concordance).all()
        and np.isfinite(ci_lows).all()
        and np.isfinite(shifts).all()
        and np.isfinite(shift_ci_lows).all()
        and np.isfinite(shift_ci_highs).all()
        and (concordance >= 0.90).all()
        and (ci_lows >= 0.80).all()
        and (shift_ci_lows >= -0.10).all()
        and (shift_ci_highs <= 0.10).all()
    )
    return {
        "dataset_cell_count": int(len(unit)),
        "successful_dataset_cell_count": int(len(successful)),
        "all_dataset_cells_ok": all_ok,
        "median_spearman_concordance": float(np.median(concordance)) if len(concordance) else float("nan"),
        "median_absolute_iqr_normalized_shift": float(np.median(np.abs(shifts))) if len(shifts) else float("nan"),
        "all_concordance_at_least_0p90": bool(all_ok and np.isfinite(concordance).all() and (concordance >= 0.90).all()),
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


def _values_match(observed: object, expected: object, *, field: str) -> bool:
    if isinstance(expected, float):
        try:
            value = float(observed)
        except (TypeError, ValueError):
            return False
        return bool(np.isnan(expected) and np.isnan(value)) or bool(np.isclose(value, expected, rtol=1e-12, atol=1e-12))
    return observed == expected


def _validate_aggregation(matrix: pd.DataFrame, aggregation: pd.DataFrame) -> None:
    expected_count = len(VIEW_PAIRS) * len(FEATURE_NAMES)
    if len(aggregation) != expected_count:
        raise ValueError(f"H5 figure requires exactly {expected_count} aggregation rows; found {len(aggregation)}")
    keys = ["view_a", "view_b", "feature"]
    if aggregation.duplicated(keys).any():
        raise ValueError("H5 figure aggregation has duplicate view-pair/feature units")
    observed_units = set(zip(aggregation["view_a"].astype(str), aggregation["view_b"].astype(str), aggregation["feature"].astype(str), strict=True))
    expected_units = {(view_a, view_b, feature) for view_a, view_b in VIEW_PAIRS for feature in FEATURE_NAMES}
    if observed_units != expected_units:
        raise ValueError("H5 figure aggregation does not exactly match the sealed H5 units")
    boolean_fields = (
        "all_dataset_cells_ok",
        "all_concordance_at_least_0p90",
        "all_concordance_ci_lower_at_least_0p80",
        "all_shift_intervals_within_plus_minus_0p10",
        "view_stable_terminal_descriptive",
    )
    for field in boolean_fields:
        aggregation[field] = _parse_boolean(aggregation[field], field=field)
    for view_a, view_b in VIEW_PAIRS:
        for feature in FEATURE_NAMES:
            row = aggregation.loc[
                (aggregation["view_a"] == view_a) & (aggregation["view_b"] == view_b) & (aggregation["feature"] == feature)
            ]
            if len(row) != 1:  # guarded above; preserves a local invariant
                raise RuntimeError("H5 figure aggregation lookup was not unique")
            observed = row.iloc[0]
            expected = _expected_aggregation_row(matrix, view_a, view_b, feature)
            for field, expected_value in expected.items():
                if not _values_match(observed[field], expected_value, field=field):
                    raise ValueError(f"H5 figure aggregation does not match the sealed matrix for {view_a}/{view_b}/{feature}: {field}")


def _load_sealed_h5_figure_inputs(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str] = H5_FIGURE_ALLOWED_INPUTS,
) -> SealedH5FigureInputs:
    """Testable core for exact, provenance-hash-validated compact H5 inputs."""
    if set(allowed_paths) != {"matrix", "aggregation"}:
        raise ValueError("H5 figure allowed paths must contain exactly matrix and aggregation")
    matrix_file = _safe_exact_path(matrix_path, key="matrix", allowed_paths=allowed_paths)
    aggregation_file = _safe_exact_path(aggregation_path, key="aggregation", allowed_paths=allowed_paths)
    if matrix_file.name != MATRIX_FILENAME or aggregation_file.name != AGGREGATION_FILENAME:
        raise ValueError("H5 figure input filenames do not match the sealed H5 run")
    if matrix_file.parent != aggregation_file.parent:
        raise ValueError("H5 figure matrix and aggregation must share one h5_analysis_001 directory")
    provenance_file = matrix_file.parent / PROVENANCE_FILENAME
    if not provenance_file.is_file() or _forbidden(str(provenance_file)):
        raise ValueError("H5 figure requires a safe sibling h5_analysis_provenance.json")
    provenance = _load_provenance(provenance_file)
    expected_cells = len(DATASETS) * len(VIEW_PAIRS) * len(FEATURE_NAMES)
    expected_units = len(VIEW_PAIRS) * len(FEATURE_NAMES)
    if provenance.get("h5_version") != H5_VERSION or provenance.get("matrix_complete") is not True:
        raise ValueError("H5 figure provenance does not describe a complete H5 matrix")
    if provenance.get("n_dataset_view_pair_feature_cells") != expected_cells or provenance.get("n_view_pair_feature_aggregation_units") != expected_units:
        raise ValueError("H5 figure provenance matrix/aggregation counts do not match the locked H5 atlas")
    matrix_hash = _validate_provenance_artifact(provenance, key="matrix", path=matrix_file)
    aggregation_hash = _validate_provenance_artifact(provenance, key="aggregation", path=aggregation_file)
    matrix = _read_exact_csv(matrix_file, MATRIX_COLUMNS, kind="matrix")
    aggregation = _read_exact_csv(aggregation_file, AGGREGATION_COLUMNS, kind="aggregation")
    _validate_matrix(matrix)
    _validate_aggregation(matrix, aggregation)
    return SealedH5FigureInputs(
        matrix=matrix,
        aggregation=aggregation,
        matrix_path=matrix_file,
        aggregation_path=aggregation_file,
        matrix_sha256=matrix_hash,
        aggregation_sha256=aggregation_hash,
        analysis_provenance_path=provenance_file,
        analysis_provenance_sha256=sha256_file(provenance_file),
    )


def load_sealed_h5_figure_inputs(
    matrix_path: Path | str,
    aggregation_path: Path | str,
) -> SealedH5FigureInputs:
    """Load only the two literal, sealed ``h5_analysis_001`` artifacts."""
    return _load_sealed_h5_figure_inputs(
        matrix_path,
        aggregation_path,
        allowed_paths=H5_FIGURE_ALLOWED_INPUTS,
    )


def _feature_label(feature: str) -> str:
    return feature.replace("_", " ")


def _write_metadata(
    destination: Path,
    inputs: SealedH5FigureInputs,
    *,
    unavailable_aggregation_units: int,
    terminal_descriptive_units: int,
) -> Path:
    pdf_path = destination / PDF_FILENAME
    png_path = destination / PNG_FILENAME
    metadata_path = destination / METADATA_FILENAME
    metadata = {
        "h5_figure_version": "h5_median_concordance_heatmap_v1",
        "input_scope": "h5_analysis_001 compact matrix and aggregation only",
        "matrix": {"absolute_path": str(inputs.matrix_path), "sha256": inputs.matrix_sha256, "byte_size": int(inputs.matrix_path.stat().st_size)},
        "aggregation": {"absolute_path": str(inputs.aggregation_path), "sha256": inputs.aggregation_sha256, "byte_size": int(inputs.aggregation_path.stat().st_size)},
        "analysis_provenance": {"absolute_path": str(inputs.analysis_provenance_path), "sha256": inputs.analysis_provenance_sha256},
        "fixed_view_pairs_lexicographic": [list(pair) for pair in VIEW_PAIRS],
        "feature_order": list(FEATURE_NAMES),
        "color_map": "viridis",
        "color_scale": {"minimum": 0.0, "maximum": 1.0, "negative_underflow": "shown at lower scale endpoint"},
        "unavailable_marker": "light-gray cell with black x; no imputation",
        "terminal_descriptive_marker": "green square; not a ranking/selection signal",
        "display_counts": {
            "unavailable_aggregation_units": unavailable_aggregation_units,
            "terminal_descriptive_units": terminal_descriptive_units,
        },
        "outputs": {
            "pdf": {"absolute_path": str(pdf_path.resolve()), "sha256": sha256_file(pdf_path), "byte_size": int(pdf_path.stat().st_size)},
            "png": {"absolute_path": str(png_path.resolve()), "sha256": sha256_file(png_path), "byte_size": int(png_path.stat().st_size), "dpi": 300},
        },
    }
    metadata_path.write_text(canonical_json(metadata), encoding="utf-8")
    return metadata_path


def _render_h5_median_concordance_heatmap(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    output_dir: Path | str,
    *,
    allowed_paths: Mapping[str, Path | str],
) -> tuple[Path, Path, Path]:
    """Export a sealed three-panel H5 supplementary figure.

    Each panel has one five-corpus-median cell per registered feature.  A
    failed/unavailable underlying dataset cell yields an explicit unavailable
    aggregation cell, not a partial value that could resemble a complete unit.
    """
    inputs = _load_sealed_h5_figure_inputs(matrix_path, aggregation_path, allowed_paths=allowed_paths)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("H5 figure output directory is non-overwritable; choose a new directory")
    destination.mkdir(parents=True, exist_ok=False)
    pdf_path = destination / PDF_FILENAME
    png_path = destination / PNG_FILENAME

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#D9D9D9")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    fig, axes = plt.subplots(1, len(VIEW_PAIRS), figsize=(8.8, 8.4), sharey=True, constrained_layout=False)
    if not isinstance(axes, np.ndarray):  # fixed three-panel layout
        axes = np.asarray([axes])
    mappable = None
    unavailable_marker = Line2D(
        [0], [0], marker="x", linestyle="None", markersize=6, markeredgewidth=1.2, color="#202020", label="Unavailable unit (not imputed)"
    )
    terminal_marker = Line2D(
        [0], [0], marker="s", linestyle="None", markersize=5.5, color="#009E73", label="Terminal descriptive view stability (not a ranking/selection signal)"
    )
    unavailable_count = 0
    terminal_count = 0
    for axis, (view_a, view_b) in zip(axes, VIEW_PAIRS, strict=True):
        rows = inputs.aggregation.loc[
            (inputs.aggregation["view_a"] == view_a) & (inputs.aggregation["view_b"] == view_b)
        ].set_index("feature").reindex(FEATURE_NAMES)
        available = rows["all_dataset_cells_ok"].to_numpy(dtype=bool) & np.isfinite(
            pd.to_numeric(rows["median_spearman_concordance"], errors="coerce").to_numpy(dtype=float)
        )
        values = pd.to_numeric(rows["median_spearman_concordance"], errors="coerce").to_numpy(dtype=float)
        displayed = np.ma.masked_where(~available, values).reshape(len(FEATURE_NAMES), 1)
        mappable = axis.pcolormesh(
            np.arange(2), np.arange(len(FEATURE_NAMES) + 1), displayed, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.45, shading="flat"
        )
        for row_index, (is_available, stable) in enumerate(zip(available, rows["view_stable_terminal_descriptive"].to_numpy(dtype=bool), strict=True)):
            if not is_available:
                axis.plot(0.5, row_index + 0.5, marker="x", markersize=5.5, markeredgewidth=1.1, color="#202020")
                unavailable_count += 1
            if stable:
                axis.plot(-0.16, row_index + 0.5, marker="s", markersize=4.8, color="#009E73", clip_on=False)
                terminal_count += 1
        axis.set_xlim(-0.29, 1.0)
        axis.set_ylim(len(FEATURE_NAMES), 0)
        axis.set_xticks([0.5])
        axis.set_xticklabels(["Five-corpus\nmedian"], fontsize=7)
        axis.set_yticks(np.arange(len(FEATURE_NAMES)) + 0.5)
        axis.set_yticklabels([_feature_label(feature) for feature in FEATURE_NAMES], fontsize=6.5)
        axis.tick_params(axis="both", length=0)
        axis.set_title(PAIR_LABELS[(view_a, view_b)], pad=7)
        for spine in axis.spines.values():
            spine.set_visible(False)
    assert mappable is not None
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.20, top=0.90, wspace=0.82)
    colorbar_axis = fig.add_axes([0.34, 0.112, 0.40, 0.017])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_label("Five-corpus median Spearman concordance (fixed [0, 1])", labelpad=2)
    fig.legend(handles=[unavailable_marker, terminal_marker], loc="lower center", bbox_to_anchor=(0.56, 0.005), frameon=False, fontsize=7.2, ncol=1)
    fig.suptitle("H5 paired waveform-view invariance atlas (terminal descriptive display)", y=0.988, fontsize=10.5, fontweight="bold")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    metadata_path = _write_metadata(
        destination,
        inputs,
        unavailable_aggregation_units=unavailable_count,
        terminal_descriptive_units=terminal_count,
    )
    return pdf_path, png_path, metadata_path


def render_h5_median_concordance_heatmap(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path, Path]:
    """Render only the literal, sealed ``h5_analysis_001`` supplementary figure."""
    return _render_h5_median_concordance_heatmap(
        matrix_path,
        aggregation_path,
        output_dir,
        allowed_paths=H5_FIGURE_ALLOWED_INPUTS,
    )
