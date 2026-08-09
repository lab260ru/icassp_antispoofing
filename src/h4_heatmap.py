"""Publication-style, score-free plotting for completed H4 atlas artifacts.

The plotting path accepts only the two compact H4 CSV outputs.  It neither
opens the feature inputs nor imports audio, ASR, detector, model, or scoring
code.  The heatmap is therefore a pure visualization of a completed H4 result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# A non-interactive backend keeps headless research nodes reproducible.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


H4_DATASETS = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
H4_VIEWS = ("full_waveform", "deterministic_crop", "preemphasized_crop")
VIEW_LABELS = {
    "full_waveform": "Full waveform",
    "deterministic_crop": "Deterministic crop",
    "preemphasized_crop": "Pre-emphasized crop",
}
MATRIX_FILENAME = "h4_label_cue_matrix.csv"
AGGREGATION_FILENAME = "h4_label_cue_aggregation.csv"
PDF_FILENAME = "h4_label_cue_delta_auc_heatmap.pdf"
PNG_FILENAME = "h4_label_cue_delta_auc_heatmap.png"
FORBIDDEN_TERMS = ("score", "logit", "detector", "model", "eer", "audio", "arena")
MATRIX_COLUMNS = ("dataset", "view", "feature", "cell_status", "delta_auc")
AGGREGATION_COLUMNS = ("view", "feature", "atlas_only_stable_label_association")


def _forbidden(value: str) -> bool:
    lowered = value.casefold()
    return any(term in lowered for term in FORBIDDEN_TERMS)


def _safe_artifact_path(path: Path | str, expected_name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"H4 heatmap input must be an absolute path: {candidate}")
    if candidate.name != expected_name:
        raise ValueError(f"H4 heatmap requires {expected_name}, not {candidate.name}")
    if _forbidden(str(candidate)):
        raise ValueError(f"H4 heatmap path firewall rejected: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"H4 heatmap input is not a file: {candidate}")
    return candidate


def _safe_csv_projection(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Reject response-like headers and project only plotting columns."""
    header = pd.read_csv(path, nrows=0)
    columns = header.columns.tolist()
    forbidden = [column for column in columns if _forbidden(column)]
    if forbidden:
        raise ValueError(f"H4 heatmap rejected response-like CSV columns: {forbidden}")
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"H4 heatmap CSV missing required columns: {missing}")
    return pd.read_csv(path, usecols=list(required))


def _parse_boolean(values: pd.Series, *, field: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalised = values.astype("string").str.strip().str.casefold()
    valid = normalised.isin(["true", "false"])
    if not valid.all():
        invalid = values.loc[~valid].head(5).tolist()
        raise ValueError(f"H4 heatmap {field} must contain only true/false values; saw {invalid}")
    return normalised.eq("true")


def load_completed_h4_artifacts(
    matrix_path: Path | str,
    aggregation_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only a complete, finite 420-cell/84-unit H4 result pair.

    The check deliberately refuses partially completed or failed-cell matrices:
    a heatmap must never turn absent H4 cells into visually plausible values.
    """
    matrix_file = _safe_artifact_path(matrix_path, MATRIX_FILENAME)
    aggregation_file = _safe_artifact_path(aggregation_path, AGGREGATION_FILENAME)
    matrix = _safe_csv_projection(matrix_file, MATRIX_COLUMNS)
    aggregation = _safe_csv_projection(aggregation_file, AGGREGATION_COLUMNS)
    if len(matrix) != 420:
        raise ValueError(f"H4 heatmap requires exactly 420 matrix rows; found {len(matrix)}")
    if matrix.duplicated(["dataset", "view", "feature"]).any():
        raise ValueError("H4 heatmap matrix has duplicate dataset/view/feature cells")
    if set(matrix["dataset"].astype(str)) != set(H4_DATASETS):
        raise ValueError("H4 heatmap matrix does not contain the locked five datasets")
    if set(matrix["view"].astype(str)) != set(H4_VIEWS):
        raise ValueError("H4 heatmap matrix does not contain the locked three views")
    if set(matrix["cell_status"].astype(str)) != {"ok"}:
        raise ValueError("H4 heatmap requires 420 completed ('ok') matrix cells")
    matrix["delta_auc"] = pd.to_numeric(matrix["delta_auc"], errors="coerce")
    if not np.isfinite(matrix["delta_auc"].to_numpy(dtype=float)).all():
        raise ValueError("H4 heatmap requires finite delta_auc in every matrix cell")
    if not matrix["delta_auc"].between(-1.0, 1.0).all():
        raise ValueError("H4 heatmap delta_auc must lie in [-1, 1]")

    features_by_view: dict[str, frozenset[str]] = {}
    for view in H4_VIEWS:
        view_rows = matrix.loc[matrix["view"].astype(str).eq(view)]
        if len(view_rows) != len(H4_DATASETS) * 28:
            raise ValueError(f"H4 heatmap matrix has incomplete rows for {view}")
        feature_set = frozenset(view_rows["feature"].astype(str))
        if len(feature_set) != 28:
            raise ValueError(f"H4 heatmap matrix must have exactly 28 features for {view}")
        for dataset in H4_DATASETS:
            dataset_rows = view_rows.loc[view_rows["dataset"].astype(str).eq(dataset)]
            if frozenset(dataset_rows["feature"].astype(str)) != feature_set:
                raise ValueError(f"H4 heatmap matrix has an incomplete feature grid for {dataset}/{view}")
        features_by_view[view] = feature_set
    if len(set(features_by_view.values())) != 1:
        raise ValueError("H4 heatmap requires the same 28 registered features in every view")

    if len(aggregation) != 84:
        raise ValueError(f"H4 heatmap requires exactly 84 aggregation rows; found {len(aggregation)}")
    if aggregation.duplicated(["view", "feature"]).any():
        raise ValueError("H4 heatmap aggregation has duplicate view/feature units")
    if set(aggregation["view"].astype(str)) != set(H4_VIEWS):
        raise ValueError("H4 heatmap aggregation does not contain the locked three views")
    aggregation["atlas_only_stable_label_association"] = _parse_boolean(
        aggregation["atlas_only_stable_label_association"], field="atlas_only_stable_label_association"
    )
    expected_units = {(view, feature) for view in H4_VIEWS for feature in features_by_view[view]}
    observed_units = set(zip(aggregation["view"].astype(str), aggregation["feature"].astype(str), strict=True))
    if observed_units != expected_units:
        raise ValueError("H4 heatmap aggregation units do not exactly match the completed matrix")
    return matrix, aggregation


def _feature_label(feature: str) -> str:
    """Keep locked registry names readable without changing their identity."""
    return feature.replace("_", " ")


def render_h4_delta_auc_heatmap(
    matrix_path: Path | str,
    aggregation_path: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Export a vector PDF and 300-DPI PNG from completed H4 results only."""
    matrix, aggregation = load_completed_h4_artifacts(matrix_path, aggregation_path)
    destination = Path(output_dir)
    pdf_path = destination / PDF_FILENAME
    png_path = destination / PNG_FILENAME
    if pdf_path.exists() or png_path.exists():
        raise FileExistsError("H4 heatmap outputs are non-overwritable; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)

    # Blue--neutral--orange is a colourblind-aware diverging treatment.  The
    # symmetric fixed delta-AUC range preserves visual comparability across
    # every view and corpus rather than scaling one subgroup opportunistically.
    cmap = LinearSegmentedColormap.from_list(
        "h4_blue_neutral_orange", ["#0072B2", "#F7F7F7", "#D55E00"], N=256
    )
    norm = Normalize(vmin=-1.0, vmax=1.0)
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
    fig, axes = plt.subplots(1, len(H4_VIEWS), figsize=(11.4, 8.4), sharey=False, constrained_layout=False)
    if not isinstance(axes, np.ndarray):  # pragma: no cover - fixed three-panel layout
        axes = np.asarray([axes])
    mappable = None
    stable_marker = Line2D(
        [0],
        [0],
        marker="s",
        linestyle="None",
        markersize=7,
        markerfacecolor="#009E73",
        markeredgecolor="#009E73",
        label="Atlas-only stable association (descriptive; not a detector candidate)",
    )
    for axis, view in zip(axes, H4_VIEWS, strict=True):
        features = sorted(matrix.loc[matrix["view"].astype(str).eq(view), "feature"].astype(str).unique().tolist())
        data = (
            matrix.loc[matrix["view"].astype(str).eq(view), ["dataset", "feature", "delta_auc"]]
            .pivot(index="feature", columns="dataset", values="delta_auc")
            .reindex(index=features, columns=H4_DATASETS)
        )
        mappable = axis.pcolormesh(
            np.arange(len(H4_DATASETS) + 1),
            np.arange(len(features) + 1),
            data.to_numpy(dtype=float),
            cmap=cmap,
            norm=norm,
            edgecolors="white",
            linewidth=0.35,
            shading="flat",
        )
        stable_units = aggregation.loc[
            aggregation["view"].astype(str).eq(view), ["feature", "atlas_only_stable_label_association"]
        ].set_index("feature")["atlas_only_stable_label_association"]
        for row, feature in enumerate(features):
            if bool(stable_units.at[feature]):
                axis.plot(-0.38, row + 0.5, marker="s", markersize=4.6, color="#009E73", clip_on=False)
        axis.set_xlim(-0.62, len(H4_DATASETS))
        axis.set_ylim(len(features), 0)
        axis.set_xticks(np.arange(len(H4_DATASETS)) + 0.5)
        axis.set_xticklabels(H4_DATASETS, rotation=35, ha="right", fontsize=7)
        axis.set_yticks(np.arange(len(features)) + 0.5)
        axis.set_yticklabels([_feature_label(feature) for feature in features], fontsize=6.5)
        axis.tick_params(axis="both", length=0)
        axis.set_title(VIEW_LABELS[view], pad=7)
        for spine in axis.spines.values():
            spine.set_visible(False)
    assert mappable is not None
    # Reserve a dedicated band below the tilted corpus labels. Letting the
    # automatic shared colorbar compete for this space makes the figure hard
    # to read in a two-column PDF.
    fig.subplots_adjust(left=0.18, right=0.995, bottom=0.24, top=0.895, wspace=0.42)
    colorbar_axis = fig.add_axes([0.27, 0.105, 0.50, 0.018])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"Signed label separation, $\Delta$AUC = 2·AUROC − 1 (raw feature orientation)", labelpad=2)
    fig.legend(handles=[stable_marker], loc="lower center", bbox_to_anchor=(0.5, 0.012), frameon=False, fontsize=8)
    fig.suptitle("H4 cross-corpus label–cue atlas (descriptive only)", y=0.996, fontsize=11, fontweight="bold")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path
