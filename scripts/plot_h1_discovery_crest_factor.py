#!/usr/bin/env python3
"""Render a fixed, discovery-only H1 crest-factor comparison figure.

Only the three CSV paths named on the command line are read. The requested
slice is fixed before input inspection: full-waveform, spoof class,
``crest_factor_db``, and partial Spearman score association for every model in
the configured score panel. This is a descriptive discovery visualization; it
does not select/freeze candidates or evaluate confirmation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


EXPECTED_DISCOVERY = ("ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF")
KEY_COLUMNS = ["dataset", "model", "view", "feature", "class_label"]
FOCUS = {
    "view": "full_waveform",
    "feature": "crest_factor_db",
    "class_label": 1,
    "analysis_stage": "screen",
}
RHO_COLUMN = "partial_spearman_rho"
Q_COLUMN = "partial_spearman_q"
COLORS = ["#0072B2", "#E69F00", "#009E73"]  # Okabe-Ito colorblind-safe palette.


def load_discovery_models(config_path: Path | str) -> tuple[str, ...]:
    """Load the configured discovery scope and fixed published-score panel."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    discovery = tuple(config["datasets"]["discovery"])
    if discovery != EXPECTED_DISCOVERY:
        raise ValueError(
            "This fixed discovery plot expects exactly "
            f"{list(EXPECTED_DISCOVERY)}, but config declares {list(discovery)}."
        )
    models = tuple(config["models"]["score_panel"])
    if len(models) != 8 or len(set(models)) != 8:
        raise ValueError("The discovery plot requires eight unique configured score-panel models.")
    return models


def validate_input_paths(inputs: Mapping[str, Path]) -> dict[str, Path]:
    """Reject absent, duplicate, or mislabeled explicit input paths before reading data."""
    if set(inputs) != set(EXPECTED_DISCOVERY):
        raise ValueError(f"Expected explicit inputs for exactly {list(EXPECTED_DISCOVERY)}.")
    resolved: dict[str, Path] = {}
    for dataset in EXPECTED_DISCOVERY:
        path = Path(inputs[dataset])
        if not path.is_file():
            raise FileNotFoundError(f"Missing explicit {dataset} association CSV: {path}")
        resolved[dataset] = path.resolve()
    duplicates = {
        str(path): [dataset for dataset, candidate in resolved.items() if candidate == path]
        for path in set(resolved.values())
    }
    duplicate_sets = [datasets for datasets in duplicates.values() if len(datasets) > 1]
    if duplicate_sets:
        raise ValueError(f"Each discovery dataset requires a distinct CSV; duplicate paths for {duplicate_sets}.")
    return resolved


def _load_one_dataset(path: Path, expected_dataset: str, models: tuple[str, ...]) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = [*KEY_COLUMNS, "analysis_stage", RHO_COLUMN, Q_COLUMN]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is not a compatible H1 association CSV; missing {missing}")
    table = table[required].copy()
    observed_datasets = set(table["dataset"].dropna().astype(str).str.strip())
    if observed_datasets != {expected_dataset}:
        raise ValueError(
            f"{path} must contain only {expected_dataset}, found {sorted(observed_datasets)}."
        )
    table["model"] = table["model"].astype("string").str.strip()
    unknown_models = sorted(set(table["model"].dropna()) - set(models))
    if unknown_models:
        raise ValueError(f"{path} contains model(s) outside the configured score panel: {unknown_models}")
    table["class_label"] = pd.to_numeric(table["class_label"], errors="raise").astype(int)
    if table.duplicated(KEY_COLUMNS).any():
        examples = table.loc[table.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(8).to_dict("records")
        raise ValueError(f"{path} duplicates exact H1 association rows: {examples}")
    subset = table.loc[
        (table["view"] == FOCUS["view"])
        & (table["feature"] == FOCUS["feature"])
        & (table["class_label"] == FOCUS["class_label"])
        & (table["analysis_stage"] == FOCUS["analysis_stage"])
    ].copy()
    if subset["model"].duplicated().any():
        raise ValueError(f"{path} has duplicate model rows in the fixed crest-factor plot slice.")
    missing_models = [model for model in models if model not in set(subset["model"])]
    if missing_models:
        raise ValueError(
            f"{path} is missing configured score-panel model(s) in the fixed plot slice: {missing_models}"
        )
    if len(subset) != len(models):
        raise ValueError(f"{path} expected {len(models)} fixed-slice rows, found {len(subset)}.")
    for column, bounds in [(RHO_COLUMN, (-1.0, 1.0)), (Q_COLUMN, (0.0, 1.0))]:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        if subset[column].isna().any() or not subset[column].between(*bounds).all():
            raise ValueError(f"{path} has non-finite or out-of-range {column} values in the fixed plot slice.")
    return subset


def load_fixed_discovery_slice(inputs: Mapping[str, Path], config_path: Path | str) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Load the predeclared slice from exactly three explicit discovery CSVs."""
    models = load_discovery_models(config_path)
    paths = validate_input_paths(inputs)
    rows = [_load_one_dataset(paths[dataset], dataset, models) for dataset in EXPECTED_DISCOVERY]
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate exact H1 association rows across explicit discovery inputs.")
    result["model"] = pd.Categorical(result["model"], categories=models, ordered=True)
    result["dataset"] = pd.Categorical(result["dataset"], categories=EXPECTED_DISCOVERY, ordered=True)
    return result.sort_values(["model", "dataset"]).reset_index(drop=True), paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_plot(
    rows: pd.DataFrame,
    models: tuple[str, ...],
    output_dir: Path,
    input_paths: Mapping[str, Path],
    *,
    output_stem: str = "h1_discovery_crest_factor_fullwave_spoof",
) -> dict[str, Path]:
    """Render vector and raster versions plus a provenance sidecar."""
    matrix = rows.pivot(index="model", columns="dataset", values=RHO_COLUMN).reindex(
        index=models, columns=EXPECTED_DISCOVERY
    )
    if matrix.isna().any().any():
        raise ValueError("The fixed discovery matrix has missing model/dataset cells.")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(7.16, 3.25))
    positions = np.arange(len(models))
    width = 0.23
    for index, dataset in enumerate(EXPECTED_DISCOVERY):
        offset = (index - 1) * width
        axis.bar(
            positions + offset,
            matrix[dataset].to_numpy(),
            width=width * 0.94,
            color=COLORS[index],
            label=dataset.replace("ASVspoof", "ASV ").replace("_", " "),
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    axis.axhline(0.0, color="#303030", linewidth=0.8, zorder=4)
    axis.set_xticks(positions)
    axis.set_xticklabels(models, rotation=28, ha="right")
    axis.set_ylabel("Partial Spearman $\\rho$ with score$_{spoof}$")
    figure.suptitle(
        "Crest factor vs. published detector scores: discovery screen",
        y=0.99,
        fontsize=10,
        fontweight="bold",
    )
    figure.legend(
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.55, 0.95),
        columnspacing=0.8,
    )
    axis.set_axisbelow(True)
    figure.text(
        0.5,
        0.012,
        "Full waveform · spoof class · discovery-only; not a portable, causal, or candidate-selection claim.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#4A4A4A",
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.32)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be a non-empty filename stem without path components.")
    stem = output_dir / output_stem
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    metadata_path = stem.with_suffix(".metadata.json")
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    metadata = {
        "figure_kind": "fixed_discovery_only_crest_factor_comparison",
        "portable_or_causal_claim": False,
        "candidate_selection_or_freezing_performed": False,
        "filters": {**FOCUS, "association_estimator": RHO_COLUMN},
        "datasets": list(EXPECTED_DISCOVERY),
        "models": list(models),
        "inputs": {
            dataset: {"path": str(input_paths[dataset]), "sha256": _sha256(input_paths[dataset])}
            for dataset in EXPECTED_DISCOVERY
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pdf": pdf_path, "png": png_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvspoof2019-la", required=True, type=Path)
    parser.add_argument("--asvspoof2021-la", required=True, type=Path)
    parser.add_argument("--asvspoof2021-df", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-stem", default="h1_discovery_crest_factor_fullwave_spoof")
    parser.add_argument("--config", default="configs/study.yaml", type=Path)
    args = parser.parse_args()
    inputs = {
        "ASVspoof2019_LA": args.asvspoof2019_la,
        "ASVspoof2021_LA": args.asvspoof2021_la,
        "ASVspoof2021_DF": args.asvspoof2021_df,
    }
    rows, paths = load_fixed_discovery_slice(inputs, args.config)
    models = load_discovery_models(args.config)
    outputs = render_plot(rows, models, args.output_dir, paths, output_stem=args.output_stem)
    for kind, path in outputs.items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
