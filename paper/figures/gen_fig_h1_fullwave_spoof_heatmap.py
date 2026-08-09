#!/usr/bin/env python3
"""Render the complete first-discovery H1 heatmap from its committed CSV.

The plot intentionally includes every v1_28 descriptor for the full-waveform,
spoof-class ASVspoof2019_LA screen.  It is not a post-hoc top-feature plot.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_features import FEATURE_NAMES


INPUT = (
    REPO_ROOT
    / "experiments/h1_feature_association/results/ASVspoof2019_LA/association_summary.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_ORDER = [
    "Spectra-AASIST",
    "AASIST",
    "Res2TCNGuard",
    "RawTFNet",
    "WhisperMFCCMesoNet",
    "W2V2-AASIST",
    "XLSR-SLS",
    "RawBMamba",
]


def prettify_feature(name: str) -> str:
    return name.replace("_", " ").replace("db", "dB").replace("hz", "Hz")


def main() -> None:
    table = pd.read_csv(INPUT)
    subset = table.loc[
        (table["dataset"] == "ASVspoof2019_LA")
        & (table["view"] == "full_waveform")
        & (table["class_label"] == 1)
        & (table["analysis_stage"] == "screen")
    ].copy()
    expected = len(MODEL_ORDER) * len(FEATURE_NAMES)
    if len(subset) != expected:
        raise ValueError(f"Expected {expected} full-waveform spoof rows, found {len(subset)}")
    matrix = subset.pivot(index="feature", columns="model", values="partial_spearman_rho")
    matrix = matrix.reindex(index=FEATURE_NAMES, columns=MODEL_ORDER)
    if matrix.isna().any().any():
        raise ValueError("H1 heatmap has missing model/feature cells")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(7.16, 6.0))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0.0,
        vmin=-0.65,
        vmax=0.65,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Partial Spearman $\\rho$", "shrink": 0.82},
        ax=ax,
    )
    ax.set_xlabel("Published Arena score artifact")
    ax.set_ylabel("Frozen waveform descriptor")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels([prettify_feature(name) for name in matrix.index], rotation=0)
    ax.axhline(FEATURE_NAMES.index("crest_factor_db"), color="#E69F00", linewidth=1.4)
    ax.axhline(FEATURE_NAMES.index("crest_factor_db") + 1, color="#E69F00", linewidth=1.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUTPUT_DIR / "fig_h1_fullwave_spoof_heatmap.pdf")
    fig.savefig(OUTPUT_DIR / "fig_h1_fullwave_spoof_heatmap.png")


if __name__ == "__main__":
    main()
