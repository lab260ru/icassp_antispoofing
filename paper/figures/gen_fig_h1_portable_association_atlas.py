#!/usr/bin/env python3
"""Plot the fixed five-corpus H1 association atlas without selecting candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "experiments/h1_feature_association/results/five_corpus_aggregate_20260809T214500Z/portable_association_feature_report.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PDF = OUTPUT_DIR / "fig_h1_portable_association_atlas.pdf"
PNG = OUTPUT_DIR / "fig_h1_portable_association_atlas.png"
METADATA = OUTPUT_DIR / "fig_h1_portable_association_atlas.metadata.json"

VIEW_LABELS = {
    "full_waveform": "Full waveform",
    "deterministic_crop": "Deterministic crop",
    "preemphasized_crop": "Pre-emphasized crop",
}
VIEW_COLORS = {
    "full_waveform": "#0072B2",
    "deterministic_crop": "#009E73",
    "preemphasized_crop": "#D55E00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    table = pd.read_csv(INPUT)
    selected = table.loc[table["final_portable_criterion_met"].astype(bool)].copy()
    if len(selected) != 19:
        raise ValueError(f"Expected exactly 19 fixed-rule units, got {len(selected)}")
    selected["view_order"] = selected["view"].map({"full_waveform": 0, "deterministic_crop": 1, "preemphasized_crop": 2})
    selected = selected.sort_values(
        ["models_meeting_portable_subcriterion", "view_order", "feature", "class_label"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected["label"] = selected.apply(
        lambda row: f"{VIEW_LABELS[row['view']]} | {row['feature']} | {'spoof' if int(row['class_label']) else 'bonafide'}",
        axis=1,
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.4,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )
    fig, ax = plt.subplots(figsize=(6.75, 5.2))
    bars = ax.barh(
        selected["label"],
        selected["models_meeting_portable_subcriterion"],
        color=[VIEW_COLORS[view] for view in selected["view"]],
        edgecolor="white",
        linewidth=0.55,
        height=0.72,
        zorder=3,
    )
    for bar, count in zip(bars, selected["models_meeting_portable_subcriterion"], strict=True):
        ax.text(float(count) + 0.06, bar.get_y() + bar.get_height() / 2, str(int(count)), va="center", fontsize=8)
    ax.set_xlim(0, 8)
    ax.set_xticks(range(0, 9))
    ax.set_xlabel("Models satisfying the fixed five-corpus subcriterion (of 8)")
    ax.set_ylabel("")
    ax.set_title("Registered five-corpus association atlas (descriptive; not H2 selection)", pad=8)
    fig.subplots_adjust(left=0.43, right=0.96, top=0.92, bottom=0.10)
    fig.savefig(PDF)
    fig.savefig(PNG, dpi=300)
    metadata = {
        "input": str(INPUT.relative_to(REPO_ROOT)),
        "input_sha256": sha256(INPUT),
        "rule": "same direction in >=4/5 core datasets, significant in both confirmation datasets, and present in >=5/8 models",
        "statistic": "partial_spearman with per-corpus BH q",
        "units_plotted": int(len(selected)),
        "selection_guard": "The plot visualizes already-evaluated registered units; it does not select or freeze H2 candidates.",
        "outputs": {"pdf": PDF.name, "png": PNG.name},
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
