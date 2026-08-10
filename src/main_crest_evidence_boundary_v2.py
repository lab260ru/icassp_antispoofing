"""Render v2 of the sealed main-paper crest evidence-boundary figure.

This is a presentation-only successor to the preserved v1 asset.  It verifies
the exact committed v1 loader and v1 eight-input manifest before it calls that
loader, then renders the same sealed values with clarified corpus/panel scope.
It neither reads new sources nor computes a new research quantity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.main_crest_evidence_boundary import (
    CORPORA,
    H2_ARMS,
    H2_GATE,
    H2_LABELS,
    MANIFEST_PATH as V1_MANIFEST_PATH,
    REPO_ROOT,
    load_locked_main_evidence,
)


V1_MODULE_PATH = REPO_ROOT / "src/main_crest_evidence_boundary.py"
V2_PROTOCOL_PATH = REPO_ROOT / "experiments/paper_extension/main_crest_evidence_boundary_v2_protocol.md"
OUTPUT_DIR = REPO_ROOT / "paper/figures"
OUTPUT_STEM = "fig_crest_evidence_boundary_main_v2"

EXPECTED_V1_MODULE_SHA256 = "ad8db67589eab3871b9108a85f5a3d61855be1eb29fd949faa36a1a14f497e9a"
EXPECTED_V1_MANIFEST_SHA256 = "10d6c6cdc0a694f8d2c93a67a3c69714f2ff98c7463da20570c80f43426b2501"
EXPECTED_V2_PROTOCOL_SHA256 = "d49042070e56659d09c7892a5ea2d533bfe915607a61c23c03b685706c0d70dd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_layout_only_dependencies() -> None:
    """Pin v2 to the exact v1 parser and sealed eight-source manifest."""
    checks = (
        (V1_MODULE_PATH, EXPECTED_V1_MODULE_SHA256, "v1 source module"),
        (V1_MANIFEST_PATH, EXPECTED_V1_MANIFEST_SHA256, "v1 input manifest"),
        (V2_PROTOCOL_PATH, EXPECTED_V2_PROTOCOL_SHA256, "v2 layout protocol"),
    )
    for path, expected, label in checks:
        if not path.is_file():
            raise FileNotFoundError(f"Main-figure v2 {label} is unavailable: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"Main-figure v2 refuses a changed {label}: {path}")


def _fixed_output_paths() -> dict[str, Path]:
    stem = OUTPUT_DIR / OUTPUT_STEM
    return {
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "metadata": stem.with_suffix(".metadata.json"),
    }


def _require_fresh_output_paths(outputs: Mapping[str, Path]) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Main-paper evidence-boundary v2 outputs already exist; preserve the "
            "inspected assets and create a new versioned contract instead: "
            + ", ".join(str(path) for path in existing)
        )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.1,
            "axes.titleweight": "bold",
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


def _display_corpus(corpus: str) -> str:
    return {
        "ASVspoof2019_LA": "ASV19 LA",
        "ASVspoof2021_LA": "ASV21 LA",
        "ASVspoof2021_DF": "ASV21 DF",
        "InTheWild": "InTheWild",
        "ASVspoof5": "ASV5",
    }[corpus]


def _render(
    manifest: Mapping[str, Any],
    validated: Mapping[str, tuple[dict[str, Any], Path]],
    evidence: Mapping[str, Any],
    authorization_note: str,
) -> dict[str, Path]:
    _configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = _fixed_output_paths()
    _require_fresh_output_paths(outputs)

    h1_values = np.asarray(evidence["h1_partial_spearman_rho"], dtype=float)
    h2_values = np.asarray(evidence["h2_retained_fraction"], dtype=float)
    # The gap and fixed divider make the discovery/held-out boundary visible
    # without moving or sorting any corpus within either locked group.
    h1_y = np.asarray([0.0, 1.0, 2.0, 3.55, 4.55])
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), gridspec_kw={"wspace": 0.40})

    axis = axes[0]
    axis.axvline(0.0, color="#303030", linewidth=0.8, zorder=1)
    axis.axhline(2.78, color="#7A7A7A", linewidth=0.75, linestyle="--", zorder=1)
    axis.scatter(h1_values, h1_y, s=36, color="#0072B2", edgecolor="white", linewidth=0.5, zorder=3)
    for index, corpus in enumerate(CORPORA):
        if corpus in evidence["h1_held_out_ci"]:
            low, high = evidence["h1_held_out_ci"][corpus]
            axis.errorbar(
                h1_values[index],
                h1_y[index],
                xerr=[[h1_values[index] - low], [high - h1_values[index]]],
                fmt="none",
                ecolor="#0072B2",
                elinewidth=1.35,
                capsize=2.4,
                zorder=2,
            )
    axis.set_yticks(h1_y, [_display_corpus(corpus) for corpus in CORPORA])
    axis.invert_yaxis()
    axis.set_xlim(-0.55, 0.08)
    axis.set_xticks((-0.50, -0.25, 0.0))
    axis.set_xlabel("Partial Spearman $\\rho$")
    axis.set_title("A  H1: five-corpus fixed association", loc="left")
    axis.text(-0.54, 2.58, "locked discovery", ha="left", va="bottom", fontsize=6.3, color="#4A4A4A")
    axis.text(-0.54, 2.96, "held-out confirmation", ha="left", va="top", fontsize=6.3, color="#4A4A4A")
    axis.text(
        0.0,
        -0.24,
        "Spectra-AASIST · spoof · full waveform\n95% CI only for held-out confirmations",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.65,
        color="#4A4A4A",
    )

    axis = axes[1]
    arm_y = np.arange(len(H2_ARMS))
    # Gray identifies polarity as an individually passing control rather than a
    # scientific winner. It cannot unfreeze the failed four-arm panel.
    colors = ["#D55E00", "#D55E00", "#D55E00", "#7A7A7A"]
    bars = axis.barh(arm_y, h2_values * 100.0, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    axis.axvline(H2_GATE * 100.0, color="#303030", linewidth=0.9, linestyle="--", zorder=4)
    axis.annotate(
        "locked 90% gate",
        xy=(H2_GATE * 100.0, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(-2, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.8,
        color="#303030",
    )
    axis.set_yticks(arm_y, H2_LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 103.0)
    axis.set_xticks((0, 50, 100))
    axis.set_xlabel("Retained pairs (%)")
    axis.set_title("B  H2: ASV19 LA detector-free quality screen", loc="left")
    for bar, value in zip(bars, h2_values):
        offset = -4.0 if value > 0.92 else 2.0
        alignment = "right" if value > 0.92 else "left"
        axis.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"{100.0 * value:.1f}%",
            ha=alignment,
            va="center",
            fontsize=7.1,
            color="white" if value > 0.92 else "#303030",
            zorder=5,
        )
    axis.text(
        0.0,
        -0.24,
        "1,000-clip panel; no score-eligible manifest frozen",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.65,
        color="#4A4A4A",
    )

    for axis in axes:
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="major", labelsize=7.2)
    figure.subplots_adjust(left=0.092, right=0.992, top=0.86, bottom=0.30)
    figure.suptitle("Crest factor: evidence boundary", y=0.985, fontsize=10.0, fontweight="bold")

    fixed_date = datetime(2026, 8, 10, tzinfo=timezone.utc)
    figure.savefig(
        outputs["pdf"],
        metadata={
            "Title": "Fig. 1 v2 — Crest-factor evidence boundary",
            "Author": "",
            "Subject": "Sealed descriptive evidence only",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    figure.savefig(outputs["png"], dpi=300, metadata={"Title": "Fig. 1 v2 — Crest-factor evidence boundary", "Software": "matplotlib"})
    plt.close(figure)
    if not outputs["pdf"].is_file() or outputs["pdf"].stat().st_size < 1024:
        raise RuntimeError("Main-paper v2 vector PDF was not written successfully.")
    if not outputs["png"].is_file() or outputs["png"].stat().st_size < 1024:
        raise RuntimeError("Main-paper v2 300-DPI PNG was not written successfully.")

    metadata = {
        "artifact_kind": "main_paper_crest_evidence_boundary_v2",
        "schema_version": "main_crest_evidence_boundary_output_v2",
        "layout_only_changes": [
            "figure title does not imply a five-corpus H2 panel",
            "fixed discovery/held-out divider and labels in H1 panel",
            "ASVspoof2019 LA 1,000-clip H2 context and score-eligible-manifest boundary",
            "neutral polarity-control color",
        ],
        "claim_boundary": {
            "causal_claim": False,
            "cross_panel_inference": False,
            "detector_scoring_available": False,
            "new_test_or_bootstrap": False,
            "pooled_estimate": False,
            "selection_or_ranking": False,
        },
        "fixed_corpus_order": list(CORPORA),
        "fixed_h2_arm_order": list(H2_ARMS),
        "fixed_h2_arm_gate": H2_GATE,
        "figure": {"pdf_vector": True, "png_dpi": 300, "size_inches": [7.16, 3.05]},
        "authorization_note": authorization_note,
        "v1_dependency": {
            "module": str(V1_MODULE_PATH.relative_to(REPO_ROOT)),
            "module_sha256": _sha256(V1_MODULE_PATH),
            "input_manifest": str(V1_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "input_manifest_sha256": _sha256(V1_MANIFEST_PATH),
        },
        "v2_protocol": {"path": str(V2_PROTOCOL_PATH.relative_to(REPO_ROOT)), "sha256": _sha256(V2_PROTOCOL_PATH)},
        "input_hashes": {
            source_id: {"kind": entry["kind"], "path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
            for source_id, (entry, path) in validated.items()
        },
        "rendered_values": {
            "h1_partial_spearman_rho": [float(value) for value in h1_values],
            "h1_held_out_partial_spearman_ci": {
                corpus: [float(limit) for limit in limits]
                for corpus, limits in evidence["h1_held_out_ci"].items()
            },
            "h2_retained_fraction": {arm: float(value) for arm, value in zip(H2_ARMS, h2_values)},
        },
        "output_hashes": {"pdf_sha256": _sha256(outputs["pdf"]), "png_sha256": _sha256(outputs["png"])},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outputs


def render_main_figure_v2(*, authorized: bool = False, authorization_note: str = "") -> dict[str, Path]:
    """Render only after a maintainer explicitly authorizes the v2 layout."""
    if not authorized:
        raise PermissionError("Refusing to read sealed inputs or render v2 without explicit maintainer authorization.")
    if not isinstance(authorization_note, str) or not authorization_note.strip():
        raise PermissionError("A non-empty maintainer authorization note is required before rendering v2.")
    _verify_layout_only_dependencies()
    manifest, validated, evidence = load_locked_main_evidence()
    return _render(manifest, validated, evidence, authorization_note.strip())


__all__ = [
    "EXPECTED_V1_MANIFEST_SHA256",
    "EXPECTED_V1_MODULE_SHA256",
    "EXPECTED_V2_PROTOCOL_SHA256",
    "OUTPUT_DIR",
    "V1_MANIFEST_PATH",
    "V1_MODULE_PATH",
    "V2_PROTOCOL_PATH",
    "_fixed_output_paths",
    "_require_fresh_output_paths",
    "_verify_layout_only_dependencies",
    "render_main_figure_v2",
]
