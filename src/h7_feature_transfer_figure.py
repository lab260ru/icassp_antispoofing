"""Hash-bound display-only forest plot for the sealed H7 transfer matrix."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.h7_feature_transfer import BOOTSTRAP_REPLICATES, CORPORA


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "freeze_manifest": (REPOSITORY_ROOT / "experiments/h7_feature_transfer/results/h7_input_freeze_001/h7_selected_sources.csv", "42592ee49a3867b86f41f83620d328272de4362b32649021028a4e1253fd194f"),
    "freeze_provenance": (REPOSITORY_ROOT / "experiments/h7_feature_transfer/results/h7_input_freeze_001/h7_input_freeze_provenance.json", "236a95a309d201b8b24276e1436d1c3043c3d5d20c6e0685e7b0aaa1503248ec"),
    "matrix": (REPOSITORY_ROOT / "experiments/h7_feature_transfer/results/h7_analysis_001/h7_leave_one_corpus_out.csv", "d8ab31532c7a3867477d3978cc7b3ebc1a9dbf6ef71e4b47086f9fc1dd541e79"),
    "analysis_provenance": (REPOSITORY_ROOT / "experiments/h7_feature_transfer/results/h7_analysis_001/h7_analysis_provenance.json", "9a044edf921ba732e50f0300c06a3ff6484edf1286f7092fb7e96fa652e32cc2"),
}
OUTPUT_DIR = REPOSITORY_ROOT / "experiments/h7_feature_transfer/results/h7_analysis_001/figures_feature_transfer_001"
OUTPUT_FILES = ("h7_feature_transfer_auroc.pdf", "h7_feature_transfer_auroc.png", "h7_feature_transfer_auroc.metadata.json")
# `eer` is an explicitly declared *derived* summary field in the sealed H7
# matrix. It is never plotted, but cannot be rejected as a raw response.
RESPONSE_TOKENS = ("score", "logit", "detector", "model", "audio", "asr")
BLUE = "#0072B2"  # Okabe--Ito
GRAY = "#6B7280"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_like(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in RESPONSE_TOKENS)


def validate_inputs(inputs: Mapping[str, tuple[Path, str]]) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    """Validate the four protocol-pinned inputs without computing a metric."""
    if set(inputs) != set(INPUTS):
        raise ValueError("H7 figure input identities drifted")
    resolved: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, (path, expected_hash) in inputs.items():
        path = Path(path).resolve()
        if _response_like(str(path)) or not path.is_file():
            raise ValueError(f"Invalid H7 figure input path: {path}")
        actual = _sha256(path)
        if actual != expected_hash:
            raise ValueError(f"H7 figure input hash mismatch: {name}")
        resolved[name] = path
        hashes[name] = actual

    table = pd.read_csv(resolved["matrix"])
    bad_headers = [column for column in table.columns if _response_like(column)]
    if bad_headers:
        raise ValueError(f"H7 figure forbids response-like matrix headers: {bad_headers}")
    required = {
        "held_out_dataset", "n_train", "n_test", "auroc", "auroc_ci_low", "auroc_ci_high",
        "fit_converged", "bootstrap_replicates_requested", "bootstrap_replicates_valid",
    }
    if set(table.columns) < required or len(table) != len(CORPORA):
        raise ValueError("H7 figure matrix schema/cardinality drift")
    if table["held_out_dataset"].tolist() != list(CORPORA):
        raise ValueError("H7 figure corpus order drifted")
    if not table["n_train"].eq(40000).all() or not table["n_test"].eq(10000).all():
        raise ValueError("H7 figure train/test cardinality drift")
    if not table["fit_converged"].astype(bool).all():
        raise ValueError("H7 figure requires all converged fits")
    if not table["bootstrap_replicates_requested"].eq(BOOTSTRAP_REPLICATES).all() or not table["bootstrap_replicates_valid"].eq(BOOTSTRAP_REPLICATES).all():
        raise ValueError("H7 figure bootstrap completeness drift")
    values = table.loc[:, ["auroc", "auroc_ci_low", "auroc_ci_high"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any() or not np.all(values[:, 1] <= values[:, 0]) or not np.all(values[:, 0] <= values[:, 2]):
        raise ValueError("H7 figure AUROC/CI validity drift")

    freeze = json.loads(resolved["freeze_provenance"].read_text(encoding="utf-8"))
    analysis = json.loads(resolved["analysis_provenance"].read_text(encoding="utf-8"))
    if freeze.get("reads") != {"score": False, "feature_values": False, "audio": False, "asr": False, "model": False}:
        raise ValueError("H7 figure freeze provenance no-read boundary drift")
    if analysis.get("complete_matrix_cells") != len(CORPORA) or analysis.get("representation") != "all_28_full_waveform_features":
        raise ValueError("H7 figure analysis provenance drift")
    if analysis.get("freeze_manifest_sha256") != hashes["freeze_manifest"] or analysis.get("freeze_provenance_sha256") != hashes["freeze_provenance"]:
        raise ValueError("H7 figure freeze linkage drift")
    if analysis.get("claims_not_supported") != ["detector_reliance", "causality", "cue_selection", "model_ranking", "mitigation_performance"]:
        raise ValueError("H7 figure claim boundary drift")
    return table, analysis, hashes


def render(table: pd.DataFrame, output_dir: Path, *, input_hashes: Mapping[str, str], analysis: Mapping[str, object]) -> dict[str, str]:
    """Render the protocol-locked AUROC forest plot and metadata."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite H7 figure output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    pdf_path = output_dir / OUTPUT_FILES[0]
    png_path = output_dir / OUTPUT_FILES[1]
    metadata_path = output_dir / OUTPUT_FILES[2]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
        }
    )
    figure, axis = plt.subplots(figsize=(6.55, 3.25))
    positions = np.arange(len(CORPORA))
    auroc = table["auroc"].to_numpy(dtype=float)
    low = table["auroc_ci_low"].to_numpy(dtype=float)
    high = table["auroc_ci_high"].to_numpy(dtype=float)
    axis.errorbar(
        auroc,
        positions,
        xerr=np.vstack([auroc - low, high - auroc]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.6,
        capsize=3.0,
        markersize=5.5,
        zorder=3,
    )
    axis.axvline(0.5, color=GRAY, linewidth=1.1, linestyle="--", zorder=1)
    axis.text(0.502, 0.22, "chance", color=GRAY, fontsize=7.5, va="bottom")
    axis.set_yticks(positions)
    axis.set_yticklabels(["ASVspoof 2019 LA", "ASVspoof 2021 LA", "ASVspoof 2021 DF", "In-the-Wild", "ASVspoof 5"])
    axis.invert_yaxis()
    axis.set_xlim(0.45, 1.00)
    axis.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_xlabel("Held-out label AUROC (95% source-ID bootstrap CI)")
    axis.set_title("H7 score-free, feature-only cross-corpus transfer")
    for x, y in zip(auroc, positions, strict=True):
        axis.text(min(x + 0.012, 0.988), y, f"{x:.3f}", va="center", ha="left", fontsize=8, color="#1F2937")
    axis.text(
        0.5,
        -0.22,
        "All 28 full-waveform features | LOO train: 40k | held-out: 10k | no detector score or causal claim",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        color="#374151",
    )
    figure.subplots_adjust(left=0.27, right=0.98, top=0.86, bottom=0.26)
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)

    output_hashes = {"pdf": _sha256(pdf_path), "png": _sha256(png_path)}
    metadata = {
        "protocol": "experiments/h7_feature_transfer/H7_SUPPLEMENTARY_FIGURE_PROTOCOL.md",
        "kind": "display_only_h7_feature_transfer_auroc_forest_plot",
        "input_hashes": dict(input_hashes),
        "output_hashes": output_hashes,
        "corpus_order": list(CORPORA),
        "metric": "held_out_label_auroc",
        "confidence_interval": "95% source-ID percentile bootstrap, 500 replicates",
        "axis_limits": [0.45, 1.0],
        "chance_reference": 0.5,
        "claims_not_supported": analysis["claims_not_supported"],
        "renderer": {"matplotlib": matplotlib.__version__, "python": platform.python_version()},
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {**output_hashes, "metadata": _sha256(metadata_path)}


def render_production() -> dict[str, str]:
    table, analysis, hashes = validate_inputs(INPUTS)
    return render(table, OUTPUT_DIR, input_hashes=hashes, analysis=analysis)
