"""Hash-bound, display-only rendering of the sealed H9-PCR terminal result."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_DIR = Path(
    "/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_terminal_evaluation_001"
)
INPUTS = {
    "metrics": (TERMINAL_DIR / "h9_terminal_target_metrics.csv", "2e744154d25b3f515d1b0121db7dc8658cd9b4fb5ff4a9a45dca79b1d8f35ee4"),
    "bootstrap": (TERMINAL_DIR / "h9_terminal_bootstrap_macro_eer_differences.csv", "771a4f3b6b794503b725e447dcd655cde65a3a2dbdedc851ee55fa71c938e9c0"),
    "decision": (TERMINAL_DIR / "h9_terminal_decision_gate.json", "a4e5f118cd34af6ae2e728e094d2ad9e9e6ebf1a8ebe1fb2d841b191c3753852"),
    "provenance": (TERMINAL_DIR / "h9_terminal_evaluation_provenance.json", "af0b85c8f05ad9f2b70cad83b45db7922cd9422732c4630dbb17d1ad2821d2f4"),
}
OUTPUT_DIR = REPOSITORY_ROOT / "experiments/h9_paired_counterfactual/results/h9_terminal_evaluation_001/figures_terminal_003"
OUTPUT_FILES = ("h9_pcr_terminal_eer.pdf", "h9_pcr_terminal_eer.png", "h9_pcr_terminal_eer.metadata.json")
DATASETS = ("SONAR", "ArAD")
METHODS = ("B1", "B2", "P")
METRIC_COLUMNS = ("dataset", "method", "n_trials", "n_bonafide", "n_spoof", "eer", "eer_percent", "auroc", "score_aggregation")
BOOTSTRAP_COLUMNS = ("replicate", "macro_eer_difference_p_minus_b1", "macro_eer_difference_p_minus_b2")
COLORS = {"B1": "#0072B2", "B2": "#E69F00", "P": "#009E73"}
METHOD_LABELS = {"B1": "B1: BCE", "B2": "B2: random-pair", "P": "P: same-item"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H9 terminal figure {description} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"H9 terminal figure {description} must be an object")
    return value


def validate_inputs(inputs: Mapping[str, tuple[Path, str]] = INPUTS) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, str]]:
    """Validate the four fixed display inputs without recomputing a metric."""
    if set(inputs) != set(INPUTS):
        raise ValueError("H9 terminal figure input identities drifted")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, (path, expected) in inputs.items():
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"H9 terminal figure input is unavailable: {resolved}")
        actual = _sha256(resolved)
        if actual != expected:
            raise ValueError(f"H9 terminal figure input hash mismatch: {name}")
        paths[name] = resolved
        hashes[name] = actual

    metrics = pd.read_csv(paths["metrics"])
    if tuple(metrics.columns.astype(str)) != METRIC_COLUMNS or len(metrics) != 6:
        raise ValueError("H9 terminal figure metric schema/cardinality drift")
    expected_keys = [(dataset, method) for dataset in DATASETS for method in METHODS]
    if list(metrics.loc[:, ["dataset", "method"]].itertuples(index=False, name=None)) != expected_keys:
        raise ValueError("H9 terminal figure metric ordering drift")
    values = metrics.loc[:, ["eer", "eer_percent", "auroc"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, [0, 2]] < 0.0).any() or (values[:, [0, 2]] > 1.0).any():
        raise ValueError("H9 terminal figure metric values are invalid")
    if not np.allclose(values[:, 1], values[:, 0] * 100.0, atol=1e-10, rtol=0.0):
        raise ValueError("H9 terminal figure EER percent relation drift")
    if not metrics["score_aggregation"].eq("mean_spoof_probability_over_four_frozen_seeds").all():
        raise ValueError("H9 terminal figure score aggregation drift")

    bootstrap = pd.read_csv(paths["bootstrap"])
    if tuple(bootstrap.columns.astype(str)) != BOOTSTRAP_COLUMNS or len(bootstrap) != 2_000:
        raise ValueError("H9 terminal figure bootstrap schema/cardinality drift")
    if bootstrap["replicate"].tolist() != list(range(2_000)) or not np.isfinite(bootstrap.iloc[:, 1:].to_numpy(dtype=float)).all():
        raise ValueError("H9 terminal figure bootstrap identity/value drift")

    decision = _load_json(paths["decision"], description="decision")
    provenance = _load_json(paths["provenance"], description="provenance")
    if decision.get("artifact_kind") != "h9_pcr_terminal_decision_gate" or decision.get("positive_result_gate_passed") is not True:
        raise ValueError("H9 terminal figure requires the sealed passing decision gate")
    if provenance.get("artifact_kind") != "h9_pcr_terminal_target_evaluation" or provenance.get("version") != "h9-pcr-terminal-evaluation-v1":
        raise ValueError("H9 terminal figure provenance identity drift")
    if provenance.get("target_metrics_sha256") != hashes["metrics"] or provenance.get("bootstrap_sha256") != hashes["bootstrap"] or provenance.get("decision_gate_sha256") != hashes["decision"]:
        raise ValueError("H9 terminal figure provenance hash linkage drift")
    if provenance.get("source_target_canonical_fingerprint_collision_count") != 0 or provenance.get("evaluation_precision") != "cuda_bfloat16_autocast":
        raise ValueError("H9 terminal figure provenance integrity drift")
    if decision.get("bootstrap", {}).get("replicates") != 2_000 or decision.get("bootstrap", {}).get("seed") != 2909:
        raise ValueError("H9 terminal figure bootstrap configuration drift")
    return metrics, decision, provenance, hashes


def _macro_eer(metrics: pd.DataFrame) -> dict[str, float]:
    return {method: float(metrics.loc[metrics["method"] == method, "eer_percent"].mean()) for method in METHODS}


def render(
    metrics: pd.DataFrame,
    decision: Mapping[str, object],
    provenance: Mapping[str, object],
    output_dir: Path,
    *,
    input_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Render the locked two-panel terminal result figure."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite H9 terminal figure output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    pdf_path, png_path, metadata_path = (output_dir / name for name in OUTPUT_FILES)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5,
        "legend.fontsize": 7.6, "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.16, "grid.linestyle": "-",
    })
    figure, (bar_axis, delta_axis) = plt.subplots(1, 2, figsize=(6.75, 2.75), gridspec_kw={"width_ratios": [1.28, 0.92]})
    categories = ("SONAR", "ArAD", "Macro")
    x = np.arange(len(categories), dtype=float)
    width = 0.22
    macro = _macro_eer(metrics)
    for index, method in enumerate(METHODS):
        dataset_values = [float(metrics.loc[(metrics["dataset"] == dataset) & (metrics["method"] == method), "eer_percent"].iloc[0]) for dataset in DATASETS]
        values = dataset_values + [macro[method]]
        bars = bar_axis.bar(x + (index - 1) * width, values, width * 0.9, color=COLORS[method], edgecolor="white", linewidth=0.5, label=METHOD_LABELS[method], zorder=3)
        for bar, value in zip(bars, values, strict=True):
            bar_axis.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=6.9, color="#27313A")
    bar_axis.set_xticks(x)
    bar_axis.set_xticklabels(categories)
    bar_axis.set_ylim(0.0, 76.0)
    bar_axis.set_ylabel("EER (%) ↓")
    bar_axis.set_title("External terminal EER")
    bar_axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.7, handlelength=1.0, columnspacing=0.75, handletextpad=0.3)

    bootstrap = decision["bootstrap"]
    rows = (("P − B1", "p_minus_b1"), ("P − B2", "p_minus_b2"))
    for y, (label, key) in enumerate(rows):
        interval = bootstrap[key]
        center = float(interval["mean"]) * 100.0
        low = float(interval["ci_low"]) * 100.0
        high = float(interval["ci_high"]) * 100.0
        delta_axis.errorbar(center, y, xerr=np.array([[center - low], [high - center]]), fmt="o", color="#009E73", ecolor="#009E73", elinewidth=1.8, capsize=3.1, markersize=5.5, zorder=3)
        delta_axis.text((low + high) / 2.0, y - 0.20, f"95% CI [{low:.1f}, {high:.1f}]", ha="center", va="center", fontsize=6.7, color="#27313A")
    delta_axis.axvline(0.0, color="#6B7280", linewidth=1.0, linestyle="--", zorder=1)
    delta_axis.set_yticks([0, 1])
    delta_axis.set_yticklabels([label for label, _ in rows])
    delta_axis.invert_yaxis()
    delta_axis.set_xlim(-10.0, 1.0)
    delta_axis.set_xticks([-10, -7.5, -5, -2.5, 0])
    delta_axis.set_xlabel("Macro EER difference (pp), 95% CI")
    delta_axis.set_title("P minus control macro EER")
    figure.text(0.5, 0.005, "4-seed probability ensemble | 2 fixed external targets | 2,000 shared-ID stratified bootstrap replicates", ha="center", va="bottom", fontsize=7.1, color="#374151")
    figure.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.24, wspace=0.45)
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)

    output_hashes = {"pdf": _sha256(pdf_path), "png": _sha256(png_path)}
    metadata = {
        "protocol": "experiments/h9_paired_counterfactual/H9_RESULT_FIGURE_PROTOCOL.md",
        "kind": "display_only_h9_pcr_terminal_eer_and_bootstrap_contrasts",
        "input_hashes": dict(input_hashes), "output_hashes": output_hashes,
        "datasets": list(DATASETS), "methods": list(METHODS), "metric": "terminal_eer_percent",
        "uncertainty": "95% percentile interval from fixed 2,000 shared-ID label-stratified trial-bootstrap replicates (seed 2909), conditional on the frozen four-seed ensemble",
        "terminal_device": provenance["evaluation_device"], "terminal_precision": provenance["evaluation_precision"],
        "claims_not_supported": ["causality", "state_of_the_art", "arbitrary_target_generalization", "representation_mechanism"],
        "renderer": {"matplotlib": matplotlib.__version__, "python": platform.python_version()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**output_hashes, "metadata": _sha256(metadata_path)}


def render_production() -> dict[str, str]:
    metrics, decision, provenance, hashes = validate_inputs(INPUTS)
    return render(metrics, decision, provenance, OUTPUT_DIR, input_hashes=hashes)
