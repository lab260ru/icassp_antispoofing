"""Render the locked supplementary crest-factor evidence-boundary figure.

This module is intentionally narrow.  It accepts no data paths at run time and
opens only the nine paths pinned (twice: in code and in a versioned manifest)
before rendering three descriptive panels.  It does not import or open audio,
model, score-catalogue, ASR, feature-table, H2B, or response artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "experiments/paper_extension/crest_evidence_boundary_input_manifest.json"
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_extension/crest_evidence_boundary_protocol.md"
OUTPUT_DIR = REPO_ROOT / "experiments/paper_extension/results/crest_evidence_boundary_s1_001"
OUTPUT_STEM = "s1_crest_evidence_boundary"

CORPORA = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
FOCUS = {
    "model": "Spectra-AASIST",
    "view": "full_waveform",
    "feature": "crest_factor_db",
    "class_label": 1,
}
H2_ARMS = ("drc_cf3", "drc_cf6", "small_gain_plus_0p1db", "polarity")
H2_LABELS = ("DRC-3", "DRC-6", "+0.1 dB gain", "Polarity")
H2_GATE = 0.90

# This duplicate pin means a modified manifest cannot redirect the renderer to
# another artifact, even one whose own hash is internally consistent.
EXPECTED_SOURCES: dict[str, dict[str, str]] = {
    "h1_association_asvspoof2019_la": {
        "dataset": "ASVspoof2019_LA",
        "kind": "h1_association_summary",
        "relative_path": "experiments/h1_feature_association/results/ASVspoof2019_LA/association_summary.csv",
        "sha256": "ab276d603a3de23d7200bac6f259dc6227f1f7f2480af096f65e9a8080f66934",
    },
    "h1_association_asvspoof2021_la": {
        "dataset": "ASVspoof2021_LA",
        "kind": "h1_association_summary",
        "relative_path": "experiments/h1_feature_association/results/ASVspoof2021_LA/association_summary.csv",
        "sha256": "3ada0bdc7f6f3f0363435e0288921779da19f9ce7bac24632c8a494fc1225b55",
    },
    "h1_association_asvspoof2021_df": {
        "dataset": "ASVspoof2021_DF",
        "kind": "h1_association_summary",
        "relative_path": "experiments/h1_feature_association/results/ASVspoof2021_DF/association_summary.csv",
        "sha256": "4cfc76949ce4fee6f9e6784fee2b1ed3599742d614c2ec9f3d16352545d26c03",
    },
    "h1_association_inthewild": {
        "dataset": "InTheWild",
        "kind": "h1_association_summary",
        "relative_path": "experiments/h1_feature_association/results/InTheWild/association_summary.csv",
        "sha256": "fca2212ce2ee026f632e56d2f4386a98b844519a1310e10f0db076d19a1b89e4",
    },
    "h1_association_asvspoof5": {
        "dataset": "ASVspoof5",
        "kind": "h1_association_summary",
        "relative_path": "experiments/h1_feature_association/results/ASVspoof5/association_summary.csv",
        "sha256": "d6539ee8194d9d87f28ca46a00716e172fa9cacb98a3f8095b8d35b836be7acb",
    },
    "h1_confirmation_inthewild": {
        "dataset": "InTheWild",
        "kind": "h1_confirmation_bootstrap",
        "relative_path": "experiments/h1_feature_association/results/InTheWild/association_confirmation_bootstrap.csv",
        "sha256": "d45ef60da53e7a35ba19a6a7e91595ed3a1637abccc7cfce0d48e029c64dd825",
    },
    "h1_confirmation_asvspoof5": {
        "dataset": "ASVspoof5",
        "kind": "h1_confirmation_bootstrap",
        "relative_path": "experiments/h1_feature_association/results/ASVspoof5/association_confirmation_bootstrap.csv",
        "sha256": "ff9d1b9690f1bcebc09765dc0d02e206b19138c7c1255b23ce2ed188b98966b9",
    },
    "h2_quality_full_002_summary": {
        "kind": "h2_detector_free_quality_summary",
        "relative_path": "experiments/h2_causal_interventions/results/quality_runs/h2_quality_full_002.summary.json",
        "sha256": "4c5adad8165e97efe44ca7d8b1e8737eb34e0847a4ee08b114c0c0fd61ea7c8e",
    },
    "h4_label_cue_matrix": {
        "kind": "h4_hash_sealed_label_cue_matrix",
        "absolute_path": "/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h4_label_transportability/h4_analysis_001/h4_label_cue_matrix.csv",
        "sha256": "03dc8674db5739bfffa1c2f2b6e2973b31a8294ab5e15f0f7d6b8e35a669b25f",
    },
}

FORBIDDEN_PATH_PARTS = ("/score", "/models", "/audio", "/asr", "/h2b")
FORBIDDEN_RESPONSE_COLUMNS = ("score", "logit", "response", "detector", "prediction")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load locked JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Locked JSON artifact {path} must contain an object.")
    return payload


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(f"{label} keys differ from the fixed contract: expected {sorted(expected)}, got {sorted(observed)}")


def _entry_path(entry: Mapping[str, Any]) -> Path:
    if "relative_path" in entry and "absolute_path" in entry:
        raise ValueError(f"Pinned input {entry.get('id')} declares both relative and absolute paths.")
    if "relative_path" in entry:
        relative = Path(str(entry["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Pinned input {entry.get('id')} has an invalid relative path.")
        path = (REPO_ROOT / relative).resolve()
    elif "absolute_path" in entry:
        path = Path(str(entry["absolute_path"])).resolve()
    else:
        raise ValueError(f"Pinned input {entry.get('id')} has no path.")
    path_text = str(path).casefold()
    if any(part in path_text for part in FORBIDDEN_PATH_PARTS):
        raise ValueError(f"Pinned input {entry.get('id')} crosses a forbidden source boundary: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Pinned input is unavailable: {path}")
    return path


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], Path]]:
    _require_exact_keys(
        manifest,
        {"schema_version", "protocol", "fixed_corpus_order", "focus", "fixed_h2_arm_gate", "inputs", "caption"},
        "input manifest",
    )
    if manifest["schema_version"] != "crest_evidence_boundary_s1_inputs_v1":
        raise ValueError("Input manifest has an unexpected schema version.")
    if tuple(manifest["fixed_corpus_order"]) != CORPORA:
        raise ValueError("Input manifest corpus order is not the locked five-corpus order.")
    if manifest["focus"] != FOCUS:
        raise ValueError("Input manifest focus is not the locked Spectra/full-waveform/spoof/crest slice.")
    if float(manifest["fixed_h2_arm_gate"]) != H2_GATE:
        raise ValueError("Input manifest changed the locked H2 90 percent arm gate.")
    if not isinstance(manifest["caption"], str) or not manifest["caption"].strip():
        raise ValueError("Input manifest must provide the fixed non-empty figure caption.")
    protocol = manifest["protocol"]
    if not isinstance(protocol, dict) or protocol.get("relative_path") != str(PROTOCOL_PATH.relative_to(REPO_ROOT)):
        raise ValueError("Input manifest does not pin the crest evidence-boundary protocol.")
    if protocol.get("sha256") != _sha256(PROTOCOL_PATH):
        raise ValueError("Crest evidence-boundary protocol hash does not match the fixed manifest.")

    entries = manifest["inputs"]
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_SOURCES):
        raise ValueError("Input manifest must declare exactly the fixed source set.")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("Every manifest input needs a string identifier.")
        if entry["id"] in indexed:
            raise ValueError(f"Input manifest duplicates {entry['id']}.")
        indexed[entry["id"]] = entry
    if set(indexed) != set(EXPECTED_SOURCES):
        raise ValueError("Input manifest source identifiers do not match the fixed set.")

    validated: dict[str, tuple[dict[str, Any], Path]] = {}
    for source_id, expected in EXPECTED_SOURCES.items():
        entry = indexed[source_id]
        for key, value in expected.items():
            if entry.get(key) != value:
                raise ValueError(f"Pinned input {source_id} changed fixed {key}.")
        if set(entry) - {"id", "kind", "dataset", "relative_path", "absolute_path", "sha256"}:
            raise ValueError(f"Pinned input {source_id} has unsupported manifest fields.")
        path = _entry_path(entry)
        actual_hash = _sha256(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"Pinned input {source_id} changed after sealing: {path}")
        validated[source_id] = (entry, path)
    return validated


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no CSV header.")
        lowered = [column.casefold() for column in reader.fieldnames]
        if any(token in column for column in lowered for token in FORBIDDEN_RESPONSE_COLUMNS):
            raise ValueError(f"{label} exposes forbidden response-like columns.")
        return list(reader)


def _finite_between(value: str | float, lower: float, upper: float, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{field} is not finite within [{lower}, {upper}]: {number}")
    return number


def _matching_rows(rows: list[dict[str, str]], *, dataset: str, stage: str, label: str) -> dict[str, str]:
    required = {"dataset", "model", "view", "feature", "class_label", "analysis_stage"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{label} does not carry the fixed crest slice keys.")
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["model"] == FOCUS["model"]
        and row["view"] == FOCUS["view"]
        and row["feature"] == FOCUS["feature"]
        and row["class_label"] == str(FOCUS["class_label"])
        and row["analysis_stage"] == stage
    ]
    if len(selected) != 1:
        raise ValueError(f"{label} must contain exactly one locked crest slice row, found {len(selected)}.")
    return selected[0]


def _load_h1_associations(validated: Mapping[str, tuple[dict[str, Any], Path]]) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    values: list[float] = []
    intervals: dict[str, tuple[float, float]] = {}
    names = {
        "ASVspoof2019_LA": "h1_association_asvspoof2019_la",
        "ASVspoof2021_LA": "h1_association_asvspoof2021_la",
        "ASVspoof2021_DF": "h1_association_asvspoof2021_df",
        "InTheWild": "h1_association_inthewild",
        "ASVspoof5": "h1_association_asvspoof5",
    }
    for corpus in CORPORA:
        _, path = validated[names[corpus]]
        row = _matching_rows(_read_csv(path, label=names[corpus]), dataset=corpus, stage="screen", label=names[corpus])
        values.append(_finite_between(row.get("partial_spearman_rho", ""), -1.0, 1.0, f"{corpus} partial rho"))
        _finite_between(row.get("partial_spearman_q", ""), 0.0, 1.0, f"{corpus} partial q")

    confirmations = {
        "InTheWild": "h1_confirmation_inthewild",
        "ASVspoof5": "h1_confirmation_asvspoof5",
    }
    for corpus, source_id in confirmations.items():
        _, path = validated[source_id]
        row = _matching_rows(
            _read_csv(path, label=source_id), dataset=corpus, stage="confirmation_bootstrap", label=source_id
        )
        if row.get("selection_status") != "frozen" or row.get("selection_split") != "discovery":
            raise ValueError(f"{source_id} is not the sealed discovery-frozen confirmation record.")
        low = _finite_between(row.get("partial_spearman_ci_low", ""), -1.0, 1.0, f"{corpus} CI low")
        high = _finite_between(row.get("partial_spearman_ci_high", ""), -1.0, 1.0, f"{corpus} CI high")
        if low > high:
            raise ValueError(f"{source_id} has reversed confidence limits.")
        intervals[corpus] = (low, high)
    return np.asarray(values, dtype=float), intervals


def _load_h2_retention(validated: Mapping[str, tuple[dict[str, Any], Path]]) -> np.ndarray:
    _, path = validated["h2_quality_full_002_summary"]
    summary = _read_json(path)
    if summary.get("artifact_kind") != "h2_detector_free_quality_summary":
        raise ValueError("H2 source is not the locked detector-free quality summary.")
    if summary.get("detector_scoring_allowed") is not False or summary.get("panel_gate_status") != "not_frozen":
        raise ValueError("H2 source does not preserve the no-detector, not-frozen gate boundary.")
    arms = summary.get("per_arm")
    if not isinstance(arms, dict) or set(arms) != set(H2_ARMS):
        raise ValueError("H2 source arms differ from the locked four-arm panel.")
    values = []
    for arm in H2_ARMS:
        item = arms[arm]
        if not isinstance(item, dict):
            raise ValueError(f"H2 arm {arm} is malformed.")
        values.append(_finite_between(item.get("retained_fraction", ""), 0.0, 1.0, f"H2 {arm} retention"))
    return np.asarray(values, dtype=float)


def _load_h4_signed_auc(validated: Mapping[str, tuple[dict[str, Any], Path]]) -> np.ndarray:
    _, path = validated["h4_label_cue_matrix"]
    rows = _read_csv(path, label="h4_label_cue_matrix")
    required = {"dataset", "view", "feature", "cell_status", "delta_auc"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("H4 source does not carry the required score-free label-AUROC fields.")
    selected = [
        row
        for row in rows
        if row["view"] == FOCUS["view"] and row["feature"] == FOCUS["feature"]
    ]
    if len(selected) != len(CORPORA) or {row["dataset"] for row in selected} != set(CORPORA):
        raise ValueError("H4 source is missing a corpus from the fixed full-waveform crest slice.")
    by_corpus: dict[str, float] = {}
    for row in selected:
        if row["cell_status"] != "ok":
            raise ValueError(f"H4 crest cell for {row['dataset']} is not valid.")
        by_corpus[row["dataset"]] = _finite_between(row["delta_auc"], -1.0, 1.0, f"H4 {row['dataset']} signed AUROC")
    return np.asarray([by_corpus[corpus] for corpus in CORPORA], dtype=float)


def load_locked_evidence() -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], Path]], dict[str, Any]]:
    """Validate the immutable manifest and return only the fixed plot quantities."""
    manifest = _read_json(MANIFEST_PATH)
    validated = _validate_manifest(manifest)
    h1_values, h1_intervals = _load_h1_associations(validated)
    h2_values = _load_h2_retention(validated)
    h4_values = _load_h4_signed_auc(validated)
    evidence = {
        "h1_partial_spearman_rho": h1_values,
        "h1_held_out_ci": h1_intervals,
        "h2_retained_fraction": h2_values,
        "h4_signed_label_auroc": h4_values,
    }
    return manifest, validated, evidence


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.0,
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


def _render_figure(
    manifest: Mapping[str, Any],
    validated: Mapping[str, tuple[dict[str, Any], Path]],
    evidence: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    _configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / OUTPUT_STEM
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    metadata_path = stem.with_suffix(".metadata.json")

    h1_values = np.asarray(evidence["h1_partial_spearman_rho"], dtype=float)
    h2_values = np.asarray(evidence["h2_retained_fraction"], dtype=float)
    h4_values = np.asarray(evidence["h4_signed_label_auroc"], dtype=float)
    y = np.arange(len(CORPORA))
    labels = [_display_corpus(corpus) for corpus in CORPORA]

    figure, axes = plt.subplots(1, 3, figsize=(7.16, 3.16), gridspec_kw={"wspace": 0.56})
    axis = axes[0]
    axis.axvline(0.0, color="#303030", linewidth=0.8, zorder=1)
    axis.scatter(h1_values, y, s=35, color="#0072B2", edgecolor="white", linewidth=0.5, zorder=3)
    for idx, corpus in enumerate(CORPORA):
        if corpus in evidence["h1_held_out_ci"]:
            low, high = evidence["h1_held_out_ci"][corpus]
            axis.errorbar(
                h1_values[idx],
                idx,
                xerr=[[h1_values[idx] - low], [high - h1_values[idx]]],
                fmt="none",
                ecolor="#0072B2",
                elinewidth=1.3,
                capsize=2.4,
                zorder=2,
            )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.55, 0.08)
    axis.set_xticks((-0.5, -0.25, 0.0))
    axis.set_xlabel("Partial Spearman $\\rho$")
    axis.set_title("A  Fixed H1 association", loc="left")
    axis.text(
        0.02,
        -0.30,
        "Spectra-AASIST · spoof · full waveform\n95% CI only for held-out confirmations",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#4A4A4A",
    )

    axis = axes[1]
    h2_y = np.arange(len(H2_ARMS))
    colors = ["#D55E00", "#D55E00", "#D55E00", "#009E73"]
    bars = axis.barh(h2_y, h2_values * 100.0, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
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
    axis.set_yticks(h2_y, H2_LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 103.0)
    axis.set_xticks((0, 50, 100))
    axis.set_xlabel("Retained pairs (%)")
    axis.set_title("B  H2 quality gate", loc="left")
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
        -0.30,
        "Detector-free quality screening;\nscoring unavailable (panel not frozen)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#4A4A4A",
    )

    axis = axes[2]
    axis.axvline(0.0, color="#303030", linewidth=0.8, zorder=1)
    h4_colors = ["#0072B2" if value < 0.0 else "#D55E00" for value in h4_values]
    for idx, (value, color) in enumerate(zip(h4_values, h4_colors)):
        axis.hlines(idx, min(0.0, value), max(0.0, value), color=color, linewidth=2.1, zorder=2)
        axis.scatter(value, idx, s=34, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.60, 0.45)
    axis.set_xticks((-0.5, -0.25, 0.0, 0.25))
    axis.set_xlabel("Signed label AUROC")
    axis.set_title("C  H4 label separation", loc="left")
    axis.text(
        0.0,
        -0.30,
        "Score-free raw label separation;\nnot detector reliance or cue selection",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#4A4A4A",
    )
    for axis in axes:
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="major", labelsize=7.1)
    figure.subplots_adjust(left=0.105, right=0.995, top=0.89, bottom=0.30)
    figure.suptitle("Crest factor: a descriptive evidence boundary across five corpora", y=0.985, fontsize=10.0, fontweight="bold")

    fixed_date = datetime(2026, 8, 10, tzinfo=timezone.utc)
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "Supplementary Fig. S1 — Crest-factor evidence boundary",
            "Author": "",
            "Subject": "Sealed descriptive evidence only",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    figure.savefig(
        png_path,
        dpi=300,
        metadata={"Title": "Supplementary Fig. S1 — Crest-factor evidence boundary", "Software": "matplotlib"},
    )
    plt.close(figure)
    if not pdf_path.is_file() or pdf_path.stat().st_size < 1024:
        raise RuntimeError("Vector PDF was not written successfully.")
    if not png_path.is_file() or png_path.stat().st_size < 1024:
        raise RuntimeError("300-DPI PNG was not written successfully.")

    metadata = {
        "artifact_kind": "supplementary_crest_factor_evidence_boundary",
        "schema_version": "crest_evidence_boundary_s1_output_v1",
        "deterministic_renderer": "src.crest_evidence_boundary.render_locked_crest_evidence_boundary",
        "caption": manifest["caption"],
        "claim_boundary": {
            "causal_claim": False,
            "cross_panel_inference": False,
            "detector_scoring_available": False,
            "new_test_or_bootstrap": False,
            "pooled_estimate": False,
            "selection_or_ranking": False,
        },
        "fixed_corpus_order": list(CORPORA),
        "fixed_h2_arm_gate": H2_GATE,
        "figure": {"pdf_vector": True, "png_dpi": 300, "size_inches": [7.16, 3.16]},
        "manifest": {"path": str(MANIFEST_PATH.relative_to(REPO_ROOT)), "sha256": _sha256(MANIFEST_PATH)},
        "protocol": {"path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)), "sha256": _sha256(PROTOCOL_PATH)},
        "input_hashes": {
            source_id: {
                "kind": entry["kind"],
                "path": str(path),
                "sha256": _sha256(path),
            }
            for source_id, (entry, path) in validated.items()
        },
        "rendered_values": {
            "h1_partial_spearman_rho": [float(value) for value in h1_values],
            "h1_held_out_partial_spearman_ci": {
                corpus: [float(limit) for limit in limits]
                for corpus, limits in evidence["h1_held_out_ci"].items()
            },
            "h2_retained_fraction": {arm: float(value) for arm, value in zip(H2_ARMS, h2_values)},
            "h4_signed_label_auroc": [float(value) for value in h4_values],
        },
        "output_hashes": {"pdf_sha256": _sha256(pdf_path), "png_sha256": _sha256(png_path)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pdf": pdf_path, "png": png_path, "metadata": metadata_path}


def render_locked_crest_evidence_boundary(output_dir: Path | None = None) -> dict[str, Path]:
    """Render the only permissible S1 output from the sealed fixed inputs."""
    manifest, validated, evidence = load_locked_evidence()
    target = OUTPUT_DIR if output_dir is None else Path(output_dir)
    return _render_figure(manifest, validated, evidence, target)


__all__ = [
    "CORPORA",
    "FOCUS",
    "H2_ARMS",
    "H2_GATE",
    "MANIFEST_PATH",
    "OUTPUT_DIR",
    "_validate_manifest",
    "load_locked_evidence",
    "render_locked_crest_evidence_boundary",
]
