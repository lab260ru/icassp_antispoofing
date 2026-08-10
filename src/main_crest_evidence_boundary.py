"""Render a two-panel, hash-validated prospective main-paper figure.

The module is intentionally narrow: it accepts only the eight compact sources
listed in its versioned manifest, and it will not read them unless a maintainer
explicitly authorizes a render.  It neither opens nor accepts raw scores,
models, audio, ASR, H2B, H4/H5/H6, or supplementary-S1 inputs.  It copies
already-sealed display values only; it does not calculate a test, interval,
pooled estimate, threshold, ranking, or cross-panel result.
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
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_extension/main_crest_evidence_boundary_protocol.md"
MANIFEST_PATH = REPO_ROOT / "experiments/paper_extension/main_crest_evidence_boundary_input_manifest.json"
OUTPUT_DIR = REPO_ROOT / "paper/figures"
OUTPUT_STEM = "fig_crest_evidence_boundary_main"

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

# This second copy of the manifest's source contract prevents a changed
# manifest from redirecting a renderer to another internally consistent input.
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
}
EXPECTED_SOURCE_ORDER = tuple(EXPECTED_SOURCES)
FORBIDDEN_SOURCE_TOKENS = ("h4", "h5", "h6", "s1", "score", "model", "audio", "asr", "h2b")
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
        raise ValueError(f"Cannot load JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact {path} must contain an object.")
    return payload


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(f"{label} keys differ from the fixed contract: expected {sorted(expected)}, got {sorted(observed)}")


def _entry_path(entry: Mapping[str, Any], *, require_file: bool) -> Path:
    if set(entry) - {"id", "kind", "dataset", "relative_path", "sha256"}:
        raise ValueError(f"Pinned input {entry.get('id')} carries unsupported fields.")
    relative_text = entry.get("relative_path")
    if not isinstance(relative_text, str) or not relative_text:
        raise ValueError(f"Pinned input {entry.get('id')} must use a non-empty repository-relative path.")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Pinned input {entry.get('id')} has an invalid relative path.")
    if any(token in component.casefold() for component in relative.parts for token in FORBIDDEN_SOURCE_TOKENS):
        raise ValueError(f"Pinned input {entry.get('id')} crosses a forbidden source boundary: {relative}")
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Pinned input {entry.get('id')} escapes the repository.") from exc
    if require_file and not path.is_file():
        raise FileNotFoundError(f"Pinned input is unavailable: {path}")
    return path


def _validate_manifest(manifest: Mapping[str, Any], *, verify_sources: bool) -> dict[str, tuple[dict[str, Any], Path]]:
    """Validate the fixed source contract, optionally including source hashes."""
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "protocol",
            "fixed_corpus_order",
            "focus",
            "fixed_h2_arm_order",
            "fixed_h2_arm_gate",
            "inputs",
            "caption",
        },
        "input manifest",
    )
    if manifest["schema_version"] != "main_crest_evidence_boundary_inputs_v1":
        raise ValueError("Input manifest has an unexpected schema version.")
    if tuple(manifest["fixed_corpus_order"]) != CORPORA:
        raise ValueError("Input manifest corpus order is not the locked five-corpus order.")
    if manifest["focus"] != FOCUS:
        raise ValueError("Input manifest focus is not the locked Spectra/full-waveform/spoof/crest slice.")
    if tuple(manifest["fixed_h2_arm_order"]) != H2_ARMS:
        raise ValueError("Input manifest H2 arm order is not the locked four-arm order.")
    if float(manifest["fixed_h2_arm_gate"]) != H2_GATE:
        raise ValueError("Input manifest changed the locked H2 90 percent arm gate.")
    if not isinstance(manifest["caption"], str) or not manifest["caption"].strip():
        raise ValueError("Input manifest must provide the fixed non-empty figure caption.")

    protocol = manifest["protocol"]
    expected_protocol_path = str(PROTOCOL_PATH.relative_to(REPO_ROOT))
    if not isinstance(protocol, dict) or protocol.get("relative_path") != expected_protocol_path:
        raise ValueError("Input manifest does not pin the main-figure protocol.")
    if protocol.get("sha256") != _sha256(PROTOCOL_PATH):
        raise ValueError("Main-figure protocol hash does not match the fixed manifest.")

    entries = manifest["inputs"]
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_SOURCES):
        raise ValueError("Input manifest must declare exactly the fixed eight-source set.")
    if [entry.get("id") if isinstance(entry, dict) else None for entry in entries] != list(EXPECTED_SOURCE_ORDER):
        raise ValueError("Input manifest source order differs from the fixed contract.")

    validated: dict[str, tuple[dict[str, Any], Path]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("Every manifest input needs a string identifier.")
        source_id = entry["id"]
        expected = EXPECTED_SOURCES.get(source_id)
        if expected is None:
            raise ValueError(f"Input manifest includes an unapproved source identifier: {source_id}")
        required_keys = {"id", *expected}
        if set(entry) != required_keys:
            raise ValueError(f"Pinned input {source_id} keys differ from the fixed contract.")
        for key, value in expected.items():
            if entry.get(key) != value:
                raise ValueError(f"Pinned input {source_id} changed fixed {key}.")
        path = _entry_path(entry, require_file=verify_sources)
        if verify_sources and _sha256(path) != expected["sha256"]:
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
        rows = list(reader)
    if not rows:
        raise ValueError(f"{label} has no data rows.")
    return rows


def _finite_between(value: str | float, lower: float, upper: float, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{field} is not finite within [{lower}, {upper}]: {number}")
    return number


def _matching_h1_row(rows: list[dict[str, str]], *, dataset: str, stage: str, label: str) -> dict[str, str]:
    required = {
        "dataset",
        "model",
        "view",
        "feature",
        "class_label",
        "analysis_stage",
        "partial_spearman_rho",
        "partial_spearman_q",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"{label} does not carry the fixed H1 slice fields.")
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
        raise ValueError(f"{label} must contain exactly one fixed H1 slice row, found {len(selected)}.")
    return selected[0]


def _parse_h1_association(rows: list[dict[str, str]], *, dataset: str, label: str) -> float:
    row = _matching_h1_row(rows, dataset=dataset, stage="screen", label=label)
    rho = _finite_between(row["partial_spearman_rho"], -1.0, 1.0, f"{dataset} partial rho")
    _finite_between(row["partial_spearman_q"], 0.0, 1.0, f"{dataset} partial q")
    return rho


def _parse_h1_confirmation(rows: list[dict[str, str]], *, dataset: str, label: str) -> tuple[float, float]:
    row = _matching_h1_row(rows, dataset=dataset, stage="confirmation_bootstrap", label=label)
    if row.get("selection_status") != "frozen" or row.get("selection_split") != "discovery":
        raise ValueError(f"{label} is not the sealed discovery-frozen confirmation record.")
    low = _finite_between(row.get("partial_spearman_ci_low", ""), -1.0, 1.0, f"{dataset} CI low")
    high = _finite_between(row.get("partial_spearman_ci_high", ""), -1.0, 1.0, f"{dataset} CI high")
    if low > high:
        raise ValueError(f"{label} has reversed confidence limits.")
    return low, high


def _parse_h2_retention(summary: Mapping[str, Any]) -> np.ndarray:
    if summary.get("artifact_kind") != "h2_detector_free_quality_summary":
        raise ValueError("H2 source is not the locked detector-free quality summary.")
    if summary.get("detector_scoring_allowed") is not False or summary.get("panel_gate_status") != "not_frozen":
        raise ValueError("H2 source does not preserve the no-detector, not-frozen gate boundary.")
    arms = summary.get("per_arm")
    if not isinstance(arms, dict) or set(arms) != set(H2_ARMS):
        raise ValueError("H2 source arms differ from the locked four-arm set.")
    values: list[float] = []
    for arm in H2_ARMS:
        item = arms[arm]
        if not isinstance(item, dict):
            raise ValueError(f"H2 arm {arm} is malformed.")
        values.append(_finite_between(item.get("retained_fraction", ""), 0.0, 1.0, f"H2 {arm} retention"))
    return np.asarray(values, dtype=float)


def load_locked_main_evidence() -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], Path]], dict[str, Any]]:
    """Revalidate and load only the fixed plot quantities after authorization."""
    manifest = _read_json(MANIFEST_PATH)
    validated = _validate_manifest(manifest, verify_sources=True)
    h1_values: list[float] = []
    association_sources = {
        "ASVspoof2019_LA": "h1_association_asvspoof2019_la",
        "ASVspoof2021_LA": "h1_association_asvspoof2021_la",
        "ASVspoof2021_DF": "h1_association_asvspoof2021_df",
        "InTheWild": "h1_association_inthewild",
        "ASVspoof5": "h1_association_asvspoof5",
    }
    for corpus in CORPORA:
        source_id = association_sources[corpus]
        _, path = validated[source_id]
        h1_values.append(_parse_h1_association(_read_csv(path, label=source_id), dataset=corpus, label=source_id))

    intervals: dict[str, tuple[float, float]] = {}
    for corpus, source_id in (("InTheWild", "h1_confirmation_inthewild"), ("ASVspoof5", "h1_confirmation_asvspoof5")):
        _, path = validated[source_id]
        intervals[corpus] = _parse_h1_confirmation(_read_csv(path, label=source_id), dataset=corpus, label=source_id)

    _, h2_path = validated["h2_quality_full_002_summary"]
    evidence = {
        "h1_partial_spearman_rho": np.asarray(h1_values, dtype=float),
        "h1_held_out_ci": intervals,
        "h2_retained_fraction": _parse_h2_retention(_read_json(h2_path)),
    }
    return manifest, validated, evidence


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


def _fixed_output_paths() -> dict[str, Path]:
    stem = OUTPUT_DIR / OUTPUT_STEM
    return {
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "metadata": stem.with_suffix(".metadata.json"),
    }


def _require_fresh_output_paths(outputs: Mapping[str, Path]) -> None:
    """Refuse a rerender that could silently replace an inspected paper asset."""
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Main-paper evidence-boundary outputs already exist; preserve the "
            "inspected assets and record a new versioned figure contract instead: "
            + ", ".join(str(path) for path in existing)
        )


def _render_figure(
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
    corpus_positions = np.arange(len(CORPORA))

    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.95), gridspec_kw={"wspace": 0.40})
    axis = axes[0]
    axis.axvline(0.0, color="#303030", linewidth=0.8, zorder=1)
    axis.scatter(h1_values, corpus_positions, s=36, color="#0072B2", edgecolor="white", linewidth=0.5, zorder=3)
    for index, corpus in enumerate(CORPORA):
        if corpus in evidence["h1_held_out_ci"]:
            low, high = evidence["h1_held_out_ci"][corpus]
            axis.errorbar(
                h1_values[index],
                index,
                xerr=[[h1_values[index] - low], [high - h1_values[index]]],
                fmt="none",
                ecolor="#0072B2",
                elinewidth=1.35,
                capsize=2.4,
                zorder=2,
            )
    axis.set_yticks(corpus_positions, [_display_corpus(corpus) for corpus in CORPORA])
    axis.invert_yaxis()
    axis.set_xlim(-0.55, 0.08)
    axis.set_xticks((-0.50, -0.25, 0.0))
    axis.set_xlabel("Partial Spearman $\\rho$")
    axis.set_title("A  H1: fixed crest association", loc="left")
    axis.text(
        0.0,
        -0.25,
        "Spectra-AASIST · spoof · full waveform\n95% CI only for held-out confirmations",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#4A4A4A",
    )

    axis = axes[1]
    arm_positions = np.arange(len(H2_ARMS))
    colors = ["#D55E00", "#D55E00", "#D55E00", "#009E73"]
    bars = axis.barh(arm_positions, h2_values * 100.0, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
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
    axis.set_yticks(arm_positions, H2_LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 103.0)
    axis.set_xticks((0, 50, 100))
    axis.set_xlabel("Retained pairs (%)")
    axis.set_title("B  H2: detector-free quality gate", loc="left")
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
        -0.25,
        "Panel not frozen; detector scoring unavailable",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#4A4A4A",
    )

    for axis in axes:
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="major", labelsize=7.2)
    figure.subplots_adjust(left=0.092, right=0.992, top=0.86, bottom=0.29)
    figure.suptitle("Crest factor: an evidence boundary across five corpora", y=0.985, fontsize=10.0, fontweight="bold")

    fixed_date = datetime(2026, 8, 10, tzinfo=timezone.utc)
    figure.savefig(
        outputs["pdf"],
        metadata={
            "Title": "Fig. 1 — Crest-factor evidence boundary",
            "Author": "",
            "Subject": "Sealed descriptive evidence only",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    figure.savefig(
        outputs["png"],
        dpi=300,
        metadata={"Title": "Fig. 1 — Crest-factor evidence boundary", "Software": "matplotlib"},
    )
    plt.close(figure)
    if not outputs["pdf"].is_file() or outputs["pdf"].stat().st_size < 1024:
        raise RuntimeError("Vector PDF was not written successfully.")
    if not outputs["png"].is_file() or outputs["png"].stat().st_size < 1024:
        raise RuntimeError("300-DPI PNG was not written successfully.")

    metadata = {
        "artifact_kind": "main_paper_crest_evidence_boundary",
        "schema_version": "main_crest_evidence_boundary_output_v1",
        "deterministic_renderer": "src.main_crest_evidence_boundary.render_locked_main_figure",
        "caption": manifest["caption"],
        "render_authorization_note": authorization_note,
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
        "figure": {"pdf_vector": True, "png_dpi": 300, "size_inches": [7.16, 2.95]},
        "manifest": {"path": str(MANIFEST_PATH.relative_to(REPO_ROOT)), "sha256": _sha256(MANIFEST_PATH)},
        "protocol": {"path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)), "sha256": _sha256(PROTOCOL_PATH)},
        "input_hashes": {
            source_id: {
                "kind": entry["kind"],
                "path": str(path.relative_to(REPO_ROOT)),
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
        },
        "output_hashes": {
            "pdf_sha256": _sha256(outputs["pdf"]),
            "png_sha256": _sha256(outputs["png"]),
        },
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outputs


def render_locked_main_figure(*, authorized: bool = False, authorization_note: str = "") -> dict[str, Path]:
    """Render only after an explicit maintainer authorization gate is passed."""
    if not authorized:
        raise PermissionError("Refusing to read sealed inputs or render without explicit maintainer authorization.")
    if not isinstance(authorization_note, str) or not authorization_note.strip():
        raise PermissionError("A non-empty maintainer authorization note is required before rendering.")
    manifest, validated, evidence = load_locked_main_evidence()
    return _render_figure(manifest, validated, evidence, authorization_note.strip())


__all__ = [
    "CORPORA",
    "EXPECTED_SOURCES",
    "FOCUS",
    "H2_ARMS",
    "H2_GATE",
    "MANIFEST_PATH",
    "OUTPUT_DIR",
    "PROTOCOL_PATH",
    "_entry_path",
    "_fixed_output_paths",
    "_parse_h1_association",
    "_parse_h1_confirmation",
    "_parse_h2_retention",
    "_read_csv",
    "_validate_manifest",
    "load_locked_main_evidence",
    "render_locked_main_figure",
]
