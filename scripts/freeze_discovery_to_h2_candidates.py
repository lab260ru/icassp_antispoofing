#!/usr/bin/env python3
"""Mechanically freeze exploratory H2 families from three explicit H1 screens.

The command implements only the locked rule in
``experiments/h1_feature_association/DISCOVERY_TO_H2_FREEZE.md``. It never
opens confirmation/H2 tables, ranks features, or runs an intervention. Its
outputs are an exact H1 bootstrap-candidate manifest and a provenance report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_associations import CANDIDATE_MANIFEST_COLUMNS


EXPECTED_DISCOVERY = ("ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF")
PARITY_VALIDATED_RUNNERS = ("Spectra-AASIST", "AASIST")
FEATURE_FAMILIES = (
    "crest_factor_db",
    "silence_fraction",
    "spectral_slope_db_per_khz",
    "group_delay_var",
)
SELECTION_BASIS = "experiments/h1_feature_association/DISCOVERY_TO_H2_FREEZE.md"
FULL_WAVEFORM_SPOOF = {
    "view": "full_waveform",
    "class_label": 1,
    "analysis_stage": "screen",
}
KEY_COLUMNS = ["dataset", "model", "view", "feature", "class_label"]
RHO_COLUMN = "partial_spearman_rho"
Q_COLUMN = "partial_spearman_q"
Q_THRESHOLD = 0.05
EFFECT_THRESHOLD = 0.05


def sha256_file(path: Path) -> str:
    """Return a content hash without retaining source data in the report."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc_timestamp(value: str) -> str:
    """Validate and preserve an explicit RFC3339 UTC timestamp for the freeze."""
    if not value.endswith("Z"):
        raise ValueError("--frozen-at-utc must be an explicit UTC timestamp ending in 'Z'.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("--frozen-at-utc must be ISO-8601 / RFC3339 UTC, e.g. 2026-08-10T12:34:56Z.") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("--frozen-at-utc must resolve to UTC.")
    return value


def load_configured_panel(config_path: Path | str) -> tuple[str, ...]:
    """Validate the exact named discovery scope and configured eight-model panel."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    discovery = tuple(config["datasets"]["discovery"])
    if discovery != EXPECTED_DISCOVERY:
        raise ValueError(
            "The freeze rule is locked to exactly "
            f"{list(EXPECTED_DISCOVERY)}, but config declares {list(discovery)}."
        )
    panel = tuple(config["models"]["score_panel"])
    if len(panel) != 8 or len(set(panel)) != 8:
        raise ValueError("The locked freeze rule requires eight unique configured score-panel models.")
    missing_runners = [runner for runner in PARITY_VALIDATED_RUNNERS if runner not in panel]
    if missing_runners:
        raise ValueError(f"Configured score panel lacks parity-validated runner(s): {missing_runners}")
    return panel


def validate_explicit_inputs(inputs: Mapping[str, Path]) -> dict[str, Path]:
    """Require one distinct existing CSV for each and only each discovery corpus."""
    if set(inputs) != set(EXPECTED_DISCOVERY):
        raise ValueError(f"Expected explicit inputs for exactly {list(EXPECTED_DISCOVERY)}.")
    resolved: dict[str, Path] = {}
    for dataset in EXPECTED_DISCOVERY:
        path = Path(inputs[dataset])
        if not path.is_file():
            raise FileNotFoundError(f"Missing explicit {dataset} association CSV: {path}")
        resolved[dataset] = path.resolve()
    reverse_paths: dict[Path, list[str]] = {}
    for dataset, path in resolved.items():
        reverse_paths.setdefault(path, []).append(dataset)
    reused = [datasets for datasets in reverse_paths.values() if len(datasets) > 1]
    if reused:
        raise ValueError(f"Each named discovery corpus requires a distinct CSV; duplicate path assignments: {reused}")
    return resolved


def _load_one_screen(path: Path, expected_dataset: str, panel: tuple[str, ...]) -> pd.DataFrame:
    """Validate one exclusive discovery screen without looking at held-out data."""
    table = pd.read_csv(path)
    required = [*KEY_COLUMNS, "analysis_stage", RHO_COLUMN, Q_COLUMN]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is not a compatible H1 association CSV; missing {missing}")
    table = table[required].copy()
    table["dataset"] = table["dataset"].astype("string").str.strip()
    table["model"] = table["model"].astype("string").str.strip()
    observed_datasets = set(table["dataset"].dropna())
    if observed_datasets != {expected_dataset}:
        raise ValueError(f"{path} must contain only {expected_dataset}, found {sorted(observed_datasets)}.")
    unknown_models = sorted(set(table["model"].dropna()) - set(panel))
    missing_models = [model for model in panel if model not in set(table["model"].dropna())]
    if unknown_models or missing_models:
        raise ValueError(
            f"{path} does not match the configured score panel; unknown={unknown_models}, missing={missing_models}."
        )
    table["class_label"] = pd.to_numeric(table["class_label"], errors="raise").astype(int)
    if not table["class_label"].isin([0, 1]).all():
        raise ValueError(f"{path} contains a class_label outside {{0, 1}}.")
    if table.duplicated(KEY_COLUMNS).any():
        examples = table.loc[table.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(8).to_dict("records")
        raise ValueError(f"{path} duplicates exact H1 association keys: {examples}")
    return table


def _json_number(value: float) -> float | None:
    """Keep provenance JSON standards-compliant when an input cell is absent/non-finite."""
    return float(value) if np.isfinite(value) else None


def load_discovery_screens(inputs: Mapping[str, Path], config_path: Path | str) -> tuple[pd.DataFrame, dict[str, Path], tuple[str, ...]]:
    """Read exactly the explicit discovery screens after strict scope validation."""
    panel = load_configured_panel(config_path)
    paths = validate_explicit_inputs(inputs)
    tables = [_load_one_screen(paths[dataset], dataset, panel) for dataset in EXPECTED_DISCOVERY]
    all_rows = pd.concat(tables, ignore_index=True)
    if all_rows.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate exact H1 association keys across explicitly supplied discovery CSVs.")
    return all_rows, paths, panel


def evaluate_fixed_rule(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all fixed runner--feature pairs; never rank or prune by strength."""
    fixed_slice = rows.loc[
        (rows["view"] == FULL_WAVEFORM_SPOOF["view"])
        & (rows["class_label"] == FULL_WAVEFORM_SPOOF["class_label"])
        & (rows["analysis_stage"] == FULL_WAVEFORM_SPOOF["analysis_stage"])
        & (rows["feature"].isin(FEATURE_FAMILIES))
        & (rows["model"].isin(PARITY_VALIDATED_RUNNERS))
    ].copy()
    fixed_slice[RHO_COLUMN] = pd.to_numeric(fixed_slice[RHO_COLUMN], errors="coerce")
    fixed_slice[Q_COLUMN] = pd.to_numeric(fixed_slice[Q_COLUMN], errors="coerce")
    evaluated: list[dict[str, object]] = []
    eligible_cells: list[pd.DataFrame] = []
    for model in PARITY_VALIDATED_RUNNERS:
        for feature in FEATURE_FAMILIES:
            subset = fixed_slice.loc[(fixed_slice["model"] == model) & (fixed_slice["feature"] == feature)].copy()
            subset = subset.set_index("dataset").reindex(EXPECTED_DISCOVERY).reset_index()
            present_all_three = bool(subset["model"].notna().all())
            rhos = subset[RHO_COLUMN].to_numpy(dtype=float)
            q_values = subset[Q_COLUMN].to_numpy(dtype=float)
            finite = bool(np.isfinite(rhos).all() and np.isfinite(q_values).all())
            q_passes = bool(finite and (q_values <= Q_THRESHOLD).all())
            signs = np.sign(rhos) if finite else np.array([np.nan, np.nan, np.nan])
            same_nonzero_sign = bool(finite and np.all(signs != 0.0) and np.all(signs == signs[0]))
            effect_passes = bool(finite and (np.abs(rhos) >= EFFECT_THRESHOLD).all())
            eligible = bool(present_all_three and q_passes and same_nonzero_sign and effect_passes)
            evaluated.append(
                {
                    "model": model,
                    "feature": feature,
                    "required_discovery_datasets": ";".join(EXPECTED_DISCOVERY),
                    "present_all_three_discovery_rows": present_all_three,
                    "finite_rho_and_q_all_three": finite,
                    "q_le_0_05_all_three": q_passes,
                    "identical_nonzero_sign_all_three": same_nonzero_sign,
                    "abs_rho_ge_0_05_all_three": effect_passes,
                    "eligible_by_locked_exploratory_rule": eligible,
                    "rho_by_dataset": json.dumps(
                        {dataset: _json_number(value) for dataset, value in zip(EXPECTED_DISCOVERY, rhos)}, sort_keys=True
                    ),
                    "q_by_dataset": json.dumps(
                        {dataset: _json_number(value) for dataset, value in zip(EXPECTED_DISCOVERY, q_values)}, sort_keys=True
                    ),
                }
            )
            if eligible:
                eligible_cells.append(subset[["dataset", "model", "view", "feature", "class_label"]].copy())
    evaluation = pd.DataFrame(evaluated)
    if eligible_cells:
        eligible_rows = pd.concat(eligible_cells, ignore_index=True)
    else:
        eligible_rows = pd.DataFrame(columns=["dataset", "model", "view", "feature", "class_label"])
    return evaluation, eligible_rows


def make_bootstrap_manifest(eligible_rows: pd.DataFrame, frozen_at_utc: str) -> pd.DataFrame:
    """Materialize exactly the bootstrap schema, with no extra selection fields."""
    manifest = eligible_rows.copy()
    for column, value in {
        "selection_status": "frozen",
        "selection_split": "discovery",
        "selection_basis": SELECTION_BASIS,
        "frozen_at_utc": frozen_at_utc,
    }.items():
        manifest[column] = value
    manifest = manifest.reindex(columns=CANDIDATE_MANIFEST_COLUMNS)
    if manifest.duplicated(KEY_COLUMNS).any():
        raise ValueError("Internal error: generated freeze manifest duplicates an exact H1 key.")
    return manifest.sort_values(["model", "feature", "dataset"], kind="stable").reset_index(drop=True)


def write_freeze_outputs(
    manifest: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    paths: Mapping[str, Path],
    config_path: Path,
    frozen_at_utc: str,
    manifest_path: Path,
    report_path: Path,
) -> None:
    """Write non-overwriting manifest/report artifacts with complete provenance."""
    if manifest_path.resolve() == report_path.resolve():
        raise ValueError("--output-manifest and --selection-report must be different paths.")
    if manifest_path.exists() or report_path.exists():
        existing = [str(path) for path in (manifest_path, report_path) if path.exists()]
        raise FileExistsError(f"Refusing to overwrite freeze artifact(s): {existing}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    report = {
        "artifact_kind": "exploratory_discovery_to_h2_candidate_freeze",
        "claim_guard": "Operational exploratory eligibility only; not a portable H1, causal H2, or confirmation claim.",
        "selection_basis": SELECTION_BASIS,
        "selection_basis_sha256": sha256_file(REPO_ROOT / SELECTION_BASIS),
        "frozen_at_utc": frozen_at_utc,
        "inputs": {
            dataset: {"path": str(paths[dataset]), "sha256": sha256_file(paths[dataset])}
            for dataset in EXPECTED_DISCOVERY
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "scope": {
            "datasets": list(EXPECTED_DISCOVERY),
            "parity_validated_runners": list(PARITY_VALIDATED_RUNNERS),
            "fixed_view": FULL_WAVEFORM_SPOOF["view"],
            "fixed_class_label": FULL_WAVEFORM_SPOOF["class_label"],
            "estimator": "partial_spearman",
            "q_threshold": Q_THRESHOLD,
            "absolute_rho_threshold": EFFECT_THRESHOLD,
            "feature_families": list(FEATURE_FAMILIES),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "row_count": int(len(manifest)),
            "schema": CANDIDATE_MANIFEST_COLUMNS,
            "eligible_feature_families_by_rule": [
                feature for feature in FEATURE_FAMILIES if feature in set(manifest["feature"])
            ],
        },
        "evaluation_matrix": evaluation.to_dict("records"),
        "confirmation_or_h2_data_read": False,
        "candidate_selection_ranked_or_top_k": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvspoof2019-la", required=True, type=Path)
    parser.add_argument("--asvspoof2021-la", required=True, type=Path)
    parser.add_argument("--asvspoof2021-df", required=True, type=Path)
    parser.add_argument("--frozen-at-utc", required=True, type=parse_utc_timestamp)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--config", default="configs/study.yaml", type=Path)
    args = parser.parse_args()
    inputs = {
        "ASVspoof2019_LA": args.asvspoof2019_la,
        "ASVspoof2021_LA": args.asvspoof2021_la,
        "ASVspoof2021_DF": args.asvspoof2021_df,
    }
    rows, paths, _panel = load_discovery_screens(inputs, args.config)
    evaluation, eligible_rows = evaluate_fixed_rule(rows)
    manifest = make_bootstrap_manifest(eligible_rows, args.frozen_at_utc)
    write_freeze_outputs(
        manifest,
        evaluation,
        paths=paths,
        config_path=args.config.resolve(),
        frozen_at_utc=args.frozen_at_utc,
        manifest_path=args.output_manifest,
        report_path=args.selection_report,
    )
    print(f"wrote {args.output_manifest} ({len(manifest)} frozen H1 cells)")
    print(f"wrote {args.selection_report}")


if __name__ == "__main__":
    main()
