from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_associations import CANDIDATE_MANIFEST_COLUMNS
from scripts.freeze_discovery_to_h2_candidates import (
    EXPECTED_DISCOVERY,
    FEATURE_FAMILIES,
    PARITY_VALIDATED_RUNNERS,
    evaluate_fixed_rule,
    load_configured_panel,
    load_discovery_screens,
    make_bootstrap_manifest,
    parse_utc_timestamp,
    write_freeze_outputs,
)


def _write_screen(path: Path, dataset: str, panel: tuple[str, ...]) -> None:
    rows: list[dict[str, object]] = []
    for model in panel:
        for feature in FEATURE_FAMILIES:
            rho, q_value = 0.01, 0.50
            if model == "Spectra-AASIST" and feature == "crest_factor_db":
                rho, q_value = 0.20, 0.01
            elif model == "AASIST" and feature == "silence_fraction":
                rho, q_value = -0.12, 0.02
            elif feature == "spectral_slope_db_per_khz":
                rho = 0.12 if dataset != "ASVspoof2021_LA" else -0.12
            elif feature == "group_delay_var":
                rho, q_value = 0.15, 0.20
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "view": "full_waveform",
                    "feature": feature,
                    "class_label": 1,
                    "analysis_stage": "screen",
                    "partial_spearman_rho": rho,
                    "partial_spearman_q": q_value,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _inputs(tmp_path: Path, panel: tuple[str, ...]) -> dict[str, Path]:
    inputs = {}
    for dataset in EXPECTED_DISCOVERY:
        path = tmp_path / f"{dataset}.csv"
        _write_screen(path, dataset, panel)
        inputs[dataset] = path
    return inputs


def test_fixed_rule_writes_exact_manifest_schema_and_provenance(tmp_path: Path) -> None:
    panel = load_configured_panel("configs/study.yaml")
    inputs = _inputs(tmp_path, panel)
    rows, paths, _ = load_discovery_screens(inputs, "configs/study.yaml")
    evaluation, eligible_rows = evaluate_fixed_rule(rows)
    manifest = make_bootstrap_manifest(eligible_rows, "2026-08-10T12:00:00Z")
    manifest_path = tmp_path / "frozen.csv"
    report_path = tmp_path / "freeze_report.json"
    write_freeze_outputs(
        manifest,
        evaluation,
        paths=paths,
        config_path=Path("configs/study.yaml").resolve(),
        frozen_at_utc="2026-08-10T12:00:00Z",
        manifest_path=manifest_path,
        report_path=report_path,
    )

    written = pd.read_csv(manifest_path)
    assert written.columns.tolist() == CANDIDATE_MANIFEST_COLUMNS
    assert len(written) == 6
    assert set(written["feature"]) == {"crest_factor_db", "silence_fraction"}
    assert set(written["model"]) == set(PARITY_VALIDATED_RUNNERS)
    assert set(written["selection_status"]) == {"frozen"}
    assert set(written["selection_split"]) == {"discovery"}
    report = json.loads(report_path.read_text())
    assert report["confirmation_or_h2_data_read"] is False
    assert report["candidate_selection_ranked_or_top_k"] is False
    assert report["manifest"]["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert evaluation["eligible_by_locked_exploratory_rule"].sum() == 2


def test_rule_reports_no_eligible_family_without_ranking(tmp_path: Path) -> None:
    panel = load_configured_panel("configs/study.yaml")
    inputs = _inputs(tmp_path, panel)
    for path in inputs.values():
        table = pd.read_csv(path)
        table["partial_spearman_q"] = 0.5
        table.to_csv(path, index=False)
    rows, _paths, _ = load_discovery_screens(inputs, "configs/study.yaml")
    evaluation, eligible_rows = evaluate_fixed_rule(rows)
    manifest = make_bootstrap_manifest(eligible_rows, "2026-08-10T12:00:00Z")
    assert manifest.columns.tolist() == CANDIDATE_MANIFEST_COLUMNS
    assert manifest.empty
    assert evaluation["eligible_by_locked_exploratory_rule"].sum() == 0


def test_freeze_rejects_reused_paths_duplicate_keys_and_bad_timestamp(tmp_path: Path) -> None:
    panel = load_configured_panel("configs/study.yaml")
    inputs = _inputs(tmp_path, panel)
    reused = dict(inputs)
    reused["ASVspoof2021_LA"] = reused["ASVspoof2019_LA"]
    with pytest.raises(ValueError, match="duplicate path assignments"):
        load_discovery_screens(reused, "configs/study.yaml")

    table = pd.read_csv(inputs["ASVspoof2021_DF"])
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    table.to_csv(inputs["ASVspoof2021_DF"], index=False)
    with pytest.raises(ValueError, match="duplicates exact H1 association keys"):
        load_discovery_screens(inputs, "configs/study.yaml")

    with pytest.raises(ValueError, match="ending in 'Z'"):
        parse_utc_timestamp("2026-08-10T12:00:00+03:00")
