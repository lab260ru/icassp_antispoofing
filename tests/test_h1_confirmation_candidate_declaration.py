from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_associations import CANDIDATE_MANIFEST_COLUMNS
from scripts.declare_h1_confirmation_candidates import (
    DISCOVERY_DATASETS,
    load_discovery_freeze,
    make_confirmation_manifest,
    parse_utc_timestamp,
    validate_confirmation_datasets,
    write_declaration_outputs,
)


def _write_discovery_freeze(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for dataset in DISCOVERY_DATASETS:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "model": "Spectra-AASIST",
                    "view": "full_waveform",
                    "feature": "crest_factor_db",
                    "class_label": 1,
                    "selection_status": "frozen",
                    "selection_split": "discovery",
                    "selection_basis": "synthetic locked decision",
                    "frozen_at_utc": "2026-08-09T20:17:11Z",
                },
                {
                    "dataset": dataset,
                    "model": "AASIST",
                    "view": "full_waveform",
                    "feature": "silence_fraction",
                    "class_label": 1,
                    "selection_status": "frozen",
                    "selection_split": "discovery",
                    "selection_basis": "synthetic locked decision",
                    "frozen_at_utc": "2026-08-09T20:17:11Z",
                },
            ]
        )
    pd.DataFrame(rows, columns=CANDIDATE_MANIFEST_COLUMNS).to_csv(path, index=False)


def test_declaration_projects_exact_frozen_identities_and_records_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "discovery_freeze.csv"
    _write_discovery_freeze(source)
    candidates = load_discovery_freeze(source)
    targets = validate_confirmation_datasets(["InTheWild", "ASVspoof5"])
    manifest, identities = make_confirmation_manifest(candidates, targets)
    output = tmp_path / "heldout.csv"
    report_path = tmp_path / "heldout.provenance.json"
    write_declaration_outputs(
        manifest,
        identities,
        source_manifest=source,
        source_candidates=candidates,
        targets=targets,
        declared_at_utc="2026-08-09T21:00:00Z",
        output_manifest=output,
        provenance_report=report_path,
    )

    written = pd.read_csv(output)
    assert written.columns.tolist() == CANDIDATE_MANIFEST_COLUMNS
    assert len(written) == 4
    assert set(written["dataset"]) == {"InTheWild", "ASVspoof5"}
    assert set(written["model"]) == {"Spectra-AASIST", "AASIST"}
    assert set(written["selection_status"]) == {"frozen"}
    assert set(written["selection_split"]) == {"discovery"}
    assert set(written["frozen_at_utc"]) == {"2026-08-09T20:17:11Z"}

    report = json.loads(report_path.read_text())
    assert report["construction"]["reads_confirmation_association_or_bootstrap_output"] is False
    assert report["construction"]["reads_confirmation_features_scores_or_audio"] is False
    assert report["construction"]["ranks_filters_or_reselects_candidates"] is False
    assert report["source_discovery_manifest"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["manifest"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(report["source_candidate_identities"]) == 2


def test_declaration_rejects_incomplete_sources_duplicate_targets_and_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "discovery_freeze.csv"
    _write_discovery_freeze(source)
    incomplete = pd.read_csv(source).loc[lambda table: table["dataset"] != "ASVspoof2021_DF"]
    incomplete.to_csv(source, index=False)
    with pytest.raises(ValueError, match="exactly the three registered discovery datasets"):
        load_discovery_freeze(source)
    with pytest.raises(ValueError, match="at most once"):
        validate_confirmation_datasets(["InTheWild", "InTheWild"])
    with pytest.raises(ValueError, match="ending in 'Z'"):
        parse_utc_timestamp("2026-08-09T21:00:00+00:00")

    _write_discovery_freeze(source)
    candidates = load_discovery_freeze(source)
    manifest, identities = make_confirmation_manifest(candidates, ("InTheWild",))
    output = tmp_path / "heldout.csv"
    report_path = tmp_path / "heldout.provenance.json"
    output.write_text("already here\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_declaration_outputs(
            manifest,
            identities,
            source_manifest=source,
            source_candidates=candidates,
            targets=("InTheWild",),
            declared_at_utc="2026-08-09T21:00:00Z",
            output_manifest=output,
            provenance_report=report_path,
        )
