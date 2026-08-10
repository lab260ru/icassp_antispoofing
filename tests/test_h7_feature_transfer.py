from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.h7_feature_transfer as h7
from src.audio_features import FEATURE_NAMES
from src.h7_feature_transfer import CORPORA, FREEZE_COLUMNS, _eer, freeze_inputs, run_analysis, write_analysis, write_freeze


def _fixture_paths(tmp_path: Path, *, rows_per_label: int = 20, bad_column: str | None = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for corpus_index, corpus in enumerate(CORPORA):
        rows = []
        for label in (0, 1):
            for index in range(rows_per_label):
                base = float(label * 2 + corpus_index * 0.1 + index / 100000)
                features = {name: base + feature_index / 1000 for feature_index, name in enumerate(FEATURE_NAMES)}
                if index == 0:
                    features["f0_median_hz"] = np.nan
                for view in ("full_waveform", "deterministic_crop", "preemphasized_crop"):
                    rows.append(
                        {
                            "sample_id": f"{corpus}-{label}-{index}",
                            "source_id": f"{corpus}-source-{label}-{index}",
                            "label": label,
                            "view": view,
                            **features,
                        }
                    )
        frame = pd.DataFrame(rows)
        if bad_column:
            frame[bad_column] = 0.0
        path = tmp_path / corpus / "features_wide.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        paths[corpus] = path
    return paths


def test_freeze_rejects_unlocked_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked seed and cap"):
        freeze_inputs(_fixture_paths(tmp_path), max_per_label=4)


def test_production_bootstrap_constant_is_locked() -> None:
    assert h7.BOOTSTRAP_REPLICATES == 500


def test_eer_handles_vertical_roc_crossing() -> None:
    # The FPR/FNR difference changes sign across two positive-score thresholds
    # with the same FPR; linear interpolation must return that shared FPR.
    labels = np.array([0, 1, 1, 1, 0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert _eer(labels, scores) == pytest.approx(0.5)


def test_freeze_rejects_response_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h7, "MAX_PER_LABEL", 20)
    with pytest.raises(ValueError, match="response-like source columns"):
        freeze_inputs(_fixture_paths(tmp_path, bad_column="score_spoof"), max_per_label=20)


def test_freeze_and_analysis_complete_locked_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Production uses a 5,000-per-class panel and 500 bootstrap replicates.
    # This synthetic integration test monkeypatches both module constants after
    # asserting the production constants above, exercising the complete five
    # fitted-cell path without fitting a 40k-row model in every test.
    monkeypatch.setattr(h7, "MAX_PER_LABEL", 20)
    monkeypatch.setattr(h7, "BOOTSTRAP_REPLICATES", 3)
    paths = _fixture_paths(tmp_path / "inputs")
    manifest, provenance = freeze_inputs(paths, max_per_label=20)
    assert tuple(manifest.columns) == FREEZE_COLUMNS
    assert len(manifest) == 200
    freeze_dir = tmp_path / "freeze"
    write_freeze(freeze_dir, manifest, provenance)
    table, summary = run_analysis(freeze_dir)
    assert table["held_out_dataset"].tolist() == list(CORPORA)
    assert len(table) == 5
    assert table["n_train"].eq(160).all()
    assert table["n_test"].eq(40).all()
    assert table["fit_converged"].all()
    assert table["auroc"].between(0.0, 1.0).all()
    assert table["bootstrap_replicates_valid"].eq(3).all()
    assert summary["representation"] == "all_28_full_waveform_features"


def test_freeze_and_analysis_outputs_refuse_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h7, "MAX_PER_LABEL", 20)
    monkeypatch.setattr(h7, "BOOTSTRAP_REPLICATES", 3)
    paths = _fixture_paths(tmp_path / "inputs")
    manifest, provenance = freeze_inputs(paths, max_per_label=20)
    freeze_dir = tmp_path / "freeze"
    write_freeze(freeze_dir, manifest, provenance)
    with pytest.raises(FileExistsError, match="Refusing to overwrite H7 input freeze"):
        write_freeze(freeze_dir, manifest, provenance)
    table, summary = run_analysis(freeze_dir)
    analysis_dir = tmp_path / "analysis"
    write_analysis(analysis_dir, table, summary)
    assert len(pd.read_csv(analysis_dir / "h7_leave_one_corpus_out.csv")) == 5
    loaded = json.loads((analysis_dir / "h7_analysis_provenance.json").read_text())
    assert loaded["complete_matrix_cells"] == 5
    with pytest.raises(FileExistsError, match="Refusing to overwrite H7 analysis output"):
        write_analysis(analysis_dir, table, summary)


def test_analysis_rejects_modified_input_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h7, "MAX_PER_LABEL", 20)
    paths = _fixture_paths(tmp_path / "inputs")
    manifest, provenance = freeze_inputs(paths, max_per_label=20)
    freeze_dir = tmp_path / "freeze"
    write_freeze(freeze_dir, manifest, provenance)
    changed = pd.read_parquet(paths["ASVspoof5"])
    changed.loc[0, "crest_factor_db"] = 123.0
    changed.to_parquet(paths["ASVspoof5"], index=False)
    with pytest.raises(ValueError, match="source hash mismatch"):
        run_analysis(freeze_dir)
