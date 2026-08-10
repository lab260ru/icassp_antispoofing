from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.h1_covariate_availability import CORPORA, locked_paths, validate_and_summarize, write_outputs


def _write_fixture(path: Path, *, bad_column: str | None = None, duplicate_full: bool = False) -> None:
    rows = []
    for label in (0, 1):
        sample_id = f"sample-{label}"
        for view in ("full_waveform", "deterministic_crop", "preemphasized_crop"):
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_id": f"source-{label}",
                    "label": label,
                    "view": view,
                    "duration_seconds": 1.0 + label,
                    "integrated_lufs": -20.0 - label,
                    "speaker_id": f"speaker-{label}",
                    "attack_id": "" if label == 0 else "attack-a",
                    "unused_feature": 99.0,
                }
            )
    if duplicate_full:
        rows.append(rows[0].copy())
    frame = pd.DataFrame(rows)
    if bad_column:
        frame[bad_column] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for corpus in CORPORA:
        path = tmp_path / corpus / "features_wide.parquet"
        _write_fixture(path)
        paths[corpus] = path
    return paths


def test_locked_paths_are_exact() -> None:
    root = Path("/tmp/root")
    paths = locked_paths(root)
    assert tuple(paths) == CORPORA
    assert all(path == root / corpus / "features_wide.parquet" for corpus, path in paths.items())


def test_summary_projects_metadata_and_records_coverage(tmp_path: Path) -> None:
    table, provenance = validate_and_summarize(_paths(tmp_path))
    assert len(table) == 10
    assert set(table["label"]) == {0, 1}
    assert table["duration_seconds_finite_n"].eq(1).all()
    assert table["attack_id_nonempty_n"].sum() == 5
    assert provenance["forbidden_operations"] == [
        "score_or_model_read",
        "feature_value_read",
        "association_or_p_value",
        "bootstrap_rerun",
        "candidate_or_intervention_selection",
    ]


def test_rejects_response_like_schema(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_fixture(paths["ASVspoof5"], bad_column="score_spoof")
    with pytest.raises(ValueError, match="Response-like source columns"):
        validate_and_summarize(paths)


def test_rejects_duplicate_full_waveform_ids(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_fixture(paths["InTheWild"], duplicate_full=True)
    with pytest.raises(ValueError, match="Duplicate full_waveform"):
        validate_and_summarize(paths)


def test_outputs_are_immutable_and_documented(tmp_path: Path) -> None:
    table, provenance = validate_and_summarize(_paths(tmp_path / "inputs"))
    output = tmp_path / "output"
    write_outputs(output, table, provenance)
    assert len(pd.read_csv(output / "h1_covariate_availability.csv")) == 10
    loaded = json.loads((output / "h1_covariate_availability_provenance.json").read_text())
    assert loaded["confirmation_bootstrap_context"]["replicates"] == 2000
    assert "No bootstrap was rerun" in (output / "H1_COVARIATE_AVAILABILITY_AUDIT_001.md").read_text()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_outputs(output, table, provenance)
