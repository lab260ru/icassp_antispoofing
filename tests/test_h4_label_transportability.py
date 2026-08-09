"""Synthetic guardrails for the locked, score-free H4 transportability atlas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audio_features import FEATURE_NAMES
from src.h4_label_transportability import (
    BOOTSTRAP_REPLICATES,
    DATASETS,
    FORBIDDEN_TERMS,
    VIEWS,
    analyze_frozen_manifest,
    freeze_inputs,
    validate_explicit_input_paths,
)


def _synthetic_inputs(root: Path, *, n_sources_per_label: int = 3, add_response_field: bool = False) -> dict[str, Path]:
    root.mkdir()
    paths: dict[str, Path] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        rows: list[dict[str, object]] = []
        for label in (0, 1):
            for source_index in range(n_sources_per_label):
                source_id = f"{dataset}_label{label}_source{source_index}"
                sample_id = f"{dataset}_label{label}_sample{source_index}"
                for view_index, view in enumerate(VIEWS):
                    row: dict[str, object] = {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "label": label,
                        "view": view,
                    }
                    for feature_index, feature in enumerate(FEATURE_NAMES):
                        # Every registered feature has a simple, finite and
                        # naturally positive synthetic label separation.
                        row[feature] = float(
                            100.0 * label + 10.0 * dataset_index + view_index + feature_index / 100.0 + source_index / 1000.0
                        )
                    if add_response_field and dataset == DATASETS[0]:
                        row["response_score"] = 0.0
                    rows.append(row)
        path = root / dataset / "features_wide.parquet"
        path.parent.mkdir()
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths[dataset] = path
    return paths


def test_h4_firewall_rejects_unlocked_path_and_response_field(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", add_response_field=True)
    with pytest.raises(ValueError, match="response-like columns"):
        freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths)

    clean_paths = _synthetic_inputs(tmp_path / "clean_features")
    supplied = dict(clean_paths)
    supplied[DATASETS[0]] = tmp_path / "outside" / "features_wide.parquet"
    with pytest.raises(ValueError, match="path firewall"):
        validate_explicit_input_paths(supplied, allowed_paths=clean_paths)


def test_h4_freeze_rejects_changed_input_and_tampered_manifest_path(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features")
    artifacts = freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-09T00:00:00Z")

    # A changed feature product is detected from its freeze hash before the
    # analyzer can read projected feature values.
    paths[DATASETS[0]].write_bytes(b"changed-after-freeze")
    with pytest.raises(ValueError, match="input changed after freeze"):
        analyze_frozen_manifest(artifacts.provenance_path, tmp_path / "analysis_changed", allowed_paths=paths)

    # Restore by creating a separate valid freeze, then mutate only its JSON
    # path record. The analyzer must reject this before accepting an alias.
    clean_paths = _synthetic_inputs(tmp_path / "other_features")
    clean = freeze_inputs(clean_paths, tmp_path / "clean_freeze", allowed_paths=clean_paths, frozen_at_utc="2026-08-09T00:00:00Z")
    provenance = json.loads(clean.provenance_path.read_text())
    provenance["inputs"][0]["absolute_path"] = str(tmp_path / "detector_alias" / "features_wide.parquet")
    clean.provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="path firewall"):
        analyze_frozen_manifest(clean.provenance_path, tmp_path / "analysis_tampered", allowed_paths=clean_paths)


def test_h4_source_cap_is_hash_deterministic(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", n_sources_per_label=6)
    first = freeze_inputs(
        paths,
        tmp_path / "freeze_first",
        allowed_paths=paths,
        source_cap=2,
        frozen_at_utc="2026-08-09T00:00:00Z",
    )
    second = freeze_inputs(
        paths,
        tmp_path / "freeze_second",
        allowed_paths=paths,
        source_cap=2,
        frozen_at_utc="2026-08-09T00:00:00Z",
    )
    first_rows = pd.read_parquet(first.manifest_path).sort_values(["dataset", "label", "selection_rank"]).reset_index(drop=True)
    second_rows = pd.read_parquet(second.manifest_path).sort_values(["dataset", "label", "selection_rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_rows, second_rows)
    assert (first_rows.groupby(["dataset", "label"]).size() == 2).all()


def test_h4_analyzer_materializes_exhaustive_outputs_without_response_columns(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", n_sources_per_label=2)
    freeze = freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-09T00:00:00Z")
    matrix_path, aggregation_path, report_path = analyze_frozen_manifest(
        freeze.provenance_path,
        tmp_path / "analysis",
        allowed_paths=paths,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    report = json.loads(report_path.read_text())
    assert len(matrix) == len(DATASETS) * len(VIEWS) * len(FEATURE_NAMES) == 420
    assert len(aggregation) == len(VIEWS) * len(FEATURE_NAMES) == 84
    assert not matrix.duplicated(["dataset", "view", "feature"]).any()
    assert set(matrix["cell_status"]) == {"ok"}
    assert set(matrix["bootstrap_replicates_requested"]) == {BOOTSTRAP_REPLICATES}
    assert (matrix["bootstrap_replicates_valid"] == BOOTSTRAP_REPLICATES).all()
    assert np.isfinite(matrix["delta_auc"]).all()
    assert not any(term in column.casefold() for column in matrix.columns for term in FORBIDDEN_TERMS)
    assert not any(term in column.casefold() for column in aggregation.columns for term in FORBIDDEN_TERMS)
    assert report["score_free_classifier_free"] is True
    assert report["n_dataset_view_feature_cells"] == 420
    assert report["n_view_feature_aggregation_units"] == 84

