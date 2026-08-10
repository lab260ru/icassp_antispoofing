"""Synthetic guardrails for the locked, score- and label-free H5 atlas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.h5_view_invariance as h5
from src.audio_features import FEATURE_NAMES


def _synthetic_inputs(
    root: Path,
    *,
    n_samples: int = 3,
    add_response_field: bool = False,
    omit_one_view: bool = False,
    constant_first_feature: bool = False,
) -> dict[str, Path]:
    """Write isolated feature containers; no real corpus product is touched."""
    root.mkdir()
    paths: dict[str, Path] = {}
    for dataset_index, dataset in enumerate(h5.DATASETS):
        rows: list[dict[str, object]] = []
        for sample_index in range(n_samples):
            for view_index, view in enumerate(h5.VIEWS):
                if omit_one_view and dataset == h5.DATASETS[0] and sample_index == 0 and view == h5.VIEWS[-1]:
                    continue
                row: dict[str, object] = {
                    "sample_id": f"{dataset}_sample_{sample_index}",
                    "view": view,
                    # These raw-container metadata fields intentionally prove
                    # that H5's exact projection, rather than a separate
                    # relabelled input, is the label/source firewall.
                    "label": sample_index % 2,
                    "source_id": f"{dataset}_source_{sample_index}",
                }
                for feature_index, feature in enumerate(FEATURE_NAMES):
                    row[feature] = (
                        1.0
                        if constant_first_feature and feature == FEATURE_NAMES[0]
                        else float(100.0 * dataset_index + 10.0 * sample_index + view_index + feature_index / 100.0)
                    )
                if add_response_field and dataset == h5.DATASETS[0]:
                    row["published_score"] = 0.0
                rows.append(row)
        path = root / dataset / "features_wide.parquet"
        path.parent.mkdir()
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths[dataset] = path
    return paths


def test_h5_freeze_projects_identity_only_and_never_emits_label_or_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _synthetic_inputs(tmp_path / "features")
    read_calls: list[tuple[str, ...]] = []
    original_read = h5.pd.read_parquet

    def tracked_read(*args: object, **kwargs: object) -> pd.DataFrame:
        columns = kwargs.get("columns")
        if columns is not None:
            read_calls.append(tuple(columns))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(h5.pd, "read_parquet", tracked_read)
    frozen = h5.freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    assert read_calls and set(read_calls) == {h5.IDENTITY_COLUMNS}
    manifest = pd.read_parquet(frozen.manifest_path)
    provenance = json.loads(frozen.provenance_path.read_text())
    assert manifest.columns.tolist() == ["dataset", "sample_id", "selection_rank", "selection_key_sha256"]
    assert provenance["freeze_projection_columns"] == list(h5.IDENTITY_COLUMNS)
    assert provenance["analysis_projection_columns"] == list(h5.ALLOWED_COLUMNS)
    flattened = json.dumps({"manifest_columns": manifest.columns.tolist(), "projections": provenance["analysis_projection_columns"]}).casefold()
    assert "label" not in flattened and "source_id" not in flattened


def test_h5_firewall_rejects_response_like_header_and_unlocked_path(tmp_path: Path) -> None:
    unsafe = _synthetic_inputs(tmp_path / "unsafe", add_response_field=True)
    with pytest.raises(ValueError, match="source-column firewall"):
        h5.freeze_inputs(unsafe, tmp_path / "freeze", allowed_paths=unsafe)

    clean = _synthetic_inputs(tmp_path / "clean")
    supplied = dict(clean)
    supplied[h5.DATASETS[0]] = tmp_path / "outside" / "features_wide.parquet"
    with pytest.raises(ValueError, match="path firewall"):
        h5.validate_explicit_input_paths(supplied, allowed_paths=clean)


def test_h5_selection_is_seed_2610_hash_deterministic_and_independent(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", n_samples=6)
    first = h5.freeze_inputs(paths, tmp_path / "freeze_first", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    second = h5.freeze_inputs(paths, tmp_path / "freeze_second", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    first_rows = pd.read_parquet(first.manifest_path).sort_values(["dataset", "selection_rank"]).reset_index(drop=True)
    second_rows = pd.read_parquet(second.manifest_path).sort_values(["dataset", "selection_rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_rows, second_rows)
    assert h5.SEED == 2610
    assert not any("h4" in value.casefold() or "label" in value.casefold() for value in first_rows.columns)
    assert (first_rows.groupby("dataset").size() == 6).all()


def test_h5_rejects_changed_input_and_tampered_manifest_before_feature_read(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features")
    frozen = h5.freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    paths[h5.DATASETS[0]].write_bytes(b"changed-after-freeze")
    with pytest.raises(ValueError, match="input changed after freeze"):
        h5.analyze_frozen_manifest(frozen.provenance_path, tmp_path / "analysis_changed", allowed_paths=paths)

    clean = _synthetic_inputs(tmp_path / "clean")
    clean_freeze = h5.freeze_inputs(clean, tmp_path / "clean_freeze", allowed_paths=clean, frozen_at_utc="2026-08-10T00:00:00Z")
    provenance = json.loads(clean_freeze.provenance_path.read_text())
    provenance["selection_manifest"]["absolute_path"] = str(tmp_path / "detector_alias" / "manifest.parquet")
    clean_freeze.provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="manifest path"):
        h5.analyze_frozen_manifest(clean_freeze.provenance_path, tmp_path / "analysis_tampered", allowed_paths=clean)


def test_h5_analyzer_materializes_complete_matrix_with_derived_seeds(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", n_samples=3)
    frozen = h5.freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    matrix_path, aggregation_path, report_path = h5.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_paths=paths,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    report = json.loads(report_path.read_text())
    assert h5.VIEW_PAIRS == (
        ("deterministic_crop", "full_waveform"),
        ("deterministic_crop", "preemphasized_crop"),
        ("full_waveform", "preemphasized_crop"),
    )
    assert len(matrix) == len(h5.DATASETS) * len(h5.VIEW_PAIRS) * len(FEATURE_NAMES) == 420
    assert len(aggregation) == len(h5.VIEW_PAIRS) * len(FEATURE_NAMES) == 84
    assert not matrix.duplicated(["dataset", "view_a", "view_b", "feature"]).any()
    assert set(matrix["cell_status"]) == {"ok"}
    assert set(matrix["bootstrap_replicates_requested"]) == {h5.BOOTSTRAP_REPLICATES}
    assert (matrix["bootstrap_replicates_valid_concordance"] > 0).all()
    assert (matrix["bootstrap_replicates_valid_concordance"] <= h5.BOOTSTRAP_REPLICATES).all()
    assert (matrix["bootstrap_replicates_valid_shift"] == h5.BOOTSTRAP_REPLICATES).all()
    assert matrix["bootstrap_seed_sha256"].nunique() == len(matrix)
    assert np.isfinite(matrix["spearman_concordance"]).all()
    assert not any(term in column.casefold() for column in matrix.columns for term in h5.FORBIDDEN_OUTPUT_TERMS)
    assert not any(term in column.casefold() for column in aggregation.columns for term in h5.FORBIDDEN_OUTPUT_TERMS)
    assert report["score_and_label_free"] is True
    assert report["n_dataset_view_pair_feature_cells"] == 420
    assert report["n_view_pair_feature_aggregation_units"] == 84
    assert report["matrix_complete"] is True
    assert report["analysis_status"] == "complete"


def test_h5_reports_explicit_failed_cells_without_relaxing_matrix(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", n_samples=3, constant_first_feature=True)
    frozen = h5.freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    matrix_path, aggregation_path, _ = h5.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_paths=paths,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    failed = matrix.loc[matrix["feature"].eq(FEATURE_NAMES[0])]
    assert len(matrix) == 420 and len(aggregation) == 84
    assert len(failed) == len(h5.DATASETS) * len(h5.VIEW_PAIRS)
    assert set(failed["cell_status"]) == {"failed_zero_or_nonfinite_pooled_iqr"}


def test_h5_incomplete_freeze_materializes_failed_complete_matrix(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path / "features", n_samples=3, omit_one_view=True)
    frozen = h5.freeze_inputs(paths, tmp_path / "freeze", allowed_paths=paths, frozen_at_utc="2026-08-10T00:00:00Z")
    provenance = json.loads(frozen.provenance_path.read_text())
    assert provenance["freeze_status"] == "failed_input_integrity"
    matrix_path, aggregation_path, report_path = h5.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_paths=paths,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    report = json.loads(report_path.read_text())
    affected = matrix.loc[matrix["dataset"].eq(h5.DATASETS[0])]
    assert len(matrix) == 420 and len(aggregation) == 84
    assert set(affected["cell_status"]) == {"failed_input_incomplete_sample_view_set"}
    assert report["failed_cell_count"] == len(affected)
    assert report["analysis_status"] == "failed_cells_present"
