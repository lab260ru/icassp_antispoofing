"""Synthetic-only tests for the sealed H5 supplementary figure renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.h5_view_invariance_heatmap as h5_plot
from src.audio_features import FEATURE_NAMES
from src.h5_view_invariance import DATASETS, H5_VERSION, VIEW_PAIRS


def _sha256(path: Path) -> str:
    return h5_plot.sha256_file(path)


def _write_provenance(root: Path, matrix: Path, aggregation: Path, *, failed_cells: int) -> Path:
    report = {
        "h5_version": H5_VERSION,
        "matrix_complete": True,
        "analysis_status": "complete" if failed_cells == 0 else "failed_cells_present",
        "n_dataset_view_pair_feature_cells": 420,
        "n_view_pair_feature_aggregation_units": 84,
        "failed_cell_count": failed_cells,
        "matrix": {
            "absolute_path": str(matrix.resolve()),
            "sha256": _sha256(matrix),
            "byte_size": matrix.stat().st_size,
        },
        "aggregation": {
            "absolute_path": str(aggregation.resolve()),
            "sha256": _sha256(aggregation),
            "byte_size": aggregation.stat().st_size,
        },
    }
    provenance = root / h5_plot.PROVENANCE_FILENAME
    provenance.write_text(json.dumps(report), encoding="utf-8")
    return provenance


def _sealed_h5_artifacts(
    root: Path,
    *,
    one_unavailable: bool = False,
    extra_matrix_column: str | None = None,
) -> tuple[Path, Path]:
    """Create a compact H5-shaped fixture without an H5 input or result run."""
    root.mkdir()
    matrix_rows: list[dict[str, object]] = []
    aggregation_rows: list[dict[str, object]] = []
    unavailable_key = (VIEW_PAIRS[0][0], VIEW_PAIRS[0][1], FEATURE_NAMES[0], DATASETS[0])
    failed_count = 0
    for pair_index, (view_a, view_b) in enumerate(VIEW_PAIRS):
        for feature_index, feature in enumerate(FEATURE_NAMES):
            ok_values: list[float] = []
            shifts: list[float] = []
            ci_lows: list[float] = []
            shift_ci_lows: list[float] = []
            shift_ci_highs: list[float] = []
            for dataset_index, dataset in enumerate(DATASETS):
                unavailable = one_unavailable and (view_a, view_b, feature, dataset) == unavailable_key
                stable_candidate = feature_index % 9 == 0
                concordance = 0.95 + dataset_index / 1000.0 if stable_candidate else 0.62 + dataset_index / 100.0
                shift = 0.02 if stable_candidate else 0.16
                ci_low = 0.88 if stable_candidate else 0.50
                shift_low = -0.04 if stable_candidate else 0.11
                shift_high = 0.04 if stable_candidate else 0.21
                status = "failed_insufficient_finite_pairs" if unavailable else "ok"
                if unavailable:
                    failed_count += 1
                else:
                    ok_values.append(concordance)
                    shifts.append(shift)
                    ci_lows.append(ci_low)
                    shift_ci_lows.append(shift_low)
                    shift_ci_highs.append(shift_high)
                matrix_rows.append(
                    {
                        "dataset": dataset,
                        "view_a": view_a,
                        "view_b": view_b,
                        "feature": feature,
                        "cell_status": status,
                        "paired_finite_sample_count": 10 if not unavailable else 0,
                        "spearman_concordance": concordance if not unavailable else np.nan,
                        "iqr_normalized_median_shift": shift if not unavailable else np.nan,
                        "bootstrap_replicates_requested": 200,
                        "bootstrap_replicates_valid_concordance": 200 if not unavailable else 0,
                        "bootstrap_replicates_valid_shift": 200 if not unavailable else 0,
                        "bootstrap_replicates_invalid_concordance": 0 if not unavailable else 200,
                        "bootstrap_replicates_invalid_shift": 0 if not unavailable else 200,
                        "bootstrap_confidence": 0.95,
                        "concordance_ci_low": ci_low if not unavailable else np.nan,
                        "concordance_ci_high": min(concordance + 0.01, 1.0) if not unavailable else np.nan,
                        "normalized_shift_ci_low": shift_low if not unavailable else np.nan,
                        "normalized_shift_ci_high": shift_high if not unavailable else np.nan,
                        "bootstrap_seed": 1000 + pair_index * 100 + feature_index * 5 + dataset_index,
                        "bootstrap_seed_sha256": f"{pair_index:02x}{feature_index:02x}{dataset_index:02x}".ljust(64, "0"),
                    }
                )
            all_ok = len(ok_values) == len(DATASETS)
            stable = bool(
                all_ok
                and min(ok_values) >= 0.90
                and min(ci_lows) >= 0.80
                and min(shift_ci_lows) >= -0.10
                and max(shift_ci_highs) <= 0.10
            )
            aggregation_rows.append(
                {
                    "view_a": view_a,
                    "view_b": view_b,
                    "feature": feature,
                    "dataset_cell_count": len(DATASETS),
                    "successful_dataset_cell_count": len(ok_values),
                    "missing_datasets": "",
                    "all_dataset_cells_ok": all_ok,
                    "median_spearman_concordance": float(np.median(ok_values)) if ok_values else np.nan,
                    "median_absolute_iqr_normalized_shift": float(np.median(np.abs(shifts))) if shifts else np.nan,
                    "all_concordance_at_least_0p90": bool(all_ok and min(ok_values) >= 0.90),
                    "all_concordance_ci_lower_at_least_0p80": bool(all_ok and min(ci_lows) >= 0.80),
                    "all_shift_intervals_within_plus_minus_0p10": bool(all_ok and min(shift_ci_lows) >= -0.10 and max(shift_ci_highs) <= 0.10),
                    "view_stable_terminal_descriptive": stable,
                    "aggregation_status": "view_stable_terminal_descriptive" if stable else "does_not_meet_view_stable_rule",
                }
            )
    matrix = root / h5_plot.MATRIX_FILENAME
    aggregation = root / h5_plot.AGGREGATION_FILENAME
    matrix_table = pd.DataFrame(matrix_rows, columns=h5_plot.MATRIX_COLUMNS)
    if extra_matrix_column is not None:
        matrix_table[extra_matrix_column] = 0
    matrix_table.to_csv(matrix, index=False)
    pd.DataFrame(aggregation_rows, columns=h5_plot.AGGREGATION_COLUMNS).to_csv(aggregation, index=False)
    _write_provenance(root, matrix, aggregation, failed_cells=failed_count)
    return matrix, aggregation


def _allowed(matrix: Path, aggregation: Path) -> dict[str, Path]:
    return {"matrix": matrix.resolve(), "aggregation": aggregation.resolve()}


def test_h5_figure_generates_hash_bound_vector_png_and_metadata(tmp_path: Path) -> None:
    matrix, aggregation = _sealed_h5_artifacts(tmp_path / "sealed")
    with pytest.raises(ValueError, match="literal h5_analysis_001"):
        h5_plot.load_sealed_h5_figure_inputs(matrix.resolve(), aggregation.resolve())
    pdf_path, png_path, metadata_path = h5_plot._render_h5_median_concordance_heatmap(
        matrix.resolve(), aggregation.resolve(), tmp_path / "figure", allowed_paths=_allowed(matrix, aggregation)
    )
    assert pdf_path.name == h5_plot.PDF_FILENAME and pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.name == h5_plot.PNG_FILENAME and png_path.stat().st_size > 0
    with Image.open(png_path) as image:
        assert image.size[0] > 1_000 and image.size[1] > 1_000
        assert image.info["dpi"][0] >= 299
    metadata = json.loads(metadata_path.read_text())
    assert metadata["matrix"]["sha256"] == _sha256(matrix)
    assert metadata["aggregation"]["sha256"] == _sha256(aggregation)
    assert metadata["color_scale"] == {"minimum": 0.0, "maximum": 1.0, "negative_underflow": "shown at lower scale endpoint"}
    assert metadata["display_counts"]["unavailable_aggregation_units"] == 0
    assert metadata["display_counts"]["terminal_descriptive_units"] > 0


def test_h5_figure_accepts_explicit_unavailable_unit_without_imputation(tmp_path: Path) -> None:
    matrix, aggregation = _sealed_h5_artifacts(tmp_path / "sealed", one_unavailable=True)
    pdf_path, png_path, metadata_path = h5_plot._render_h5_median_concordance_heatmap(
        matrix.resolve(), aggregation.resolve(), tmp_path / "figure", allowed_paths=_allowed(matrix, aggregation)
    )
    assert pdf_path.exists() and png_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["display_counts"]["unavailable_aggregation_units"] == 1


def test_h5_figure_rejects_hash_drift_and_response_like_or_schema_headers(tmp_path: Path) -> None:
    matrix, aggregation = _sealed_h5_artifacts(tmp_path / "sealed")
    matrix.write_text(matrix.read_text() + "\n")
    with pytest.raises(ValueError, match="byte-size mismatch|hash changed"):
        h5_plot._load_sealed_h5_figure_inputs(matrix.resolve(), aggregation.resolve(), allowed_paths=_allowed(matrix, aggregation))

    response_matrix, response_aggregation = _sealed_h5_artifacts(tmp_path / "response", extra_matrix_column="response_score")
    with pytest.raises(ValueError, match="response-like"):
        h5_plot._load_sealed_h5_figure_inputs(
            response_matrix.resolve(), response_aggregation.resolve(), allowed_paths=_allowed(response_matrix, response_aggregation)
        )

    aggregation_response_matrix, aggregation_response = _sealed_h5_artifacts(tmp_path / "aggregation_response")
    aggregation_table = pd.read_csv(aggregation_response)
    aggregation_table["model_response"] = 0
    aggregation_table.to_csv(aggregation_response, index=False)
    _write_provenance(aggregation_response.parent, aggregation_response_matrix, aggregation_response, failed_cells=0)
    with pytest.raises(ValueError, match="response-like"):
        h5_plot._load_sealed_h5_figure_inputs(
            aggregation_response_matrix.resolve(), aggregation_response.resolve(), allowed_paths=_allowed(aggregation_response_matrix, aggregation_response)
        )

    schema_matrix, schema_aggregation = _sealed_h5_artifacts(tmp_path / "schema", extra_matrix_column="benign_extra")
    with pytest.raises(ValueError, match="schema"):
        h5_plot._load_sealed_h5_figure_inputs(
            schema_matrix.resolve(), schema_aggregation.resolve(), allowed_paths=_allowed(schema_matrix, schema_aggregation)
        )


def test_h5_figure_rejects_aggregation_mismatch_and_unlocked_path(tmp_path: Path) -> None:
    matrix, aggregation = _sealed_h5_artifacts(tmp_path / "sealed")
    table = pd.read_csv(aggregation)
    table.loc[0, "median_spearman_concordance"] = 0.123
    table.to_csv(aggregation, index=False)
    _write_provenance(aggregation.parent, matrix, aggregation, failed_cells=0)
    with pytest.raises(ValueError, match="does not match"):
        h5_plot._load_sealed_h5_figure_inputs(matrix.resolve(), aggregation.resolve(), allowed_paths=_allowed(matrix, aggregation))

    outside = tmp_path / "outside" / h5_plot.MATRIX_FILENAME
    outside.parent.mkdir()
    outside.write_bytes(matrix.read_bytes())
    with pytest.raises(ValueError, match="literal"):
        h5_plot._load_sealed_h5_figure_inputs(outside.resolve(), aggregation.resolve(), allowed_paths=_allowed(matrix, aggregation))
