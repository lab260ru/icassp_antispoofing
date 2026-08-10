"""Synthetic-only tests for the sealed H6 supplementary figure renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.h6_score_agreement_heatmap as h6_plot
from src.h6_score_agreement import DATASETS, LABELS, MODEL_PAIRS


def _compact_contract(root: Path, matrix: Path, aggregation: Path, provenance: Path) -> h6_plot.H6FigureInputContract:
    return h6_plot.H6FigureInputContract(
        matrix_path=matrix.resolve(),
        aggregation_path=aggregation.resolve(),
        provenance_path=provenance.resolve(),
        matrix_sha256=h6_plot.sha256_file(matrix),
        aggregation_sha256=h6_plot.sha256_file(aggregation),
        provenance_sha256=h6_plot.sha256_file(provenance),
    )


def _write_synthetic_h6_compact_artifacts(
    root: Path,
    *,
    extra_matrix_column: str | None = None,
    aggregation_mismatch: bool = False,
    drop_last_matrix_row: bool = False,
    incomplete_provenance: bool = False,
) -> tuple[Path, Path, Path, h6_plot.H6FigureInputContract]:
    """Create a compact H6-shaped fixture without a score artifact or project run."""
    root.mkdir(parents=True)
    matrix_path = root / h6_plot.MATRIX_FILENAME
    aggregation_path = root / h6_plot.AGGREGATION_FILENAME
    provenance_path = root / h6_plot.PROVENANCE_FILENAME
    matrix_rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        for label in LABELS:
            for pair_index, (model_a, model_b) in enumerate(MODEL_PAIRS):
                agreement = -0.9 + 0.1 * ((dataset_index * 11 + int(label) * 5 + pair_index * 3) % 19)
                seed, seed_hash = h6_plot._seed_for_cell(dataset, int(label), model_a, model_b)
                matrix_rows.append(
                    {
                        "dataset": dataset,
                        "label": int(label),
                        "model_a": model_a,
                        "model_b": model_b,
                        "cell_status": "ok",
                        "selected_sample_count": 5_000,
                        "model_a_available_score_count": 5_000,
                        "model_b_available_score_count": 5_000,
                        "exact_joined_sample_count": 5_000,
                        "spearman_agreement": agreement,
                        "bootstrap_replicates_requested": 200,
                        "bootstrap_replicates_valid": 200,
                        "bootstrap_replicates_invalid": 0,
                        "bootstrap_confidence": 0.95,
                        "spearman_ci_low": max(-1.0, agreement - 0.04),
                        "spearman_ci_high": min(1.0, agreement + 0.04),
                        "bootstrap_cluster_count": 5_000,
                        "bootstrap_seed": seed,
                        "bootstrap_seed_sha256": seed_hash,
                    }
                )
    if drop_last_matrix_row:
        matrix_rows.pop()
    matrix = pd.DataFrame(matrix_rows, columns=h6_plot.MATRIX_COLUMNS)
    if extra_matrix_column is not None:
        matrix[extra_matrix_column] = 0.0
    matrix.to_csv(matrix_path, index=False)

    aggregation_rows: list[dict[str, object]] = []
    for model_a, model_b in MODEL_PAIRS:
        unit = matrix.loc[(matrix["model_a"] == model_a) & (matrix["model_b"] == model_b)]
        values = unit["spearman_agreement"].to_numpy(dtype=float)
        aggregation_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "expected_dataset_class_cells": 10,
                "observed_dataset_class_cells": 10,
                "ok_cell_count": 10,
                "failed_cell_count": 0,
                "all_ten_cells_ok": True,
                "median_spearman_agreement": float(np.median(values)),
                "minimum_spearman_agreement": float(np.min(values)),
                "maximum_spearman_agreement": float(np.max(values)),
                "cross_cell_spearman_range": float(np.max(values) - np.min(values)),
                "cell_statuses": "ok",
            }
        )
    aggregation = pd.DataFrame(aggregation_rows, columns=h6_plot.AGGREGATION_COLUMNS)
    if aggregation_mismatch:
        aggregation.loc[0, "median_spearman_agreement"] = 0.123
    aggregation.to_csv(aggregation_path, index=False)

    provenance = {
        "h6_version": h6_plot.H6_VERSION,
        "raw_score_only": True,
        "score_values_emitted": False,
        "analysis_status": "complete" if not incomplete_provenance else "stopped_with_explicit_unavailable_cells",
        "stop_reason": "complete",
        "n_expected_dataset_class_model_pair_cells": 280,
        "n_written_dataset_class_model_pair_cells": 280,
        "complete_280_cell_matrix": True,
        "ok_cell_count": 280,
        "failed_or_unavailable_cell_count": 0,
        "n_model_pair_summaries": 28,
        "outputs": {
            "matrix": {
                "absolute_path": str(matrix_path.resolve()),
                "sha256": h6_plot.sha256_file(matrix_path),
                "byte_size": matrix_path.stat().st_size,
            },
            "aggregation": {
                "absolute_path": str(aggregation_path.resolve()),
                "sha256": h6_plot.sha256_file(aggregation_path),
                "byte_size": aggregation_path.stat().st_size,
            },
        },
    }
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    return matrix_path, aggregation_path, provenance_path, _compact_contract(root, matrix_path, aggregation_path, provenance_path)


def test_h6_figure_validates_synthetic_complete_contract_and_renders(tmp_path: Path) -> None:
    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(tmp_path / "sealed")
    inputs = h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)
    assert len(inputs.matrix) == 280
    assert len(inputs.aggregation) == 28
    display = h6_plot._ordered_display_matrix(inputs.matrix)
    assert display.shape == (28, 10)
    assert np.isfinite(display).all()

    output = tmp_path / "figure"
    pdf_path, png_path, metadata_path = h6_plot._render_h6_within_class_score_agreement_heatmap(
        matrix.resolve(), aggregation.resolve(), provenance.resolve(), output, contract=contract
    )
    assert pdf_path.read_bytes().startswith(b"%PDF")
    with Image.open(png_path) as image:
        assert image.width > 2_000 and image.height > 2_000
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["color_scale"] == {"fixed": True, "maximum": 1.0, "minimum": -1.0}
    assert len(metadata["fixed_model_pair_order"]) == 28
    assert len(metadata["fixed_column_order"]) == 10
    assert metadata["sorting_or_selection"] == "none; rows and columns use only the registered fixed orders"
    with pytest.raises(FileExistsError, match="non-overwritable"):
        h6_plot._render_h6_within_class_score_agreement_heatmap(
            matrix.resolve(), aggregation.resolve(), provenance.resolve(), output, contract=contract
        )


def test_h6_figure_rejects_hash_changed_and_substituted_paths(tmp_path: Path) -> None:
    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(tmp_path / "sealed")
    matrix.write_text(matrix.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matrix hash changed"):
        h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)

    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(tmp_path / "sealed_again")
    substitute = tmp_path / "substitute" / h6_plot.MATRIX_FILENAME
    substitute.parent.mkdir()
    substitute.write_bytes(matrix.read_bytes())
    with pytest.raises(ValueError, match="literal sealed"):
        h6_plot._load_sealed_h6_figure_inputs(substitute.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)


def test_h6_figure_rejects_raw_response_schema_and_aggregation_mismatch(tmp_path: Path) -> None:
    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(
        tmp_path / "raw_response", extra_matrix_column="raw_score"
    )
    with pytest.raises(ValueError, match="raw/model-response"):
        h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)

    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(
        tmp_path / "bad_aggregation", aggregation_mismatch=True
    )
    with pytest.raises(ValueError, match="does not match"):
        h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)


def test_h6_figure_rejects_incomplete_matrix_and_incomplete_provenance(tmp_path: Path) -> None:
    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(
        tmp_path / "missing_cell", drop_last_matrix_row=True
    )
    with pytest.raises(ValueError, match="exactly 280"):
        h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)

    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(
        tmp_path / "incomplete_provenance", incomplete_provenance=True
    )
    with pytest.raises(ValueError, match="analysis_status"):
        h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=contract)


def test_h6_figure_contract_rejects_impossible_path_before_any_input_read(tmp_path: Path) -> None:
    matrix, aggregation, provenance, contract = _write_synthetic_h6_compact_artifacts(tmp_path / "sealed")
    impossible = h6_plot.H6FigureInputContract(
        matrix_path=Path(h6_plot.MATRIX_FILENAME),
        aggregation_path=contract.aggregation_path,
        provenance_path=contract.provenance_path,
        matrix_sha256=contract.matrix_sha256,
        aggregation_sha256=contract.aggregation_sha256,
        provenance_sha256=contract.provenance_sha256,
    )
    with pytest.raises(ValueError, match="impossible matrix path"):
        h6_plot._load_sealed_h6_figure_inputs(matrix.resolve(), aggregation.resolve(), provenance.resolve(), contract=impossible)
