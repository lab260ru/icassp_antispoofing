"""Synthetic tests for the completed-result-only H4 supplementary heatmap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.h4_heatmap import (
    AGGREGATION_FILENAME,
    H4_DATASETS,
    H4_VIEWS,
    MATRIX_FILENAME,
    PDF_FILENAME,
    PNG_FILENAME,
    load_completed_h4_artifacts,
    render_h4_delta_auc_heatmap,
)


def _completed_h4_csvs(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    features = [f"feature_{index:02d}" for index in range(28)]
    matrix_rows: list[dict[str, object]] = []
    aggregation_rows: list[dict[str, object]] = []
    for view_index, view in enumerate(H4_VIEWS):
        for feature_index, feature in enumerate(features):
            stable = feature_index % 11 == 0
            aggregation_rows.append(
                {
                    "view": view,
                    "feature": feature,
                    "atlas_only_stable_label_association": stable,
                }
            )
            for dataset_index, dataset in enumerate(H4_DATASETS):
                matrix_rows.append(
                    {
                        "dataset": dataset,
                        "view": view,
                        "feature": feature,
                        "cell_status": "ok",
                        "delta_auc": (dataset_index - 2) / 5.0 + view_index / 100.0 + feature_index / 10_000.0,
                    }
                )
    matrix = root / MATRIX_FILENAME
    aggregation = root / AGGREGATION_FILENAME
    pd.DataFrame(matrix_rows).to_csv(matrix, index=False)
    pd.DataFrame(aggregation_rows).to_csv(aggregation, index=False)
    return matrix, aggregation


def test_h4_heatmap_generates_vector_and_300dpi_outputs(tmp_path: Path) -> None:
    matrix, aggregation = _completed_h4_csvs(tmp_path / "completed")
    pdf_path, png_path = render_h4_delta_auc_heatmap(matrix.resolve(), aggregation.resolve(), tmp_path / "figure")
    assert pdf_path.name == PDF_FILENAME and pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.name == PNG_FILENAME and png_path.stat().st_size > 0
    with Image.open(png_path) as image:
        assert image.size[0] > 1_000 and image.size[1] > 1_000
        assert image.info["dpi"][0] >= 299


def test_h4_heatmap_refuses_incomplete_or_response_columns(tmp_path: Path) -> None:
    matrix, aggregation = _completed_h4_csvs(tmp_path / "completed")
    table = pd.read_csv(matrix)
    table.loc[0, "cell_status"] = "failed_unavailable_finite_label"
    table.to_csv(matrix, index=False)
    with pytest.raises(ValueError, match="completed"):
        load_completed_h4_artifacts(matrix.resolve(), aggregation.resolve())

    matrix, aggregation = _completed_h4_csvs(tmp_path / "completed_again")
    table = pd.read_csv(matrix)
    table["response_score"] = 0.0
    table.to_csv(matrix, index=False)
    with pytest.raises(ValueError, match="response-like"):
        load_completed_h4_artifacts(matrix.resolve(), aggregation.resolve())

