from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_h1_discovery_crest_factor import (
    EXPECTED_DISCOVERY,
    load_discovery_models,
    load_fixed_discovery_slice,
    render_plot,
)


def _write_table(path: Path, dataset: str, models: tuple[str, ...], *, missing_model: bool = False) -> None:
    rows = []
    for index, model in enumerate(models):
        if missing_model and index == 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "view": "full_waveform",
                "feature": "crest_factor_db",
                "class_label": 1,
                "analysis_stage": "screen",
                "partial_spearman_rho": -0.35 + 0.04 * index,
                "partial_spearman_q": 0.01,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _valid_inputs(tmp_path: Path, models: tuple[str, ...]) -> dict[str, Path]:
    inputs = {}
    for dataset in EXPECTED_DISCOVERY:
        path = tmp_path / f"{dataset}.csv"
        _write_table(path, dataset, models)
        inputs[dataset] = path
    return inputs


def test_fixed_discovery_plot_writes_pdf_png_and_nonportable_metadata(tmp_path: Path) -> None:
    models = load_discovery_models("configs/study.yaml")
    inputs = _valid_inputs(tmp_path, models)
    rows, paths = load_fixed_discovery_slice(inputs, "configs/study.yaml")
    outputs = render_plot(rows, models, tmp_path / "output", paths)

    assert len(rows) == 3 * len(models)
    assert outputs["pdf"].is_file()
    assert outputs["png"].is_file()
    metadata = outputs["metadata"].read_text(encoding="utf-8")
    assert '"portable_or_causal_claim": false' in metadata
    assert '"candidate_selection_or_freezing_performed": false' in metadata


def test_fixed_discovery_plot_rejects_duplicate_explicit_paths(tmp_path: Path) -> None:
    models = load_discovery_models("configs/study.yaml")
    inputs = _valid_inputs(tmp_path, models)
    inputs["ASVspoof2021_LA"] = inputs["ASVspoof2019_LA"]
    with pytest.raises(ValueError, match="distinct CSV"):
        load_fixed_discovery_slice(inputs, "configs/study.yaml")


def test_fixed_discovery_plot_rejects_dataset_or_panel_omission(tmp_path: Path) -> None:
    models = load_discovery_models("configs/study.yaml")
    inputs = _valid_inputs(tmp_path, models)
    _write_table(inputs["ASVspoof2021_DF"], "ASVspoof2021_LA", models)
    with pytest.raises(ValueError, match="must contain only ASVspoof2021_DF"):
        load_fixed_discovery_slice(inputs, "configs/study.yaml")

    _write_table(inputs["ASVspoof2021_DF"], "ASVspoof2021_DF", models, missing_model=True)
    with pytest.raises(ValueError, match="missing configured score-panel model"):
        load_fixed_discovery_slice(inputs, "configs/study.yaml")


def test_fixed_discovery_plot_rejects_missing_or_unknown_explicit_content(tmp_path: Path) -> None:
    models = load_discovery_models("configs/study.yaml")
    inputs = _valid_inputs(tmp_path, models)
    missing_input = dict(inputs)
    missing_input.pop("ASVspoof2021_DF")
    with pytest.raises(ValueError, match="Expected explicit inputs"):
        load_fixed_discovery_slice(missing_input, "configs/study.yaml")

    table = pd.read_csv(inputs["ASVspoof2021_DF"])
    table.loc[len(table)] = {
        "dataset": "ASVspoof2021_DF",
        "model": "unconfigured-model",
        "view": "deterministic_crop",
        "feature": "crest_factor_db",
        "class_label": 1,
        "analysis_stage": "screen",
        "partial_spearman_rho": 0.1,
        "partial_spearman_q": 0.1,
    }
    table.to_csv(inputs["ASVspoof2021_DF"], index=False)
    with pytest.raises(ValueError, match="outside the configured score panel"):
        load_fixed_discovery_slice(inputs, "configs/study.yaml")
