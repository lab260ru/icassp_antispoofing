from __future__ import annotations

import pandas as pd
import pytest

from scripts.aggregate_h1_associations import (
    aggregate_portable_associations,
    load_explicit_inputs,
    load_study_scope,
)


def _complete_rows(scope, *, qualifying_models: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(scope.models):
        for dataset_index, dataset in enumerate(scope.core):
            qualifying = model_index < qualifying_models
            rho = 0.2 if qualifying or dataset_index < 3 else -0.2
            q_value = 0.01 if qualifying else 0.20
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "view": "full_waveform",
                    "feature": "crest_factor_db",
                    "class_label": 1,
                    "partial_spearman_rho": rho,
                    "partial_spearman_q": q_value,
                    "spearman_rho": rho,
                    "spearman_q": q_value,
                }
            )
    return pd.DataFrame(rows)


def test_final_rule_requires_all_five_datasets_and_five_models() -> None:
    scope = load_study_scope("configs/study.yaml")
    model_report, feature_report, metadata = aggregate_portable_associations(_complete_rows(scope), scope)

    assert metadata["all_five_core_datasets_supplied"] is True
    assert feature_report["models_meeting_portable_subcriterion"].tolist() == [5]
    assert feature_report["final_portable_criterion_met"].tolist() == [True]
    assert model_report["model_meets_portable_subcriterion"].sum() == 5


def test_four_of_five_observed_rows_are_missing_not_imputed() -> None:
    scope = load_study_scope("configs/study.yaml")
    rows = _complete_rows(scope).query("model == @scope.models[0]").copy()
    rows = rows.loc[rows["dataset"] != scope.confirmation[-1]].copy()
    model_report, feature_report, metadata = aggregate_portable_associations(rows, scope)

    assert metadata["all_five_core_datasets_supplied"] is False
    first_model = model_report.loc[model_report["model"] == scope.models[0]].iloc[0]
    assert first_model["available_core_dataset_rows"] == 4
    assert bool(first_model["complete_core_matrix"]) is False
    assert bool(first_model["model_meets_portable_subcriterion"]) is False
    assert feature_report["final_status"].tolist() == ["not_evaluable_requires_all_5_core_datasets"]


def test_missing_model_dataset_cell_blocks_model_even_when_all_datasets_supplied() -> None:
    scope = load_study_scope("configs/study.yaml")
    rows = _complete_rows(scope)
    rows = rows.loc[~((rows["model"] == scope.models[0]) & (rows["dataset"] == scope.discovery[0]))].copy()
    model_report, feature_report, metadata = aggregate_portable_associations(rows, scope)

    first_model = model_report.loc[model_report["model"] == scope.models[0]].iloc[0]
    assert metadata["all_five_core_datasets_supplied"] is True
    assert bool(first_model["complete_core_matrix"]) is False
    assert bool(first_model["model_meets_portable_subcriterion"]) is False
    assert feature_report["models_meeting_portable_subcriterion"].tolist() == [4]
    assert feature_report["final_portable_criterion_met"].tolist() == [False]


def test_explicit_input_rejects_unknown_dataset_and_duplicate_keys(tmp_path) -> None:
    scope = load_study_scope("configs/study.yaml")
    table = _complete_rows(scope).iloc[:1].copy()
    table.loc[:, "dataset"] = "unregistered_dataset"
    unknown_path = tmp_path / "unknown.csv"
    table.to_csv(unknown_path, index=False)
    with pytest.raises(ValueError, match="outside the configured five-core-dataset"):
        load_explicit_inputs([unknown_path], scope, "partial_spearman")

    duplicate_path = tmp_path / "duplicate.csv"
    _complete_rows(scope).iloc[:1].to_csv(duplicate_path, index=False)
    with pytest.raises(ValueError, match="Duplicate exact H1 association keys"):
        load_explicit_inputs([duplicate_path, duplicate_path], scope, "partial_spearman")
