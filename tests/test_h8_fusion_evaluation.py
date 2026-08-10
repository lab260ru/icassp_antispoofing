"""Pure tests for the final H8 decision and uncertainty computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.h8_fusion_evaluation import (
    BOOTSTRAP_CAP_PER_TARGET_LABEL,
    BOOTSTRAP_REPLICATES,
    bootstrap_mean_eer_difference,
    decision_gate,
    freeze_bootstrap_panel,
)
from src.h8_fusion_training import FUSION_METHODS
from src.h8_score_fusion import TARGET_DATASETS


def _joined() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(TARGET_DATASETS):
        for label in (0, 1):
            for sample in range(4):
                score = -1.0 + sample * 0.1 if label == 0 else 1.0 + sample * 0.1
                row: dict[str, object] = {"dataset": dataset, "sample_id": f"{dataset}_{label}_{sample}", "label": label}
                for method in FUSION_METHODS:
                    row[method] = score
                row["B1"] = score - (0.1 if label else -0.1)
                row["B2"] = score - (0.2 if label else -0.2)
                row["P"] = score
                rows.append(row)
    return pd.DataFrame(rows)


def test_bootstrap_panel_is_label_stratified_and_deterministic() -> None:
    panel = freeze_bootstrap_panel(_joined())
    assert len(panel) == len(TARGET_DATASETS) * 2 * 4
    assert int(panel["selection_rank"].max()) == 4
    assert BOOTSTRAP_CAP_PER_TARGET_LABEL >= 4


def test_bootstrap_difference_is_reproducible() -> None:
    panel = freeze_bootstrap_panel(_joined())
    first = bootstrap_mean_eer_difference(panel)
    second = bootstrap_mean_eer_difference(panel)
    assert len(first) == BOOTSTRAP_REPLICATES
    assert np.array_equal(first, second)


def test_decision_gate_reports_all_literal_rules() -> None:
    rows: list[dict[str, object]] = []
    for dataset in TARGET_DATASETS:
        for method in FUSION_METHODS:
            value = {"B0": 0.15, "B1": 0.12, "B2": 0.10, "P": 0.08, "A1": 0.09}[method]
            rows.append({"dataset": dataset, "method": method, "eer": value})
    decision = decision_gate(pd.DataFrame(rows), np.full(BOOTSTRAP_REPLICATES, -0.02))
    assert decision["positive_result_gate_passed"] is True
    assert all(decision["rules"].values())
