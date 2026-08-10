"""Synthetic tests for H8 source-only robust fusion optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.h8_fusion_training import _eer, fit_nonnegative_fuser
from src.h8_score_fusion import MODELS, feature_column


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, offset in (("one", 0.0), ("two", 0.15)):
        for label in (0, 1):
            for index in range(5):
                row: dict[str, object] = {"dataset": dataset, "label": label}
                evidence = -1.0 + offset + index * 0.02 if label == 0 else 1.0 + offset + index * 0.02
                for model_index, model in enumerate(MODELS):
                    row[feature_column(model)] = evidence + model_index * 0.01
                rows.append(row)
    return pd.DataFrame(rows)


def test_eer_handles_a_vertical_crossing() -> None:
    assert _eer(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        ("erm", {}),
        ("groupdro", {"groupdro_eta": 0.05}),
        ("vrex", {"vrex_lambda": 0.1}),
    ],
)
def test_nonnegative_fusers_emit_finite_scores(kind: str, kwargs: dict[str, float]) -> None:
    model = fit_nonnegative_fuser(_frame(), kind=kind, l2=0.01, seed=1701, steps=8, **kwargs)
    assert (model.weights >= 0).all()
    assert np.isfinite(model.predict(_frame().loc[:, [feature_column(item) for item in MODELS]].to_numpy())).all()
    assert set(model.final_group_losses) == {"one|0", "one|1", "two|0", "two|1"}
