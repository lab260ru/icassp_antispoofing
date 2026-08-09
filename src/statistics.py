"""Transparent, dependency-light statistical utilities for confirmatory H1."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    output = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(output, index=p_values.index)
    subset = values[valid]
    order = np.argsort(subset)
    ranked = subset[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    output[valid] = restored
    return pd.Series(output, index=p_values.index)


def _rank_residual(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(values, method="average")
    coefficients, *_ = np.linalg.lstsq(design, ranked, rcond=None)
    return ranked - design @ coefficients


def partial_spearman(frame: pd.DataFrame, x: str, y: str, numeric_controls: list[str], categorical_controls: list[str]) -> tuple[int, float, float]:
    """Rank-residual partial correlation with explicit numeric/category controls."""
    needed = [x, y, *numeric_controls, *categorical_controls]
    data = frame[needed].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[x, y])
    if len(data) < 12:
        return len(data), float("nan"), float("nan")
    columns: list[np.ndarray] = [np.ones(len(data))]
    for name in numeric_controls:
        values = pd.to_numeric(data[name], errors="coerce")
        values = values.fillna(values.median())
        columns.append(stats.rankdata(values.to_numpy(dtype=float)))
    for name in categorical_controls:
        categories = data[name].fillna("__missing__").astype(str)
        dummies = pd.get_dummies(categories, drop_first=True, dtype=float)
        if not dummies.empty:
            columns.extend(dummies[column].to_numpy(dtype=float) for column in dummies)
    design = np.column_stack(columns)
    if len(data) <= design.shape[1] + 3:
        return len(data), float("nan"), float("nan")
    x_residual = _rank_residual(data[x].to_numpy(dtype=float), design)
    y_residual = _rank_residual(data[y].to_numpy(dtype=float), design)
    rho = float(np.corrcoef(x_residual, y_residual)[0, 1])
    if not np.isfinite(rho):
        return len(data), float("nan"), float("nan")
    rho = float(np.clip(rho, -0.999999, 0.999999))
    df = len(data) - design.shape[1] - 2
    statistic = abs(rho) * math.sqrt(df / max(1.0 - rho * rho, 1e-12))
    p_value = float(2.0 * stats.t.sf(statistic, df))
    return len(data), rho, p_value
