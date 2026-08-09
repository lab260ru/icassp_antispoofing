"""Transparent, dependency-light statistical utilities for confirmatory H1."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

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
    """Rank-residual partial correlation with explicit numeric/category controls.

    A tested variable is never also included as a control.  This matters for
    feature registries that intentionally include covariates such as loudness:
    conditioning loudness on itself is undefined and can create duplicate
    DataFrame columns.  Repeated requested controls are likewise reduced to a
    single design column, preserving first-occurrence order.
    """
    targets = {x, y}
    active_numeric: list[str] = []
    for name in numeric_controls:
        if name not in targets and name not in active_numeric:
            active_numeric.append(name)
    active_categorical: list[str] = []
    for name in categorical_controls:
        if name not in targets and name not in active_numeric and name not in active_categorical:
            active_categorical.append(name)
    needed = [x, y, *active_numeric, *active_categorical]
    data = frame[needed].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[x, y])
    if len(data) < 12:
        return len(data), float("nan"), float("nan")
    columns: list[np.ndarray] = [np.ones(len(data))]
    for name in active_numeric:
        values = pd.to_numeric(data[name], errors="coerce")
        median = values.median()
        values = values.fillna(0.0 if not np.isfinite(median) else median)
        columns.append(stats.rankdata(values.to_numpy(dtype=float)))
    for name in active_categorical:
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


def stratified_cluster_bootstrap_ci(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    cluster_column: str,
    strata_columns: Sequence[str],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Return a deterministic percentile CI from a stratified cluster bootstrap.

    Complete clusters are sampled with replacement independently inside each
    declared stratum.  ``metric`` receives the concatenated resample and must
    return ``nan`` when the resample is not estimable.  Invalid replicates are
    counted in the result rather than silently discarded.

    The caller owns all row filtering (for example, the within-label slice for
    H1).  This prevents this generic utility from making implicit analytic
    choices or selecting candidates from the data.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    required = [cluster_column, *strata_columns]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Bootstrap input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty frame")
    if frame[cluster_column].isna().any():
        raise ValueError(f"Bootstrap cluster column contains missing values: {cluster_column}")
    if any(frame[name].isna().any() for name in strata_columns):
        raise ValueError("Bootstrap stratum columns contain missing values")

    # Resetting the index makes sampled integer locations unambiguous even when
    # the upstream merge retained a non-contiguous index.
    data = frame.reset_index(drop=True)
    grouped = list(data.groupby(list(strata_columns), sort=True, dropna=False))
    if not grouped:
        raise ValueError("No bootstrap strata available")
    strata: list[list[np.ndarray]] = []
    for _, stratum in grouped:
        cluster_rows = [group.index.to_numpy(dtype=int) for _, group in stratum.groupby(cluster_column, sort=True)]
        if not cluster_rows:
            raise ValueError("Bootstrap stratum has no clusters")
        strata.append(cluster_rows)

    generator = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan, dtype=float)
    for replicate in range(replicates):
        sampled_rows: list[np.ndarray] = []
        for cluster_rows in strata:
            selected = generator.integers(0, len(cluster_rows), size=len(cluster_rows))
            sampled_rows.extend(cluster_rows[position] for position in selected)
        sample = data.iloc[np.concatenate(sampled_rows)]
        estimate = float(metric(sample))
        if np.isfinite(estimate):
            estimates[replicate] = estimate

    valid = estimates[np.isfinite(estimates)]
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, object] = {
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid": int(len(valid)),
        "bootstrap_replicates_invalid": int(replicates - len(valid)),
        "bootstrap_confidence": float(confidence),
        "bootstrap_n_input_rows": int(len(data)),
        "bootstrap_n_strata": int(len(strata)),
        "bootstrap_n_clusters": int(sum(len(clusters) for clusters in strata)),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_excludes_zero": False,
    }
    if len(valid):
        lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
        output["ci_low"] = float(lower)
        output["ci_high"] = float(upper)
        output["ci_excludes_zero"] = bool(lower > 0.0 or upper < 0.0)
    return output
