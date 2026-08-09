#!/usr/bin/env python3
"""Aggregate explicit H1 result tables without selecting or freezing features.

This utility reads only the CSV files supplied with ``--input``.  It evaluates
the portable-association rule from ``configs/study.yaml`` only if every one of
the configured discovery and confirmation datasets is present.  It never
searches for result files, fills absent model/dataset cells, or creates a
candidate manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


KEY_COLUMNS = ["dataset", "model", "view", "feature", "class_label"]
UNIT_COLUMNS = ["view", "feature", "class_label"]
CRITERION_PATTERN = re.compile(
    r"^same direction in >=(?P<direction_count>\d+)/(?P<core_count>\d+) core datasets, "
    r"significant in both confirmation datasets, and present in >=(?P<model_count>\d+)/(?P<panel_count>\d+) models$"
)


@dataclass(frozen=True)
class PortableAssociationRule:
    """The parsed, configuration-pinned portable-association acceptance rule."""

    text: str
    direction_count: int
    core_count: int
    model_count: int
    panel_count: int


@dataclass(frozen=True)
class H1StudyScope:
    """Configured datasets/models needed to evaluate, but never select, H1 units."""

    discovery: tuple[str, ...]
    confirmation: tuple[str, ...]
    models: tuple[str, ...]
    rule: PortableAssociationRule

    @property
    def core(self) -> tuple[str, ...]:
        return self.discovery + self.confirmation


def load_study_scope(config_path: Path | str) -> H1StudyScope:
    """Load and validate exactly the criterion declared in the study config."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    datasets = config["datasets"]
    discovery = tuple(datasets["discovery"])
    confirmation = tuple(datasets["confirmation"])
    models = tuple(config["models"]["score_panel"])
    text = str(config["statistics"]["portable_association"])
    match = CRITERION_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "The portable_association configuration is not in the locked, "
            "machine-checkable form required by this aggregation utility."
        )
    rule = PortableAssociationRule(text=text, **{key: int(value) for key, value in match.groupdict().items()})
    if len(discovery) + len(confirmation) != rule.core_count:
        raise ValueError(
            f"Configured core datasets ({len(discovery) + len(confirmation)}) do not match "
            f"the criterion denominator ({rule.core_count})."
        )
    if len(confirmation) != 2:
        raise ValueError("The locked rule requires exactly two confirmation datasets.")
    if len(models) != rule.panel_count:
        raise ValueError(
            f"Configured score panel ({len(models)}) does not match the criterion denominator "
            f"({rule.panel_count})."
        )
    if len(set(discovery + confirmation)) != rule.core_count:
        raise ValueError("Core dataset names must be unique.")
    if len(set(models)) != rule.panel_count:
        raise ValueError("Score-panel model names must be unique.")
    return H1StudyScope(discovery=discovery, confirmation=confirmation, models=models, rule=rule)


def statistic_columns(statistic: str) -> tuple[str, str]:
    """Map a precomputed association estimator name to its rho and q columns."""
    if statistic not in {"spearman", "partial_spearman"}:
        raise ValueError(f"Unsupported statistic: {statistic}")
    return f"{statistic}_rho", f"{statistic}_q"


def load_explicit_inputs(paths: Iterable[Path], scope: H1StudyScope, statistic: str) -> pd.DataFrame:
    """Read only explicitly supplied tables and reject out-of-scope observations."""
    rho_column, q_column = statistic_columns(statistic)
    tables: list[pd.DataFrame] = []
    for path in paths:
        table = pd.read_csv(path)
        missing = [column for column in [*KEY_COLUMNS, rho_column, q_column] if column not in table.columns]
        if missing:
            raise ValueError(f"{path} is not a compatible H1 association table; missing {missing}")
        table = table[[*KEY_COLUMNS, rho_column, q_column]].copy()
        table["_input_path"] = str(path)
        tables.append(table)
    if not tables:
        raise ValueError("At least one explicit --input association CSV is required.")
    rows = pd.concat(tables, ignore_index=True)
    rows["dataset"] = rows["dataset"].astype("string").str.strip()
    rows["model"] = rows["model"].astype("string").str.strip()
    rows["view"] = rows["view"].astype("string").str.strip()
    rows["feature"] = rows["feature"].astype("string").str.strip()
    if rows[["dataset", "model", "view", "feature"]].isna().any().any() or (
        rows[["dataset", "model", "view", "feature"]] == ""
    ).any().any():
        raise ValueError("H1 association keys cannot be missing or empty.")
    rows["class_label"] = pd.to_numeric(rows["class_label"], errors="raise").astype(int)
    if not rows["class_label"].isin([0, 1]).all():
        raise ValueError("H1 class_label must be 0 or 1.")
    rows[rho_column] = pd.to_numeric(rows[rho_column], errors="coerce")
    rows[q_column] = pd.to_numeric(rows[q_column], errors="coerce")
    unknown_datasets = sorted(set(rows["dataset"]) - set(scope.core))
    unknown_models = sorted(set(rows["model"]) - set(scope.models))
    if unknown_datasets:
        raise ValueError(
            "Inputs contain datasets outside the configured five-core-dataset audit scope: "
            f"{unknown_datasets}. Supply only explicitly scoped core result tables."
        )
    if unknown_models:
        raise ValueError(
            "Inputs contain models outside the configured eight-model score panel: "
            f"{unknown_models}."
        )
    if rows.duplicated(KEY_COLUMNS).any():
        duplicates = rows.loc[rows.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(8).to_dict("records")
        raise ValueError(f"Duplicate exact H1 association keys across explicit inputs: {duplicates}")
    valid_q = rows[q_column].dropna()
    if not valid_q.between(0.0, 1.0).all():
        raise ValueError(f"{q_column} must be in [0, 1] when present.")
    if not rows[rho_column].dropna().between(-1.0, 1.0).all():
        raise ValueError(f"{rho_column} must be in [-1, 1] when present.")
    return rows


def _direction(value: float) -> str:
    if not np.isfinite(value) or value == 0.0:
        return "none"
    return "positive" if value > 0.0 else "negative"


def _model_unit_row(
    unit: tuple[object, ...],
    model: str,
    rows: pd.DataFrame,
    scope: H1StudyScope,
    rho_column: str,
    q_column: str,
    alpha: float,
) -> dict[str, object]:
    """Evaluate one model/unit matrix; absent cells remain explicit missingness."""
    unit_rows = rows.loc[(rows["model"] == model)]
    by_dataset = unit_rows.set_index("dataset")
    available = by_dataset.index.intersection(scope.core)
    finite = by_dataset.loc[available, rho_column].dropna()
    directions = finite.map(_direction)
    positive = int((directions == "positive").sum())
    negative = int((directions == "negative").sum())
    if positive > negative:
        adopted_direction = "positive"
    elif negative > positive:
        adopted_direction = "negative"
    else:
        adopted_direction = "none"
    same_direction = max(positive, negative)
    missing_datasets = [dataset for dataset in scope.core if dataset not in by_dataset.index]
    nonfinite_datasets = [
        dataset
        for dataset in scope.core
        if dataset in by_dataset.index and not np.isfinite(float(by_dataset.at[dataset, rho_column]))
    ]
    complete_core_matrix = not missing_datasets and not nonfinite_datasets
    directionally_stable = bool(
        complete_core_matrix and adopted_direction != "none" and same_direction >= scope.rule.direction_count
    )
    confirmation_significant = True
    confirmation_direction_matched = True
    for dataset in scope.confirmation:
        if dataset not in by_dataset.index:
            confirmation_significant = False
            confirmation_direction_matched = False
            continue
        rho = by_dataset.at[dataset, rho_column]
        q_value = by_dataset.at[dataset, q_column]
        if not np.isfinite(float(q_value)) or float(q_value) > alpha:
            confirmation_significant = False
        if _direction(float(rho)) != adopted_direction:
            confirmation_direction_matched = False
    model_meets_rule = bool(
        directionally_stable and confirmation_significant and confirmation_direction_matched
    )
    return {
        "view": unit[0],
        "feature": unit[1],
        "class_label": unit[2],
        "model": model,
        "available_core_dataset_rows": int(len(available)),
        "finite_rho_core_dataset_rows": int(len(finite)),
        "positive_direction_datasets": positive,
        "negative_direction_datasets": negative,
        "same_direction_dataset_count": same_direction,
        "adopted_direction": adopted_direction,
        "missing_core_datasets": ";".join(missing_datasets),
        "nonfinite_rho_core_datasets": ";".join(nonfinite_datasets),
        "complete_core_matrix": complete_core_matrix,
        "directionally_stable_4_of_5": directionally_stable,
        "both_confirmation_q_le_alpha": confirmation_significant,
        "both_confirmation_match_direction": confirmation_direction_matched,
        "model_meets_portable_subcriterion": model_meets_rule,
    }


def aggregate_portable_associations(
    rows: pd.DataFrame,
    scope: H1StudyScope,
    *,
    statistic: str = "partial_spearman",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Produce exhaustive audit reports, without choosing any feature for later work."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    rho_column, q_column = statistic_columns(statistic)
    supplied_datasets = tuple(dataset for dataset in scope.core if dataset in set(rows["dataset"]))
    all_core_supplied = supplied_datasets == scope.core
    model_rows: list[dict[str, object]] = []
    for unit in rows[UNIT_COLUMNS].drop_duplicates().sort_values(UNIT_COLUMNS).itertuples(index=False, name=None):
        unit_rows = rows.loc[
            (rows["view"] == unit[0])
            & (rows["feature"] == unit[1])
            & (rows["class_label"] == unit[2])
        ]
        for model in scope.models:
            model_rows.append(_model_unit_row(unit, model, unit_rows, scope, rho_column, q_column, alpha))
    model_report = pd.DataFrame(model_rows)
    unit_rows: list[dict[str, object]] = []
    for unit, group in model_report.groupby(UNIT_COLUMNS, sort=True, dropna=False):
        qualified_models = int(group["model_meets_portable_subcriterion"].sum())
        result_evaluable = bool(all_core_supplied)
        criterion_met = bool(result_evaluable and qualified_models >= scope.rule.model_count)
        unit_rows.append(
            {
                "view": unit[0],
                "feature": unit[1],
                "class_label": int(unit[2]),
                "models_in_configured_panel": len(scope.models),
                "models_with_complete_core_matrix": int(group["complete_core_matrix"].sum()),
                "models_meeting_portable_subcriterion": qualified_models,
                "final_portable_criterion_evaluable": result_evaluable,
                "final_portable_criterion_met": criterion_met,
                "final_status": (
                    "not_evaluable_requires_all_5_core_datasets"
                    if not result_evaluable
                    else ("meets_configured_portable_association_criterion" if criterion_met else "does_not_meet_configured_portable_association_criterion")
                ),
            }
        )
    feature_report = pd.DataFrame(unit_rows)
    report = {
        "aggregation_kind": "descriptive_portable_association_audit",
        "selection_or_freezing_performed": False,
        "statistic": statistic,
        "rho_column": rho_column,
        "q_column": q_column,
        "alpha": alpha,
        "portable_association_rule": scope.rule.text,
        "core_datasets": list(scope.core),
        "discovery_datasets": list(scope.discovery),
        "confirmation_datasets": list(scope.confirmation),
        "configured_models": list(scope.models),
        "explicitly_supplied_datasets": list(supplied_datasets),
        "all_five_core_datasets_supplied": all_core_supplied,
        "confirmation_datasets_explicitly_supplied": [
            dataset for dataset in scope.confirmation if dataset in supplied_datasets
        ],
        "n_model_unit_rows": int(len(model_report)),
        "n_feature_view_class_units": int(len(feature_report)),
        "n_units_meeting_rule": int(feature_report["final_portable_criterion_met"].sum()),
    }
    return model_report, feature_report, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path, help="Explicit association_summary.csv input; repeat as needed.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for new aggregation reports.")
    parser.add_argument("--config", default="configs/study.yaml", type=Path)
    parser.add_argument("--statistic", choices=["spearman", "partial_spearman"], default="partial_spearman")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    scope = load_study_scope(args.config)
    rows = load_explicit_inputs(args.input, scope, args.statistic)
    model_report, feature_report, report = aggregate_portable_associations(
        rows, scope, statistic=args.statistic, alpha=args.alpha
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "portable_association_model_report.csv"
    feature_path = args.output_dir / "portable_association_feature_report.csv"
    metadata_path = args.output_dir / "portable_association_aggregation_report.json"
    model_report.to_csv(model_path, index=False)
    feature_report.to_csv(feature_path, index=False)
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {model_path} ({len(model_report)} model/unit rows)")
    print(f"wrote {feature_path} ({len(feature_report)} feature/view/class rows)")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
