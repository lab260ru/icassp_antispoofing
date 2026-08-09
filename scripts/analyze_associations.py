#!/usr/bin/env python3
"""Run the locked H1 score-feature association tests on local pinned artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_features import FEATURE_NAMES, FEATURE_VERSION
from src.statistics import benjamini_hochberg, partial_spearman


def load_inputs(feature_root: Path, score_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(feature_root / FEATURE_VERSION / dataset / "features_wide.parquet")
    scores = pd.read_parquet(score_root / dataset / "score_panel.parquet")
    return features, scores


def analyze_dataset(dataset: str, features: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    label_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    for view, view_features in features.groupby("view", sort=True):
        for feature in FEATURE_NAMES:
            data = view_features[["sample_id", "label", feature]].dropna()
            if data["label"].nunique() == 2:
                auc = roc_auc_score(data["label"], data[feature])
                label_rows.append({
                    "dataset": dataset,
                    "view": view,
                    "feature": feature,
                    "n": len(data),
                    "label_auroc": float(auc),
                    "label_separation_auroc": float(max(auc, 1.0 - auc)),
                })
        for model, model_scores in scores.groupby("model", sort=True):
            joined = view_features.merge(
                model_scores[["sample_id", "label", "score_spoof", "orientation"]],
                on="sample_id",
                suffixes=("_feature", "_score"),
                validate="one_to_one",
            )
            if not (joined["label_feature"] == joined["label_score"]).all():
                raise ValueError(f"Label mismatch joining {dataset}/{model}/{view}")
            joined = joined.rename(columns={"label_feature": "label"})
            for class_label, class_data in joined.groupby("label", sort=True):
                for feature in FEATURE_NAMES:
                    subset = class_data[[feature, "score_spoof"]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(subset) < 12:
                        rho, p_value = float("nan"), float("nan")
                    else:
                        rho, p_value = stats.spearmanr(subset[feature], subset["score_spoof"])
                    partial_n, partial_rho, partial_p = partial_spearman(
                        class_data,
                        feature,
                        "score_spoof",
                        numeric_controls=["duration_seconds", "integrated_lufs"],
                        categorical_controls=["speaker_id", "attack_id"],
                    )
                    association_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "view": view,
                        "feature": feature,
                        "class_label": int(class_label),
                        "n": len(subset),
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_value),
                        "partial_n": partial_n,
                        "partial_spearman_rho": partial_rho,
                        "partial_spearman_p": partial_p,
                        "orientation": model_scores["orientation"].iloc[0],
                    })
    associations = pd.DataFrame(association_rows)
    associations["spearman_q"] = benjamini_hochberg(associations["spearman_p"])
    associations["partial_spearman_q"] = benjamini_hochberg(associations["partial_spearman_p"])
    label_metrics = pd.DataFrame(label_rows)
    report = {
        "dataset": dataset,
        "n_feature_samples": int(features["sample_id"].nunique()),
        "n_score_rows": int(len(scores)),
        "n_models": int(scores["model"].nunique()),
        "n_joined_per_model": {name: int(len(features.merge(group[["sample_id"]], on="sample_id", how="inner"))) for name, group in scores.groupby("model")},
    }
    return associations, label_metrics, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--feature-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features")
    parser.add_argument("--score-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/scores")
    parser.add_argument("--output-dir", default="experiments/h1_feature_association/results")
    args = parser.parse_args()

    feature_root, score_root, output_dir = Path(args.feature_root), Path(args.score_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_associations: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    for dataset in args.dataset:
        features, scores = load_inputs(feature_root, score_root, dataset)
        associations, labels, report = analyze_dataset(dataset, features, scores)
        all_associations.append(associations)
        all_labels.append(labels)
        reports.append(report)
    association_output = pd.concat(all_associations, ignore_index=True)
    label_output = pd.concat(all_labels, ignore_index=True)
    association_output.to_csv(output_dir / "association_summary.csv", index=False)
    label_output.to_csv(output_dir / "feature_label_metrics.csv", index=False)
    (output_dir / "association_join_report.json").write_text(json.dumps(reports, indent=2, sort_keys=True))
    print(f"wrote {output_dir / 'association_summary.csv'} ({len(association_output)} rows)")


if __name__ == "__main__":
    main()
