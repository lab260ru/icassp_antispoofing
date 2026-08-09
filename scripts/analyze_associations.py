#!/usr/bin/env python3
"""Run the locked H1 score-feature association tests on local pinned artifacts."""

from __future__ import annotations

import argparse
import hashlib
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
from src.statistics import benjamini_hochberg, partial_spearman, stratified_cluster_bootstrap_ci


CANDIDATE_KEY = ["dataset", "model", "view", "feature", "class_label"]
CANDIDATE_MANIFEST_COLUMNS = [
    *CANDIDATE_KEY,
    "selection_status",
    "selection_split",
    "selection_basis",
    "frozen_at_utc",
]


def load_frozen_candidates(path: Path) -> pd.DataFrame:
    """Load explicit, frozen candidates; this function never selects by scores."""
    if path.suffix == ".parquet":
        candidates = pd.read_parquet(path)
    else:
        candidates = pd.read_csv(path)
    missing = [column for column in CANDIDATE_MANIFEST_COLUMNS if column not in candidates]
    if missing:
        raise ValueError(
            "Candidate manifest must explicitly record its freeze and selection "
            f"provenance; missing columns: {missing}"
        )
    candidates = candidates[CANDIDATE_MANIFEST_COLUMNS].copy()
    for column in ["dataset", "model", "view", "feature", "selection_status", "selection_split", "selection_basis", "frozen_at_utc"]:
        candidates[column] = candidates[column].astype("string").str.strip()
        if candidates[column].isna().any() or (candidates[column] == "").any():
            raise ValueError(f"Candidate manifest contains an empty {column}")
    candidates["class_label"] = pd.to_numeric(candidates["class_label"], errors="raise").astype(int)
    if not candidates["class_label"].isin([0, 1]).all():
        raise ValueError("Candidate manifest class_label must be 0 or 1")
    if not (candidates["selection_status"] == "frozen").all():
        raise ValueError("Candidate manifest selection_status must be 'frozen'")
    allowed_selection_splits = {"discovery", "preregistration", "preregistered", "pre_registered"}
    if not candidates["selection_split"].str.lower().isin(allowed_selection_splits).all():
        raise ValueError("Candidates must be selected from discovery or preregistration, not confirmation data")
    pd.to_datetime(candidates["frozen_at_utc"], utc=True, errors="raise")
    if candidates.duplicated(CANDIDATE_KEY).any():
        raise ValueError(f"Candidate manifest duplicates an exact H1 test: {CANDIDATE_KEY}")
    return candidates


def candidate_seed(base_seed: int, candidate: dict[str, object]) -> int:
    """Derive a stable per-candidate RNG seed independent of row ordering."""
    key = "|".join(str(candidate[field]) for field in CANDIDATE_KEY)
    digest = hashlib.blake2b(f"{base_seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32)


def bootstrap_cluster_ids(frame: pd.DataFrame) -> pd.Series:
    """Use speaker clusters when available, otherwise stable source utterances."""
    if "source_id" not in frame or frame["source_id"].isna().any():
        raise ValueError("H1 bootstrap requires a non-missing source_id for every selected sample")
    source = frame["source_id"].astype("string").str.strip()
    if (source == "").any():
        raise ValueError("H1 bootstrap source_id cannot be empty")
    if "speaker_id" not in frame:
        return "utterance:" + source
    speaker = frame["speaker_id"].astype("string").str.strip()
    has_speaker = speaker.notna() & (speaker != "")
    return pd.Series(np.where(has_speaker, "speaker:" + speaker.fillna(""), "utterance:" + source), index=frame.index, dtype="string")


def _spearman_metric(frame: pd.DataFrame, feature: str) -> float:
    if len(frame) < 12:
        return float("nan")
    value = stats.spearmanr(frame[feature], frame["score_spoof"])
    return float(value.statistic)


def _partial_spearman_metric(frame: pd.DataFrame, feature: str) -> float:
    return partial_spearman(
        frame,
        feature,
        "score_spoof",
        numeric_controls=["duration_seconds", "integrated_lufs"],
        categorical_controls=["speaker_id", "attack_id"],
    )[1]


def candidate_join(features: pd.DataFrame, scores: pd.DataFrame, candidate: dict[str, object]) -> pd.DataFrame:
    """Recreate the exact stable-ID join for one preselected confirmation test."""
    view_features = features.loc[features["view"] == candidate["view"]]
    model_scores = scores.loc[scores["model"] == candidate["model"]]
    if view_features.empty or model_scores.empty:
        raise ValueError(f"Candidate has unavailable data: {candidate}")
    joined = view_features.merge(
        model_scores[["sample_id", "label", "score_spoof", "orientation"]],
        on="sample_id",
        suffixes=("_feature", "_score"),
        validate="one_to_one",
    )
    if not (joined["label_feature"] == joined["label_score"]).all():
        raise ValueError(f"Label mismatch joining candidate: {candidate}")
    joined = joined.rename(columns={"label_feature": "label"})
    selected = joined.loc[joined["label"] == int(candidate["class_label"])].copy()
    if selected.empty:
        raise ValueError(f"Candidate class has no joined samples: {candidate}")
    if selected["sample_id"].duplicated().any():
        raise ValueError(f"Candidate stable-ID join is non-unique: {candidate}")
    selected["_bootstrap_cluster"] = bootstrap_cluster_ids(selected)
    return selected


def bootstrap_candidate(candidate: dict[str, object], joined: pd.DataFrame, replicates: int, base_seed: int) -> dict[str, object]:
    """Compute raw and adjusted CIs only for one manifest-selected H1 test."""
    feature = str(candidate["feature"])
    if feature not in joined:
        raise ValueError(f"Candidate feature is absent from feature table: {feature}")
    raw_data = joined[[feature, "score_spoof", "label", "_bootstrap_cluster"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, "score_spoof"])
    partial_data = joined[[
        feature,
        "score_spoof",
        "label",
        "_bootstrap_cluster",
        "duration_seconds",
        "integrated_lufs",
        "speaker_id",
        "attack_id",
    ]].replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, "score_spoof"])
    seed = candidate_seed(base_seed, candidate)
    raw_ci = stratified_cluster_bootstrap_ci(
        raw_data,
        lambda sample: _spearman_metric(sample, feature),
        cluster_column="_bootstrap_cluster",
        strata_columns=["label"],
        replicates=replicates,
        seed=seed,
    )
    partial_ci = stratified_cluster_bootstrap_ci(
        partial_data,
        lambda sample: _partial_spearman_metric(sample, feature),
        cluster_column="_bootstrap_cluster",
        strata_columns=["label"],
        replicates=replicates,
        seed=seed,
    )
    result: dict[str, object] = {
        "bootstrap_method": "stratified_clustered_percentile",
        "bootstrap_cluster_unit": "speaker_id_when_available_else_source_utterance",
        "bootstrap_strata": "label (within manifest-selected class)",
        "bootstrap_seed": seed,
        "bootstrap_replicates_requested": replicates,
    }
    for prefix, metrics in [("spearman", raw_ci), ("partial_spearman", partial_ci)]:
        for key, value in metrics.items():
            if key == "bootstrap_replicates_requested":
                continue
            result[f"{prefix}_{key.removeprefix('bootstrap_')}"] = value
    return result


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
    associations["analysis_stage"] = "screen"
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


def run_bootstrap_confirmation(
    candidates: pd.DataFrame,
    inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    screen_results: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    candidate_manifest: Path,
) -> pd.DataFrame:
    """Confirm only exact manifest rows; there is intentionally no top-k path."""
    screen_index = screen_results.set_index(CANDIDATE_KEY)
    if not screen_index.index.is_unique:
        raise ValueError(f"Screen results duplicate an exact H1 test: {CANDIDATE_KEY}")
    manifest_hash = hashlib.sha256(candidate_manifest.read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        key = tuple(candidate[field] for field in CANDIDATE_KEY)
        if key not in screen_index.index:
            raise ValueError(f"Frozen candidate is absent from the requested screen results: {key}")
        dataset = str(candidate["dataset"])
        if dataset not in inputs:
            raise ValueError(f"Frozen candidate dataset was not requested: {dataset}")
        features, scores = inputs[dataset]
        joined = candidate_join(features, scores, candidate)
        screen_row = screen_index.loc[key].to_dict()
        row = {**candidate, **screen_row}
        row["analysis_stage"] = "confirmation_bootstrap"
        row["candidate_manifest_sha256"] = manifest_hash
        row.update(bootstrap_candidate(candidate, joined, replicates, seed))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--feature-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features")
    parser.add_argument("--score-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/scores")
    parser.add_argument("--output-dir", default="experiments/h1_feature_association/results")
    parser.add_argument(
        "--bootstrap-candidates",
        type=Path,
        help="Explicit frozen CSV/Parquet candidate manifest; no candidates are inferred from screening results.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2609)
    args = parser.parse_args()

    feature_root, score_root, output_dir = Path(args.feature_root), Path(args.score_root), Path(args.output_dir)
    requested_datasets = list(dict.fromkeys(args.dataset))
    if len(requested_datasets) != len(args.dataset):
        raise ValueError("Each --dataset may be requested at most once per invocation")
    # Flat result names caused a later single-dataset run to overwrite an
    # earlier corpus. Keep each corpus immutable by default; a combined view is
    # written only when the caller explicitly requests multiple datasets.
    run_output_dir = output_dir / ("combined" if len(requested_datasets) > 1 else requested_datasets[0])
    run_output_dir.mkdir(parents=True, exist_ok=True)
    all_associations: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dataset in requested_datasets:
        features, scores = load_inputs(feature_root, score_root, dataset)
        inputs[dataset] = (features, scores)
        associations, labels, report = analyze_dataset(dataset, features, scores)
        all_associations.append(associations)
        all_labels.append(labels)
        reports.append(report)
    association_output = pd.concat(all_associations, ignore_index=True)
    label_output = pd.concat(all_labels, ignore_index=True)
    association_output.to_csv(run_output_dir / "association_summary.csv", index=False)
    label_output.to_csv(run_output_dir / "feature_label_metrics.csv", index=False)
    (run_output_dir / "association_join_report.json").write_text(json.dumps(reports, indent=2, sort_keys=True))
    if args.bootstrap_candidates is not None:
        candidates = load_frozen_candidates(args.bootstrap_candidates)
        confirmation_output = run_bootstrap_confirmation(
            candidates,
            inputs,
            association_output,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
            candidate_manifest=args.bootstrap_candidates,
        )
        confirmation_path = run_output_dir / "association_confirmation_bootstrap.csv"
        confirmation_output.to_csv(confirmation_path, index=False)
        print(f"wrote {confirmation_path} ({len(confirmation_output)} manifest-selected rows)")
    print(f"wrote {run_output_dir / 'association_summary.csv'} ({len(association_output)} rows)")


if __name__ == "__main__":
    main()
