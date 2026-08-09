from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts.analyze_associations import candidate_seed, load_frozen_candidates, run_bootstrap_confirmation
from src.statistics import partial_spearman, stratified_cluster_bootstrap_ci


def test_partial_spearman_never_conditions_a_feature_on_itself() -> None:
    index = np.arange(48, dtype=float)
    frame = pd.DataFrame(
        {
            "integrated_lufs": index + np.sin(index),
            "score_spoof": 0.5 * index + np.cos(index),
            "duration_seconds": 1.0 + (index % 5),
            "speaker_id": [f"spk-{int(value) % 4}" for value in index],
            "attack_id": [f"a-{int(value) % 2}" for value in index],
        }
    )
    n, rho, p_value = partial_spearman(
        frame,
        "integrated_lufs",
        "score_spoof",
        numeric_controls=["duration_seconds", "integrated_lufs", "duration_seconds"],
        categorical_controls=["speaker_id", "attack_id", "speaker_id"],
    )
    assert n == len(frame)
    assert np.isfinite(rho)
    assert np.isfinite(p_value)


def test_stratified_cluster_bootstrap_is_deterministic_and_reports_clusters() -> None:
    rows: list[dict[str, object]] = []
    for label in (0, 1):
        for cluster in range(18):
            for observation in range(2):
                value = cluster + 0.1 * label + 0.01 * observation
                rows.append(
                    {
                        "label": label,
                        "cluster": f"speaker-{cluster}",
                        "feature": value,
                        "score_spoof": 0.8 * value + 0.03 * observation,
                    }
                )
    frame = pd.DataFrame(rows)

    def metric(sample: pd.DataFrame) -> float:
        return float(stats.spearmanr(sample["feature"], sample["score_spoof"]).statistic)

    first = stratified_cluster_bootstrap_ci(
        frame,
        metric,
        cluster_column="cluster",
        strata_columns=["label"],
        replicates=300,
        seed=2609,
    )
    second = stratified_cluster_bootstrap_ci(
        frame,
        metric,
        cluster_column="cluster",
        strata_columns=["label"],
        replicates=300,
        seed=2609,
    )
    assert first == second
    assert first["bootstrap_n_strata"] == 2
    assert first["bootstrap_n_clusters"] == 36
    assert first["bootstrap_replicates_valid"] == 300
    assert float(first["ci_low"]) > 0.0
    assert first["ci_excludes_zero"] is True


def test_candidate_manifest_requires_frozen_discovery_provenance(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "dataset": "ASVspoof2019_LA",
                "model": "Spectra-AASIST",
                "view": "full_waveform",
                "feature": "crest_factor_db",
                "class_label": 1,
                "selection_status": "frozen",
                "selection_split": "discovery",
                "selection_basis": "predeclared outer-loop decision",
                "frozen_at_utc": "2026-08-09T12:00:00Z",
            }
        ]
    ).to_csv(path, index=False)
    candidates = load_frozen_candidates(path)
    candidate = candidates.iloc[0].to_dict()
    assert candidate_seed(2609, candidate) == candidate_seed(2609, candidate)

    candidates.loc[0, "selection_split"] = "confirmation"
    candidates.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not confirmation data"):
        load_frozen_candidates(path)


def test_bootstrap_confirmation_uses_only_the_frozen_manifest(tmp_path: pytest.TempPathFactory) -> None:
    rows: list[dict[str, object]] = []
    for index in range(24):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "source_id": f"source-{index}",
                "label": 1,
                "view": "full_waveform",
                "speaker_id": f"speaker-{index % 4}",
                "attack_id": f"attack-{index % 2}",
                "duration_seconds": 1.0 + index % 3,
                "integrated_lufs": -24.0 + index / 10,
                "crest_factor_db": index / 3,
            }
        )
    features = pd.DataFrame(rows)
    scores = pd.DataFrame(
        {
            "sample_id": features["sample_id"],
            "label": features["label"],
            "score_spoof": 0.6 * features["crest_factor_db"] + 0.1,
            "model": "Spectra-AASIST",
            "orientation": "raw_is_spoof",
        }
    )
    path = tmp_path / "frozen.csv"
    pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "Spectra-AASIST",
                "view": "full_waveform",
                "feature": "crest_factor_db",
                "class_label": 1,
                "selection_status": "frozen",
                "selection_split": "discovery",
                "selection_basis": "predeclared outer-loop decision",
                "frozen_at_utc": "2026-08-09T12:00:00Z",
            }
        ]
    ).to_csv(path, index=False)
    candidates = load_frozen_candidates(path)
    screen = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "Spectra-AASIST",
                "view": "full_waveform",
                "feature": "crest_factor_db",
                "class_label": 1,
                "spearman_rho": 1.0,
            }
        ]
    )
    output = run_bootstrap_confirmation(
        candidates,
        {"toy": (features, scores)},
        screen,
        replicates=50,
        seed=2609,
        candidate_manifest=path,
    )
    assert output["analysis_stage"].tolist() == ["confirmation_bootstrap"]
    assert output["spearman_replicates_valid"].tolist() == [50]
    assert output["partial_spearman_replicates_valid"].tolist() == [50]
    assert output["candidate_manifest_sha256"].str.len().tolist() == [64]
