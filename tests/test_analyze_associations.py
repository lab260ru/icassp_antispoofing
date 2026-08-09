from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_associations import analyze_dataset
from src.audio_features import FEATURE_NAMES


def test_label_auroc_excludes_nonfinite_feature_values() -> None:
    """A diagnostic AUROC must not abort an otherwise valid H1 screen."""
    n_per_label = 14
    sample_ids = [f"sample-{index}" for index in range(2 * n_per_label)]
    labels = [0] * n_per_label + [1] * n_per_label
    feature_name = FEATURE_NAMES[0]
    features: dict[str, object] = {
        "sample_id": sample_ids,
        "label": labels,
        "view": ["full_waveform"] * len(sample_ids),
        "duration_seconds": np.linspace(1.0, 2.0, len(sample_ids)),
        "integrated_lufs": np.linspace(-25.0, -15.0, len(sample_ids)),
        "speaker_id": [f"speaker-{index // 2}" for index in range(len(sample_ids))],
        "attack_id": ["attack-a"] * len(sample_ids),
    }
    for offset, name in enumerate(FEATURE_NAMES):
        features[name] = np.linspace(float(offset), float(offset + 1), len(sample_ids))
    features[feature_name] = np.asarray(features[feature_name], dtype=float)
    features[feature_name][0] = np.inf
    features[feature_name][n_per_label] = -np.inf
    scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "model": ["Spectra-AASIST"] * len(sample_ids),
            "score_spoof": np.linspace(-1.0, 1.0, len(sample_ids)),
            "orientation": ["higher_is_spoof"] * len(sample_ids),
        }
    )

    associations, label_metrics, _ = analyze_dataset("synthetic", pd.DataFrame(features), scores)

    label_row = label_metrics.loc[label_metrics["feature"] == feature_name].iloc[0]
    assert label_row["n"] == 2 * n_per_label - 2
    assert np.isfinite(label_row["label_auroc"])
    # The same finite-only convention is already used by score associations.
    assert associations.loc[associations["feature"] == feature_name, "n"].tolist() == [n_per_label - 1] * 2
