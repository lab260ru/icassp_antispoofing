from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audio_features import FEATURE_NAMES
from src.h2_pre_score_pairs import (
    QualityGateConfig,
    apply_registered_arm,
    arm_definitions_frame,
    assert_score_independent_columns,
    evaluate_quality_pair,
    freeze_input_manifest,
    quality_rows_frame,
    registered_crest_factor_arms,
    ArmDefinition,
)
from scripts.freeze_h2_pre_score_manifest import _new_path


SAMPLE_RATE = 16_000


def _input_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("dataset_a", "dataset_b"):
        for label in (0, 1):
            for index in range(5):
                rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": f"{dataset}_{label}_{index}",
                        "label": label,
                        "source_id": f"source_{dataset}_{label}_{index}",
                    }
                )
    return pd.DataFrame(rows)


def _crest_factor(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(np.max(np.abs(audio)) / rms))


def _feature_extractor(audio: np.ndarray, _sample_rate: int) -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["crest_factor_db"] = _crest_factor(audio)
    return features


def _speech_like_waveform() -> np.ndarray:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    waveform = (0.08 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    waveform[2_000] = 0.75
    waveform[7_000] = -0.75
    return waveform


def test_freeze_is_balanced_seeded_and_independent_of_csv_row_order() -> None:
    source = _input_rows()
    first = freeze_input_manifest(source, per_label=3, seed=2609, input_csv_sha256="a" * 64)
    second = freeze_input_manifest(source.sample(frac=1.0, random_state=7), per_label=3, seed=2609)
    assert len(first.rows) == 12
    assert first.rows.groupby(["dataset", "label"]).size().eq(3).all()
    assert first.rows[["dataset", "label", "sample_id", "selection_rank"]].equals(
        second.rows[["dataset", "label", "sample_id", "selection_rank"]]
    )
    assert first.provenance["detector_scores_read"] is False
    assert first.provenance["detector_scoring_allowed"] is False
    assert set(first.rows["source_input_csv_sha256"]) == {"a" * 64}
    assert set(first.rows["selection_rank"]) == {1, 2, 3}


def test_freeze_rejects_detector_responses_and_undersized_groups() -> None:
    bad = _input_rows().assign(detector_score=0.1)
    with pytest.raises(ValueError, match="detector-response"):
        freeze_input_manifest(bad, per_label=3, seed=2609)
    with pytest.raises(ValueError, match="undersized"):
        freeze_input_manifest(_input_rows(), per_label=6, seed=2609)
    with pytest.raises(ValueError, match="detector-response"):
        assert_score_independent_columns(["dataset", "sample_id", "raw_logit"])


def test_frozen_cli_paths_canonicalize_aliases_before_collision_checks(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    assert _new_path(str(target)) == target.resolve()
    assert _new_path(str(tmp_path / "." / "artifact.json")) == target.resolve()


def test_registered_arms_are_fixed_crest_interventions_and_controls() -> None:
    arms = {arm.arm_id: arm for arm in registered_crest_factor_arms()}
    assert set(arms) == {"drc_cf3", "drc_cf6", "small_gain_plus_0p1db", "polarity"}
    assert arms["drc_cf3"].parameters == {"target_reduction_db": 3.0}
    assert arms["drc_cf6"].parameters == {"target_reduction_db": 6.0}
    assert arms["drc_cf3"].target_direction == "decrease"
    assert arms["small_gain_plus_0p1db"].negative_control is True
    frame = arm_definitions_frame()
    assert frame["definition_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_quality_pair_records_all_gates_without_detector_outputs() -> None:
    frozen = freeze_input_manifest(_input_rows(), per_label=3, seed=2609)
    row = frozen.rows.iloc[0].to_dict()
    drc = next(arm for arm in registered_crest_factor_arms() if arm.arm_id == "drc_cf3")
    quality = evaluate_quality_pair(
        row,
        audio=_speech_like_waveform(),
        sample_rate=SAMPLE_RATE,
        arm=drc,
        transcribe=lambda _audio, _rate: "same transcript",
        stoi_measure=lambda _original, _transformed, _rate: 0.99,
        feature_extractor=_feature_extractor,
    )
    assert quality["detector_stage"] == "blocked_pending_quality_freeze"
    assert quality["pass_stoi"] is True
    assert quality["pass_wer"] is True
    assert quality["pass_clipping"] is True
    assert quality["pass_target_direction"] is True
    assert quality["target_feature_after"] < quality["target_feature_before"]
    assert len(json.loads(quality["all_feature_deltas_json"])) == len(FEATURE_NAMES)
    assert not any("score" in key.casefold() for key in quality)
    table = quality_rows_frame([quality])
    assert len(table) == 1
    # This particular short DRC waveform exposes unavailable output loudness;
    # it must survive as an explicit failed quality row rather than be dropped.
    assert bool(table.loc[0, "retained"]) is False
    assert "loudness_gate_failed_or_unavailable" in json.loads(table.loc[0, "failure_reasons_json"])


def test_quality_pair_preserves_asr_failure_row_and_rejects_response_columns() -> None:
    frozen = freeze_input_manifest(_input_rows(), per_label=3, seed=2609)
    row = frozen.rows.iloc[0].to_dict()
    polarity = next(arm for arm in registered_crest_factor_arms() if arm.arm_id == "polarity")
    quality = evaluate_quality_pair(
        row,
        audio=_speech_like_waveform(),
        sample_rate=SAMPLE_RATE,
        arm=polarity,
        transcribe=lambda audio, _rate: "original words" if audio[2_000] >= 0 else "different words",
        stoi_measure=lambda _original, _transformed, _rate: 0.99,
        feature_extractor=_feature_extractor,
        gates=QualityGateConfig(),
    )
    assert quality["retained"] is False
    assert quality["pass_wer"] is False
    assert "wer_gate_failed" in json.loads(quality["failure_reasons_json"])
    assert len(quality_rows_frame([quality])) == 1
    quality["accidental_score"] = 0.7
    with pytest.raises(ValueError, match="detector-response"):
        quality_rows_frame([quality])


def test_clipping_gate_rejects_only_transform_induced_clipping() -> None:
    frozen = freeze_input_manifest(_input_rows(), per_label=3, seed=2609)
    row = frozen.rows.iloc[0].to_dict()
    polarity = next(arm for arm in registered_crest_factor_arms() if arm.arm_id == "polarity")
    clipped_source = _speech_like_waveform()
    clipped_source[0] = 1.0
    quality = evaluate_quality_pair(
        row,
        audio=clipped_source,
        sample_rate=SAMPLE_RATE,
        arm=polarity,
        transcribe=lambda _audio, _rate: "same transcript",
        stoi_measure=lambda _original, _transformed, _rate: 0.99,
        feature_extractor=_feature_extractor,
    )
    assert quality["original_clipping_fraction"] > 0.0
    assert quality["transformed_clipping_fraction"] == pytest.approx(quality["original_clipping_fraction"])
    assert quality["added_clipping_fraction"] == pytest.approx(0.0)
    assert quality["pass_clipping"] is True


def test_generic_quality_only_arm_dispatch_supports_h2b_transform_families() -> None:
    waveform = _speech_like_waveform()
    arms = (
        ArmDefinition(
            arm_id="tilt_plus",
            transform="spectral_tilt",
            parameters={"tilt_db_per_octave": 0.5},
            negative_control=False,
            target_feature="spectral_slope_db_per_khz",
            target_direction="increase",
            target_tolerance=0.0,
        ),
        ArmDefinition(
            arm_id="allpass",
            transform="allpass_phase",
            parameters={"coefficient": 0.1, "stages": 1.0},
            negative_control=False,
            target_feature="group_delay_var",
            target_direction="absolute_change",
            target_tolerance=1e-8,
        ),
        ArmDefinition(
            arm_id="endpoint",
            transform="endpoint_silence_fixed_length",
            parameters={"frame_ms": 10.0, "hop_ms": 10.0, "guard_ms": 0.0},
            negative_control=False,
            target_feature="silence_fraction",
            target_direction="increase",
            target_tolerance=0.0,
        ),
    )
    for arm in arms:
        transformed = apply_registered_arm(waveform, SAMPLE_RATE, arm)
        assert transformed.waveform.shape == waveform.shape
        assert transformed.waveform.dtype == np.float32
        assert np.isfinite(transformed.waveform).all()
    malformed = ArmDefinition(
        arm_id="bad-allpass",
        transform="allpass_phase",
        parameters={"coefficient": 0.1, "stages": 1.5},
        negative_control=False,
        target_feature="group_delay_var",
        target_direction="absolute_change",
        target_tolerance=1e-8,
    )
    with pytest.raises(ValueError, match="positive integer"):
        apply_registered_arm(waveform, SAMPLE_RATE, malformed)
