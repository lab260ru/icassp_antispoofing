from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from src.arena_io import AudioRecord
from src.h2_asr_wer import WhisperQualityGateConfig
from src.h2_pre_score_pairs import (
    ArmDefinition,
    arm_definitions_frame,
    evaluate_quality_pair,
    freeze_input_manifest,
    registered_crest_factor_arms,
)
from src.h2_quality_runner import (
    OriginalTranscriptCache,
    _base_failure_row,
    pair_id_for,
    persist_pair_checkpoint,
    quality_run_paths,
    read_pair_checkpoint,
    run_quality_panel,
    validate_frozen_inputs,
)


SAMPLE_RATE = 16_000


class CountingTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, _audio: np.ndarray) -> str:
        self.calls += 1
        return "stable reference transcript"


def _manifest_frame() -> pd.DataFrame:
    source = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "sample_id": "clip_a",
                "source_id": "clip_a",
                "label": 0,
                "dataset_revision": "rev-toy",
            }
        ]
    )
    return freeze_input_manifest(source, per_label=1, seed=2609, input_csv_sha256="a" * 64).rows


def _write_frozen_inputs(root: Path, manifest: pd.DataFrame | None = None) -> tuple[Path, Path, Path]:
    input_manifest = root / "input_manifest.csv"
    ledger = root / "arm_ledger.json"
    index = root / "arena-index.yaml"
    (manifest if manifest is not None else _manifest_frame()).to_csv(input_manifest, index=False)
    ledger.write_text(json.dumps(arm_definitions_frame().to_dict(orient="records"), indent=2), encoding="utf-8")
    index.write_text(
        yaml.safe_dump({"datasets": {"toy": {"repo_id": "example/toy", "revision": "rev-toy"}}}),
        encoding="utf-8",
    )
    return input_manifest, ledger, index


def _waveform() -> np.ndarray:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    waveform = (0.08 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    waveform[2_000] = 0.75
    waveform[7_000] = -0.75
    return waveform


def _audio_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, _waveform(), SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _toy_audio_iterator(_dataset: dict[str, object], selected: pd.DataFrame):
    for row in selected.to_dict(orient="records"):
        yield AudioRecord(
            sample_id=str(row["sample_id"]),
            label=int(row["label"]),
            source_id=str(row["source_id"]),
            audio_bytes=_audio_bytes(),
            notes={},
        )


def _config() -> WhisperQualityGateConfig:
    return WhisperQualityGateConfig(model_root=Path("/nonexistent/test-whisper"))


def test_validate_frozen_inputs_checks_marker_ledger_and_dataset_revision(tmp_path: Path) -> None:
    manifest, ledger, index = _write_frozen_inputs(tmp_path)
    validated = validate_frozen_inputs(manifest, ledger, index)
    assert validated.manifest["manifest_version"].eq("h2_pre_score_pairs_v1").all()
    assert [arm.arm_id for arm in validated.arms] == sorted(arm.arm_id for arm in registered_crest_factor_arms())

    malformed = pd.read_csv(manifest)
    malformed.loc[0, "manifest_version"] = "not_frozen"
    malformed.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="marker"):
        validate_frozen_inputs(manifest, ledger, index)

    _manifest_frame().to_csv(manifest, index=False)
    ledger_rows = json.loads(ledger.read_text(encoding="utf-8"))
    ledger_rows[0]["definition_sha256"] = "0" * 64
    ledger.write_text(json.dumps(ledger_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_inputs(manifest, ledger, index)

    ledger.write_text(json.dumps(arm_definitions_frame().to_dict(orient="records")), encoding="utf-8")
    index.write_text(yaml.safe_dump({"datasets": {"toy": {"repo_id": "example/toy", "revision": "other"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_frozen_inputs(manifest, ledger, index)


def test_pair_id_matches_quality_pair_and_checkpoint_refuses_conflict(tmp_path: Path) -> None:
    manifest_row = _manifest_frame().iloc[0].to_dict()
    arm = next(item for item in registered_crest_factor_arms() if item.arm_id == "polarity")
    quality = evaluate_quality_pair(
        manifest_row,
        audio=_waveform(),
        sample_rate=SAMPLE_RATE,
        arm=arm,
        transcribe=lambda _audio, _rate: "stable reference transcript",
        stoi_measure=lambda _source, _transformed, _rate: 0.99,
        feature_extractor=lambda _audio, _rate: {"crest_factor_db": 9.0, **{name: 0.0 for name in []}},
    )
    # The synthetic extractor is intentionally incomplete and creates a retained failure row.
    assert quality["pair_id"] == pair_id_for(manifest_row, arm)
    checkpoint = tmp_path / f"{quality['pair_id']}.json"
    assert persist_pair_checkpoint(checkpoint, quality) is True
    assert persist_pair_checkpoint(checkpoint, quality) is False
    assert read_pair_checkpoint(checkpoint)["pair_id"] == quality["pair_id"]
    conflicting = dict(quality)
    conflicting["failure_reasons_json"] = '["conflict"]'
    with pytest.raises(RuntimeError, match="conflicting"):
        persist_pair_checkpoint(checkpoint, conflicting)


def test_original_transcript_cache_reuses_one_original_transcription(tmp_path: Path) -> None:
    row = _manifest_frame().iloc[0].to_dict()
    transcriber = CountingTranscriber()
    cache = OriginalTranscriptCache(tmp_path, transcriber, {"runtime": "test"})
    assert cache.get_or_create(row, _waveform(), SAMPLE_RATE) == "stable reference transcript"
    assert cache.get_or_create(row, _waveform(), SAMPLE_RATE) == "stable reference transcript"
    assert transcriber.calls == 1


def test_resumable_runner_writes_only_four_frozen_arms_and_caches_original(tmp_path: Path) -> None:
    manifest_path, ledger, index = _write_frozen_inputs(tmp_path)
    validated = validate_frozen_inputs(manifest_path, ledger, index)
    paths = quality_run_paths(tmp_path / "hdd", "pilot_001", tmp_path / "repo")
    first = CountingTranscriber()
    summary = run_quality_panel(
        validated,
        paths=paths,
        run_id="pilot_001",
        mode="pilot_not_panel_gate",
        limit=1,
        whisper_config=_config(),
        transcriber=first,
        audio_iterator=_toy_audio_iterator,
    )
    assert summary["expected_pairs"] == 4
    assert summary["completed_pairs"] == 4
    assert summary["panel_gate_status"] == "not_panel_gate"
    # One original + one transformed transcript for each of the four arms.
    assert first.calls == 5
    assert paths.full_table_path.is_file()
    assert paths.repo_summary_path.is_file()
    assert len(list(paths.pair_rows_dir.glob("*.json"))) == 4

    resumed = CountingTranscriber()
    second = run_quality_panel(
        validated,
        paths=paths,
        run_id="pilot_001",
        mode="pilot_not_panel_gate",
        limit=1,
        whisper_config=_config(),
        transcriber=resumed,
        audio_iterator=_toy_audio_iterator,
    )
    assert second["newly_written_pairs"] == 0
    assert resumed.calls == 0
