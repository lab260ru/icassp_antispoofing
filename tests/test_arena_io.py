from __future__ import annotations

import pytest
import numpy as np

from src.arena_io import audio_row_identity
from src.arena_io import AudioRecord
import scripts.extract_features as extract_features


def test_audio_row_identity_prefers_explicit_notes_utterance_id_over_positional_path() -> None:
    sample_id, notes = audio_row_identity(
        {
            "path": "0.wav",
            "notes": '{"utterance_id": "ITW_0", "speaker": "example"}',
        }
    )
    assert sample_id == "ITW_0"
    assert notes["speaker"] == "example"


def test_audio_row_identity_falls_back_to_path_and_rejects_missing_identity() -> None:
    sample_id, notes = audio_row_identity({"path": "nested/DF_E_1.flac", "notes": None})
    assert sample_id == "DF_E_1"
    assert notes == {}
    with pytest.raises(ValueError, match="neither an explicit stable ID"):
        audio_row_identity({"path": "", "notes": ""})


def test_feature_rows_preserve_explicit_in_the_wild_speaker_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    record = AudioRecord(
        sample_id="ITW_0",
        source_id="ITW_0",
        label=1,
        audio_bytes=b"unused",
        notes={"speaker": "example speaker"},
    )
    monkeypatch.setattr(extract_features, "decode_audio", lambda _: (np.ones(16, dtype=np.float32), 16_000))
    monkeypatch.setattr(extract_features, "waveform_views", lambda *_: {"full_waveform": np.ones(16, dtype=np.float32)})
    monkeypatch.setattr(extract_features, "compute_features", lambda _: {name: 0.0 for name in extract_features.FEATURE_NAMES})
    rows = extract_features.process_record(record)
    assert rows[0]["speaker_id"] == "example speaker"
