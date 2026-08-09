from __future__ import annotations

import pytest

from src.arena_io import audio_row_identity


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
