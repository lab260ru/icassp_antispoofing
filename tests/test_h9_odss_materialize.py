from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from src.h9_odss_materialize import (
    EXCLUDED_COLUMNS,
    H9SourceFreezeExpectation,
    PAIR_COLUMNS,
    RANDOM_PAIR_COLUMNS,
    TRIAL_COLUMNS,
    _sha256_json,
    canonical_mono16k_pcm_fingerprint,
    materialize_h9_odss_source,
)
from src.h9_pcr_training import REQUIRED_SOURCE_COLUMNS, load_source_manifest


REVISION = "1968e6d0ef141c4572073695bdc1d17a8706177f"


def _wav_bytes(value: float, *, sample_rate: int = 16_000, stereo: bool = False) -> bytes:
    samples = np.full(40, value, dtype=np.float32)
    audio = np.stack([samples, -samples], axis=1) if stereo else samples
    stream = io.BytesIO()
    sf.write(stream, audio, sample_rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _synthetic_freeze(tmp_path: Path) -> tuple[Path, Path, H9SourceFreezeExpectation, dict[str, bytes]]:
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    raw_dir = tmp_path / "datasets" / "odss" / REVISION / "data"
    raw_dir.mkdir(parents=True)
    trial_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    # Two pairs per split supply content-unmatched B2 counterparts. This small
    # fixture is injected only into the library test seam; the real CLI is
    # fixed to the 23,883-row production freeze.
    for split, index in (("train", 0), ("train", 1), ("dev", 2), ("dev", 3)):
        speaker = f"s{index}"
        content = f"hifi-tts/{speaker}/u{index}"
        voice = f"hifi-tts/{speaker}"
        group = f"{voice}::{content}"
        natural_id = f"natural__hifi-tts__{speaker}__u{index}"
        natural_path = f"natural/hifi-tts/{speaker}/u{index}.wav"
        for generator, label in (("natural", 0), ("vits", 1), ("fastpitch-hifigan", 1)):
            sample_id = f"{generator}__hifi-tts__{speaker}__u{index}"
            relative_path = f"{generator}/hifi-tts/{speaker}/u{index}.wav"
            payloads[relative_path] = _wav_bytes(
                (index + 1) / 20.0,
                sample_rate=8_000 if index == 0 else 16_000,
                stereo=index == 0,
            )
            trial_rows.append(
                {
                    "utterance_id": sample_id,
                    "relative_path": relative_path,
                    "label": label,
                    "label_name": "bonafide" if label == 0 else "spoof",
                    "generator": generator,
                    "source_corpus": "hifi-tts",
                    "language": "en",
                    "speaker": speaker,
                    "voice_key": voice,
                    "content_key": content,
                    "group_key": group,
                    "split": split,
                    "n_counterfactual_partners": 2 if label == 0 else 1,
                    "source_pool": "shared_b1_b2_p_complete_matched_only",
                }
            )
            if label == 1:
                pair_rows.append(
                    {
                        "pair_id": content,
                        "split": split,
                        "group_key": group,
                        "voice_key": voice,
                        "content_key": content,
                        "source_corpus": "hifi-tts",
                        "language": "en",
                        "bona_utterance_id": natural_id,
                        "bona_relative_path": natural_path,
                        "spoof_utterance_id": sample_id,
                        "spoof_relative_path": relative_path,
                        "spoof_generator": generator,
                    }
                )
    trials = pd.DataFrame(trial_rows, columns=TRIAL_COLUMNS).sort_values(
        ["split", "content_key", "generator"], kind="stable"
    ).reset_index(drop=True)
    pairs = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS).sort_values(
        ["split", "content_key", "spoof_generator"], kind="stable"
    ).reset_index(drop=True)
    natural_by_content = {
        str(row.content_key): row
        for row in trials.loc[trials.label.eq(0)].itertuples(index=False)
    }
    alternate = {
        "hifi-tts/s0/u0": "hifi-tts/s1/u1",
        "hifi-tts/s1/u1": "hifi-tts/s0/u0",
        "hifi-tts/s2/u2": "hifi-tts/s3/u3",
        "hifi-tts/s3/u3": "hifi-tts/s2/u2",
    }
    random_rows: list[dict[str, object]] = []
    for edge in pairs.itertuples(index=False):
        natural = natural_by_content[alternate[str(edge.content_key)]]
        random_rows.append(
            {
                "pair_id": edge.pair_id,
                "split": edge.split,
                "group_key": edge.group_key,
                "voice_key": edge.voice_key,
                "content_key": edge.content_key,
                "source_corpus": edge.source_corpus,
                "language": edge.language,
                "spoof_generator": edge.spoof_generator,
                "random_bona_utterance_id": natural.utterance_id,
                "random_bona_relative_path": natural.relative_path,
                "random_bona_content_key": natural.content_key,
                "random_bona_voice_key": natural.voice_key,
                "random_bona_split": natural.split,
                "random_bona_language": natural.language,
                "random_bona_source_corpus": natural.source_corpus,
                "spoof_utterance_id": edge.spoof_utterance_id,
                "spoof_relative_path": edge.spoof_relative_path,
            }
        )
    random_pairs = pd.DataFrame(random_rows, columns=RANDOM_PAIR_COLUMNS).sort_values(
        ["split", "content_key", "spoof_generator"], kind="stable"
    ).reset_index(drop=True)
    excluded = pd.DataFrame(
        [
            {
                "utterance_id": "vits__vctk__x__unmatched",
                "relative_path": "vits/vctk/x/unmatched.wav",
                "label": 1,
                "label_name": "spoof",
                "generator": "vits",
                "source_corpus": "vctk",
                "language": "en",
                "speaker": "x",
                "voice_key": "vctk/x",
                "content_key": "vctk/x/unmatched",
                "group_key": "vctk/x::vctk/x/unmatched",
                "exclusion_reason": "missing_bonafide",
            }
        ],
        columns=EXCLUDED_COLUMNS,
    )
    tables = {
        "h9_odss_source_trials.csv": trials,
        "h9_odss_source_pairs.csv": pairs,
        "h9_odss_b2_random_pairs.csv": random_pairs,
        "h9_odss_excluded_unmatched.csv": excluded,
    }
    for name, table in tables.items():
        table.to_csv(freeze_dir / name, index=False)
    hashes = {name: hashlib.sha256((freeze_dir / name).read_bytes()).hexdigest() for name in tables}
    aliases = {
        "h9_odss_source_trials.csv": "trials_csv",
        "h9_odss_source_pairs.csv": "pairs_csv",
        "h9_odss_b2_random_pairs.csv": "b2_random_pairs_csv",
        "h9_odss_excluded_unmatched.csv": "excluded_unmatched_csv",
    }
    provenance = {
        "artifact_kind": "h9_odss_paired_counterfactual_source_freeze",
        "version": "h9_odss_paired_counterfactual_source_freeze_v1",
        "source_access": {"target_data_read": False, "audio_opened_or_decoded": False},
        "source": {
            "repo_id": "SpeechAntiSpoofingBenchmarks/ODSS",
            "revision": REVISION,
            "metadata": {"sha256": "synthetic-metadata"},
        },
        "outputs": {
            aliases[name]: {
                "path": str((freeze_dir / name).resolve()),
                "sha256": digest,
                "n_rows": len(tables[name]),
            }
            for name, digest in hashes.items()
        },
        "counts": {
            "input_metadata_rows": 13,
            "retained_trials": len(trials),
            "complete_groups": 4,
            "retained_pairs": len(pairs),
            "random_control_pairs": len(random_pairs),
            "excluded_unmatched_trials": len(excluded),
        },
        "same_source_pool_guard": {
            "conditions": ["B1", "B2", "P"],
            "eligible_trial_ids_sha256": _sha256_json(trials.utterance_id.tolist()),
            "excluded_trial_ids_sha256": _sha256_json(excluded.utterance_id.tolist()),
            "p_edge_keys_sha256": _sha256_json(
                pairs[["pair_id", "spoof_generator"]]
                .sort_values(["pair_id", "spoof_generator"], kind="stable")
                .to_dict(orient="records")
            ),
            "b2_edge_keys_sha256": _sha256_json(
                random_pairs[["pair_id", "spoof_generator"]]
                .sort_values(["pair_id", "spoof_generator"], kind="stable")
                .to_dict(orient="records")
            ),
        },
        "trials_rows_sha256": _sha256_json(trials.to_dict(orient="records")),
        "pairs_rows_sha256": _sha256_json(pairs.to_dict(orient="records")),
        "random_pairs_rows_sha256": _sha256_json(random_pairs.to_dict(orient="records")),
    }
    provenance_path = freeze_dir / "h9_odss_source_freeze.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = H9SourceFreezeExpectation(
        revision=REVISION,
        provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        artifact_hashes=hashes,
        metadata_sha256="synthetic-metadata",
        retained_trials=len(trials),
        complete_groups=4,
        p_edges=len(pairs),
        excluded_unmatched=len(excluded),
        input_metadata_rows=13,
    )
    raw_rows = [
        {
            "path": path,
            "audio": {"bytes": payload},
            "label": int(trials.set_index("relative_path").loc[path, "label"]),
        }
        for path, payload in payloads.items()
    ]
    pq.write_table(pa.Table.from_pylist(raw_rows), raw_dir / "data-00000.parquet")
    return freeze_dir, raw_dir, expected, payloads


def test_materializes_synthetic_source_exactly_and_emits_trainer_schema(tmp_path: Path) -> None:
    freeze_dir, raw_dir, expected, payloads = _synthetic_freeze(tmp_path)
    output = tmp_path / "hdd" / "materialized"
    result = materialize_h9_odss_source(
        freeze_dir=freeze_dir,
        raw_shard_dir=raw_dir,
        output_dir=output,
        allowed_output_root=tmp_path / "hdd",
        _expectation=expected,
    )
    manifest = pd.read_csv(result.source_manifest)
    assert tuple(manifest.columns) == REQUIRED_SOURCE_COLUMNS
    assert len(manifest) == expected.retained_trials
    assert manifest.group_id.eq(manifest.source_corpus + "|" + manifest.speaker_id).all()
    assert manifest.canonical_fingerprint.str.fullmatch(r"[0-9a-f]{64}").all()
    parsed = load_source_manifest(result.source_manifest)
    assert sum(map(len, parsed.paired_eligible_ids.values())) == expected.retained_trials
    audit = pd.read_csv(result.audio_audit)
    for row in audit.itertuples(index=False):
        assert (result.waveforms_dir / f"{row.sample_id}.wav").read_bytes() == payloads[row.relative_path]
        assert row.raw_payload_sha256 == row.extracted_wav_sha256
    ledger = json.loads(result.provenance.read_text(encoding="utf-8"))
    assert ledger["source_access"]["target_data_read"] is False
    assert ledger["sealed_source_freeze"]["p_edges"] == expected.p_edges
    assert ledger["sealed_source_freeze"]["b2_edges"] == expected.p_edges
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_h9_odss_source(
            freeze_dir=freeze_dir,
            raw_shard_dir=raw_dir,
            output_dir=output,
            allowed_output_root=tmp_path / "hdd",
            _expectation=expected,
        )


def test_rejects_hash_drift_before_opening_a_raw_shard(tmp_path: Path) -> None:
    freeze_dir, raw_dir, expected, _payloads = _synthetic_freeze(tmp_path)
    with (freeze_dir / "h9_odss_source_trials.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    output = tmp_path / "hdd" / "must_not_exist"
    with pytest.raises(ValueError, match="artifact SHA-256"):
        materialize_h9_odss_source(
            freeze_dir=freeze_dir,
            raw_shard_dir=raw_dir,
            output_dir=output,
            allowed_output_root=tmp_path / "hdd",
            _expectation=expected,
        )
    assert not output.exists()


def test_rejects_duplicate_raw_frozen_id_and_cleans_staging(tmp_path: Path) -> None:
    freeze_dir, raw_dir, expected, payloads = _synthetic_freeze(tmp_path)
    duplicate = {
        "path": next(iter(payloads)),
        "audio": {"bytes": payloads[next(iter(payloads))]},
        "label": 0 if next(iter(payloads)).startswith("natural/") else 1,
    }
    pq.write_table(pa.Table.from_pylist([duplicate]), raw_dir / "data-00001.parquet")
    output = tmp_path / "hdd" / "bad_raw"
    with pytest.raises(ValueError, match="duplicate frozen sample"):
        materialize_h9_odss_source(
            freeze_dir=freeze_dir,
            raw_shard_dir=raw_dir,
            output_dir=output,
            allowed_output_root=tmp_path / "hdd",
            _expectation=expected,
        )
    assert not output.exists()


def test_canonical_fingerprint_is_deterministic_and_rejects_non_wav() -> None:
    payload = _wav_bytes(0.25, sample_rate=8_000, stereo=True)
    digest, frames, byte_count = canonical_mono16k_pcm_fingerprint(payload)
    assert digest == canonical_mono16k_pcm_fingerprint(payload)[0]
    assert frames > 0 and byte_count == 2 * frames
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        canonical_mono16k_pcm_fingerprint(b"not-a-wav")
