"""Fail-closed source-only ODSS materialization for H9-PCR.

This adapter validates the sealed H9 source freeze before opening any audio
shard.  It copies only frozen RIFF/WAVE payloads unchanged, then emits the
exact canonical source-manifest schema expected by the H9 trainer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from src.h9_odss_pairs import CORPUS_TO_LANGUAGE, HDD_ROOT, H9_ODSS_PAIRING_VERSION, H9_ODSS_REPO_ID, H9_ODSS_REVISION, sha256_file
from src.h9_pcr_training import REQUIRED_SOURCE_COLUMNS, load_source_manifest


H9_ODSS_MATERIALIZATION_VERSION = "h9_odss_paired_counterfactual_source_materialization_v1"
SOURCE_POOL_NAME = "shared_b1_b2_p_complete_matched_only"
CANONICAL_AUDIO_POLICY: dict[str, Any] = {
    "kind": "h9_canonical_mono16k_pcm_sha256",
    "version": 1,
    "decode": "soundfile.read(BytesIO(raw_wav), dtype=float32, always_2d=True)",
    "channels": "arithmetic_mean_over_channels_in_float64_then_cast_float32",
    "resample": "scipy.signal.resample_poly(reduced_16000_over_source_rate, window=('kaiser',5.0), padtype='constant')",
    "pcm": "clip_to_minus1_plus1; rint(sample*32767); little_endian_signed_int16",
    "header": "ASCII policy tag plus uint64 little-endian frame count before PCM bytes",
    "sample_rate_hz": 16_000,
}

TRIAL_COLUMNS = (
    "utterance_id", "relative_path", "label", "label_name", "generator", "source_corpus", "language", "speaker",
    "voice_key", "content_key", "group_key", "split", "n_counterfactual_partners", "source_pool",
)
PAIR_COLUMNS = (
    "pair_id", "split", "group_key", "voice_key", "content_key", "source_corpus", "language", "bona_utterance_id",
    "bona_relative_path", "spoof_utterance_id", "spoof_relative_path", "spoof_generator",
)
RANDOM_PAIR_COLUMNS = (
    "pair_id", "split", "group_key", "voice_key", "content_key", "source_corpus", "language", "spoof_generator",
    "random_bona_utterance_id", "random_bona_relative_path", "random_bona_content_key", "random_bona_voice_key",
    "random_bona_split", "random_bona_language", "random_bona_source_corpus", "spoof_utterance_id", "spoof_relative_path",
)
EXCLUDED_COLUMNS = (
    "utterance_id", "relative_path", "label", "label_name", "generator", "source_corpus", "language", "speaker",
    "voice_key", "content_key", "group_key", "exclusion_reason",
)


@dataclass(frozen=True)
class H9SourceFreezeExpectation:
    """Production is fixed to ``H9_LIVE_SOURCE_FREEZE``; tests can inject one."""

    revision: str
    provenance_sha256: str
    artifact_hashes: Mapping[str, str]
    metadata_sha256: str
    retained_trials: int
    complete_groups: int
    p_edges: int
    excluded_unmatched: int
    input_metadata_rows: int


H9_LIVE_SOURCE_FREEZE = H9SourceFreezeExpectation(
    revision=H9_ODSS_REVISION,
    provenance_sha256="289675d647ac379e8eabe0e56431cc4fec488d769362abb6c2e667aafadb2f93",
    artifact_hashes={
        "h9_odss_source_trials.csv": "a953a6476dc7b623fc1f27e969fa8401a53da9426ad4873dfd93c1fd33ab588a",
        "h9_odss_source_pairs.csv": "d9df508a5cbce8484742208e92b744e89178baf5e58cce99f42aaa257bb336bb",
        "h9_odss_b2_random_pairs.csv": "cced5cc79a60a5b08e7f7cadb57d6ab9f0e0307ad08506cbe2517fc29a7b1957",
        "h9_odss_excluded_unmatched.csv": "390c2a55eb1ef144a5d6c36982a80ab4b5e27cd83a12fdf57ecb6b8af92638af",
    },
    metadata_sha256="1e4466cce807222807fffba83709b6250e0defa8431a441e2bbc6aec15d46162",
    retained_trials=23_883,
    complete_groups=7_961,
    p_edges=15_922,
    excluded_unmatched=3_071,
    input_metadata_rows=26_954,
)


@dataclass(frozen=True)
class FrozenODSSSource:
    freeze_dir: Path
    provenance_path: Path
    provenance: Mapping[str, Any]
    trials: pd.DataFrame
    pairs: pd.DataFrame
    random_pairs: pd.DataFrame
    excluded_unmatched: pd.DataFrame


@dataclass(frozen=True)
class SourceMaterialization:
    output_dir: Path
    source_manifest: Path
    audio_audit: Path
    provenance: Path
    waveforms_dir: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _path_below(path: Path, root: Path, description: str) -> Path:
    result = path.resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{description} must be below {root.resolve()}, got {result}") from error
    return result


def _read_csv(path: Path, columns: Sequence[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"H9 source freeze lacks {name}: {path}")
    result = pd.read_csv(path, keep_default_na=False)
    if tuple(result.columns) != tuple(columns):
        raise ValueError(f"H9 source freeze {name} schema mismatch")
    return result.loc[:, list(columns)].copy()


def _text(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"H9 source freeze {name} has empty {column}")


def _integers(frame: pd.DataFrame, column: str, name: str) -> None:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"H9 source freeze {name} has non-integer {column}") from error
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"H9 source freeze {name} has non-integer {column}")
    frame[column] = values.astype(int)


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in H9_LIVE_SOURCE_FREEZE.artifact_hashes}


def _validate_trials(trials: pd.DataFrame, expected: H9SourceFreezeExpectation) -> None:
    _text(trials, [name for name in TRIAL_COLUMNS if name not in {"label", "n_counterfactual_partners"}], "trials")
    _integers(trials, "label", "trials")
    _integers(trials, "n_counterfactual_partners", "trials")
    if len(trials) != expected.retained_trials or trials["utterance_id"].duplicated().any() or trials["relative_path"].duplicated().any():
        raise ValueError("H9 source freeze retained trial count or IDs differ from the sealed pool")
    if set(trials["label"]) != {0, 1} or set(trials["split"]) != {"train", "dev"}:
        raise ValueError("H9 source freeze binary-label/split contract failed")
    if set(trials["source_pool"]) != {SOURCE_POOL_NAME} or (trials["n_counterfactual_partners"] < 1).any():
        raise ValueError("H9 source freeze source-pool guard failed")
    if trials["group_key"].nunique() != expected.complete_groups:
        raise ValueError("H9 source freeze complete-group count differs from sealed pool")
    if trials.groupby("voice_key")["split"].nunique().gt(1).any():
        raise ValueError("H9 source freeze voice-disjoint split violation")
    for row in trials.itertuples(index=False):
        parts = str(row.utterance_id).split("__", 3)
        if len(parts) != 4 or any(not value or "/" in value or "\\" in value for value in parts):
            raise ValueError(f"H9 source freeze invalid ODSS ID {row.utterance_id!r}")
        generator, corpus, speaker, stem = parts
        if generator not in {"natural", "vits", "fastpitch-hifigan"} or corpus not in CORPUS_TO_LANGUAGE:
            raise ValueError(f"H9 source freeze unsupported ODSS semantics for {row.utterance_id!r}")
        expected_values = {
            "relative_path": f"{generator}/{corpus}/{speaker}/{stem}.wav",
            "label": 0 if generator == "natural" else 1,
            "label_name": "bonafide" if generator == "natural" else "spoof",
            "generator": generator,
            "source_corpus": corpus,
            "language": CORPUS_TO_LANGUAGE[corpus],
            "speaker": speaker,
            "voice_key": f"{corpus}/{speaker}",
            "content_key": f"{corpus}/{speaker}/{stem}",
        }
        if any(str(getattr(row, key)) != str(value) for key, value in expected_values.items()):
            raise ValueError(f"H9 source freeze trial semantics drifted for {row.utterance_id!r}")
        if row.group_key != f"{row.voice_key}::{row.content_key}":
            raise ValueError(f"H9 source freeze group key drifted for {row.utterance_id!r}")
    for _key, group in trials.groupby("group_key", sort=True):
        if len(group) != 3 or set(group["generator"]) != {"natural", "vits", "fastpitch-hifigan"}:
            raise ValueError("H9 source freeze complete group does not have one natural plus two spoof rows")
        if group[["split", "voice_key", "content_key", "source_corpus", "language"]].nunique().max() != 1:
            raise ValueError("H9 source freeze complete group crosses an invariant")


def _validate_pairs(trials: pd.DataFrame, pairs: pd.DataFrame, random_pairs: pd.DataFrame, expected: H9SourceFreezeExpectation) -> None:
    _text(pairs, PAIR_COLUMNS, "P edges")
    _text(random_pairs, RANDOM_PAIR_COLUMNS, "B2 edges")
    if len(pairs) != expected.p_edges or len(random_pairs) != expected.p_edges:
        raise ValueError("H9 source freeze P/B2 edge count differs from the sealed pool")
    keys = ["pair_id", "spoof_generator"]
    if pairs.duplicated(keys).any() or random_pairs.duplicated(keys).any():
        raise ValueError("H9 source freeze contains ambiguous P/B2 edges")
    pkeys = pairs[keys].sort_values(keys, kind="stable").reset_index(drop=True)
    b2keys = random_pairs[keys].sort_values(keys, kind="stable").reset_index(drop=True)
    all_spoofs = trials.loc[trials.label.eq(1), ["content_key", "generator"]].rename(columns={"content_key": "pair_id", "generator": "spoof_generator"})
    if not pkeys.equals(b2keys) or not pkeys.equals(all_spoofs.sort_values(keys, kind="stable").reset_index(drop=True)):
        raise ValueError("H9 source freeze P/B2 edge set does not exactly cover sealed spoof trials")
    rows = trials.set_index("utterance_id")
    for edge in pairs.itertuples(index=False):
        if edge.bona_utterance_id not in rows.index or edge.spoof_utterance_id not in rows.index:
            raise ValueError("H9 source freeze P edge references a missing trial")
        bona, spoof = rows.loc[edge.bona_utterance_id], rows.loc[edge.spoof_utterance_id]
        if int(bona.label) != 0 or int(spoof.label) != 1:
            raise ValueError("H9 source freeze P edge labels are invalid")
        values = {"pair_id": spoof.content_key, "split": spoof.split, "group_key": spoof.group_key, "voice_key": spoof.voice_key, "content_key": spoof.content_key, "source_corpus": spoof.source_corpus, "language": spoof.language, "spoof_generator": spoof.generator, "bona_relative_path": bona.relative_path, "spoof_relative_path": spoof.relative_path}
        if any(str(getattr(edge, key)) != str(value) for key, value in values.items()):
            raise ValueError("H9 source freeze P edge semantics drifted")
        if any(bona[key] != spoof[key] for key in ("split", "group_key", "voice_key", "content_key", "source_corpus", "language")):
            raise ValueError("H9 source freeze P edge is no longer content-aligned")
    ptable = pairs.set_index(keys)
    for edge in random_pairs.itertuples(index=False):
        matched = ptable.loc[(edge.pair_id, edge.spoof_generator)]
        if edge.spoof_utterance_id != matched.spoof_utterance_id or edge.spoof_relative_path != matched.spoof_relative_path:
            raise ValueError("H9 source freeze B2 edge changed its sealed spoof endpoint")
        if edge.random_bona_utterance_id not in rows.index:
            raise ValueError("H9 source freeze B2 edge references a missing natural trial")
        natural = rows.loc[edge.random_bona_utterance_id]
        if int(natural.label) != 0 or natural.content_key == edge.content_key:
            raise ValueError("H9 source freeze B2 edge has an invalid random natural endpoint")
        values = {"random_bona_relative_path": natural.relative_path, "random_bona_content_key": natural.content_key, "random_bona_voice_key": natural.voice_key, "random_bona_split": natural.split, "random_bona_language": natural.language, "random_bona_source_corpus": natural.source_corpus}
        if any(str(getattr(edge, key)) != str(value) for key, value in values.items()):
            raise ValueError("H9 source freeze B2 edge semantics drifted")


def load_h9_odss_source_freeze(freeze_dir: str | Path, *, expectation: H9SourceFreezeExpectation = H9_LIVE_SOURCE_FREEZE) -> FrozenODSSSource:
    """Validate the compact source freeze; this function never opens an audio shard."""
    root = Path(freeze_dir).resolve()
    provenance_path = root / "h9_odss_source_freeze.json"
    if not root.is_dir() or not provenance_path.is_file() or sha256_file(provenance_path) != expectation.provenance_sha256:
        raise ValueError("H9 ODSS source freeze provenance SHA-256 does not match the sealed freeze")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("H9 ODSS source freeze provenance is invalid JSON") from error
    if not isinstance(provenance, Mapping) or provenance.get("artifact_kind") != "h9_odss_paired_counterfactual_source_freeze" or provenance.get("version") != H9_ODSS_PAIRING_VERSION:
        raise ValueError("H9 ODSS source freeze provenance kind/version mismatch")
    source = provenance.get("source", {})
    metadata = source.get("metadata", {}) if isinstance(source, Mapping) else {}
    access = provenance.get("source_access", {})
    if not isinstance(source, Mapping) or source.get("repo_id") != H9_ODSS_REPO_ID or source.get("revision") != expectation.revision or not isinstance(metadata, Mapping) or metadata.get("sha256") != expectation.metadata_sha256:
        raise ValueError("H9 ODSS source freeze repository/metadata provenance mismatch")
    if not isinstance(access, Mapping) or access.get("target_data_read") is not False or access.get("audio_opened_or_decoded") is not False:
        raise ValueError("H9 ODSS source freeze source/target access boundary mismatch")
    output_names = {"h9_odss_source_trials.csv": "trials_csv", "h9_odss_source_pairs.csv": "pairs_csv", "h9_odss_b2_random_pairs.csv": "b2_random_pairs_csv", "h9_odss_excluded_unmatched.csv": "excluded_unmatched_csv"}
    outputs = provenance.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ValueError("H9 ODSS source freeze lacks output provenance")
    paths = _artifact_paths(root)
    for name, expected_hash in expectation.artifact_hashes.items():
        path, record = paths[name], outputs.get(output_names[name])
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"H9 ODSS source freeze artifact SHA-256 mismatch for {name}")
        if not isinstance(record, Mapping) or Path(str(record.get("path", ""))).resolve() != path.resolve() or record.get("sha256") != expected_hash:
            raise ValueError(f"H9 ODSS source freeze output provenance mismatch for {name}")
    trials = _read_csv(paths["h9_odss_source_trials.csv"], TRIAL_COLUMNS, "trials")
    pairs = _read_csv(paths["h9_odss_source_pairs.csv"], PAIR_COLUMNS, "P edges")
    random_pairs = _read_csv(paths["h9_odss_b2_random_pairs.csv"], RANDOM_PAIR_COLUMNS, "B2 edges")
    excluded = _read_csv(paths["h9_odss_excluded_unmatched.csv"], EXCLUDED_COLUMNS, "excluded rows")
    _validate_trials(trials, expectation)
    _validate_pairs(trials, pairs, random_pairs, expectation)
    _text(excluded, [name for name in EXCLUDED_COLUMNS if name != "label"], "excluded rows")
    _integers(excluded, "label", "excluded rows")
    if len(excluded) != expectation.excluded_unmatched or set(trials.utterance_id).intersection(excluded.utterance_id):
        raise ValueError("H9 source freeze excluded-row ledger mismatch")
    counts = provenance.get("counts", {})
    required_counts = {"input_metadata_rows": expectation.input_metadata_rows, "retained_trials": expectation.retained_trials, "complete_groups": expectation.complete_groups, "retained_pairs": expectation.p_edges, "random_control_pairs": expectation.p_edges, "excluded_unmatched_trials": expectation.excluded_unmatched}
    if not isinstance(counts, Mapping) or any(int(counts.get(key, -1)) != value for key, value in required_counts.items()):
        raise ValueError("H9 source freeze count ledger mismatch")
    guard = provenance.get("same_source_pool_guard", {})
    pkeys = pairs[["pair_id", "spoof_generator"]].sort_values(["pair_id", "spoof_generator"], kind="stable").to_dict(orient="records")
    b2keys = random_pairs[["pair_id", "spoof_generator"]].sort_values(["pair_id", "spoof_generator"], kind="stable").to_dict(orient="records")
    if not isinstance(guard, Mapping) or guard.get("conditions") != ["B1", "B2", "P"] or guard.get("eligible_trial_ids_sha256") != _sha256_json(trials.utterance_id.tolist()) or guard.get("excluded_trial_ids_sha256") != _sha256_json(excluded.utterance_id.tolist()) or guard.get("p_edge_keys_sha256") != _sha256_json(pkeys) or guard.get("b2_edge_keys_sha256") != _sha256_json(b2keys):
        raise ValueError("H9 source freeze source-pool or pair-edge hash mismatch")
    if provenance.get("trials_rows_sha256") != _sha256_json(trials.to_dict(orient="records")) or provenance.get("pairs_rows_sha256") != _sha256_json(pairs.to_dict(orient="records")) or provenance.get("random_pairs_rows_sha256") != _sha256_json(random_pairs.to_dict(orient="records")):
        raise ValueError("H9 source freeze row hash mismatch")
    return FrozenODSSSource(root, provenance_path, provenance, trials, pairs, random_pairs, excluded)


def canonical_mono16k_pcm_fingerprint(raw_wav: bytes) -> tuple[str, int, int]:
    """Fingerprint deterministic mono 16-kHz signed-16-bit PCM, not raw bytes."""
    if len(raw_wav) < 12 or raw_wav[:4] != b"RIFF" or raw_wav[8:12] != b"WAVE":
        raise ValueError("H9 ODSS source payload is not a RIFF/WAVE byte stream")
    try:
        decoded, sample_rate = sf.read(io.BytesIO(raw_wav), dtype="float32", always_2d=True)
    except RuntimeError as error:
        raise ValueError("H9 ODSS source payload cannot be decoded as WAV") from error
    values = np.asarray(decoded, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all() or int(sample_rate) <= 0:
        raise ValueError("H9 ODSS source WAV is empty, malformed, or non-finite")
    mono = values.astype(np.float64).mean(axis=1, dtype=np.float64).astype(np.float32)
    if int(sample_rate) != 16_000:
        from scipy.signal import resample_poly
        divisor = math.gcd(int(sample_rate), 16_000)
        mono = resample_poly(mono, 16_000 // divisor, int(sample_rate) // divisor, window=("kaiser", 5.0), padtype="constant").astype(np.float32, copy=False)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError("H9 canonical PCM is empty or non-finite")
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes(order="C")
    prefix = f"{CANONICAL_AUDIO_POLICY['kind']}|v{CANONICAL_AUDIO_POLICY['version']}|sr=16000|frames=".encode("ascii")
    return hashlib.sha256(prefix + struct.pack("<Q", mono.size) + pcm).hexdigest(), int(mono.size), len(pcm)


def _raw_shards(raw_shard_dir: Path) -> list[Path]:
    if not raw_shard_dir.is_dir() or raw_shard_dir.name != "data" or H9_ODSS_REVISION not in raw_shard_dir.parts:
        raise ValueError("H9 ODSS raw-shard input must be the pinned revision's data directory")
    result = sorted(path for path in raw_shard_dir.glob("*.parquet") if path.is_file())
    if not result:
        raise FileNotFoundError(f"H9 ODSS raw-shard directory has no Parquet shards: {raw_shard_dir}")
    return result


def _audio_bytes(value: object, path: str) -> bytes:
    payload = value.get("bytes") if isinstance(value, Mapping) else value
    if payload is None:
        raise ValueError(f"H9 ODSS raw-shard audio bytes unavailable for {path}")
    result = bytes(payload)
    if not result:
        raise ValueError(f"H9 ODSS raw-shard audio bytes empty for {path}")
    return result


def materialize_h9_odss_source(*, freeze_dir: str | Path, raw_shard_dir: str | Path, output_dir: str | Path, allowed_output_root: str | Path = HDD_ROOT, _expectation: H9SourceFreezeExpectation = H9_LIVE_SOURCE_FREEZE) -> SourceMaterialization:
    """Extract exactly the sealed source trials to one new immutable HDD directory.

    The private expectation injection exists solely for small synthetic tests;
    the production CLI has no way to change the real 23,883-trial freeze.
    """
    frozen = load_h9_odss_source_freeze(freeze_dir, expectation=_expectation)
    raw_dir = Path(raw_shard_dir).resolve()
    shards = _raw_shards(raw_dir)
    target = _path_below(Path(output_dir), Path(allowed_output_root), "H9 ODSS materialization output")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing H9 ODSS materialization directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        waveforms = stage / "waveforms"
        waveforms.mkdir()
        by_path = frozen.trials.set_index("relative_path")
        extracted: dict[str, dict[str, Any]] = {}
        shard_records: list[dict[str, Any]] = []
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            schema = parquet.schema_arrow
            if not {"path", "audio", "label"}.issubset(schema.names):
                raise ValueError(f"H9 ODSS raw-shard schema lacks path/audio/label: {shard}")
            if "bytes" not in str(schema.field("audio").type) and not str(schema.field("audio").type).startswith(("binary", "large_binary")):
                raise ValueError(f"H9 ODSS raw-shard audio schema lacks bytes: {shard}")
            # ``str(schema)`` is available across the pinned PyArrow variants
            # in this environment; newer-only ``show_metadata`` flags are not.
            schema_text = str(schema)
            shard_records.append({"path": str(shard.resolve()), "sha256": sha256_file(shard), "byte_size": int(shard.stat().st_size), "schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest(), "schema": schema_text})
            for batch in parquet.iter_batches(batch_size=256, columns=["path", "audio", "label"]):
                for row in batch.to_pylist():
                    relative_path = str(row["path"]).replace("\\", "/").strip()
                    if relative_path not in by_path.index:
                        continue
                    source_row = by_path.loc[relative_path]
                    sample_id = str(source_row.utterance_id)
                    if sample_id in extracted:
                        raise ValueError(f"H9 ODSS raw shards duplicate frozen sample {sample_id}")
                    if int(row["label"]) != int(source_row.label):
                        raise ValueError(f"H9 ODSS raw-shard label disagrees with sealed source freeze for {sample_id}")
                    payload = _audio_bytes(row["audio"], relative_path)
                    fingerprint, frames, pcm_bytes = canonical_mono16k_pcm_fingerprint(payload)
                    output_wav = waveforms / f"{sample_id}.wav"
                    output_wav.write_bytes(payload)
                    raw_hash = hashlib.sha256(payload).hexdigest()
                    extracted[sample_id] = {"sample_id": sample_id, "relative_path": relative_path, "raw_shard_path": str(shard.resolve()), "raw_payload_sha256": raw_hash, "raw_payload_bytes": len(payload), "extracted_wav_sha256": sha256_file(output_wav), "extracted_wav_bytes": int(output_wav.stat().st_size), "canonical_fingerprint": fingerprint, "canonical_pcm_frames": frames, "canonical_pcm_bytes": pcm_bytes}
        expected_ids = set(frozen.trials.utterance_id)
        if set(extracted) != expected_ids or len(extracted) != _expectation.retained_trials:
            missing = sorted(expected_ids - set(extracted))
            raise ValueError(f"H9 ODSS raw shards do not contain exactly the sealed source pool; missing={len(missing)} {missing[:5]}")
        manifest_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for row in frozen.trials.itertuples(index=False):
            audit = extracted[str(row.utterance_id)]
            manifest_rows.append({"sample_id": str(row.utterance_id), "audio_path": str((target / "waveforms" / f"{row.utterance_id}.wav").resolve()), "label": int(row.label), "split": str(row.split), "pair_id": str(row.content_key), "group_id": f"{row.source_corpus}|{row.speaker}", "language": str(row.language), "canonical_fingerprint": audit["canonical_fingerprint"], "source_corpus": str(row.source_corpus), "speaker_id": str(row.speaker), "spoof_generator": str(row.generator)})
            audit_rows.append(audit)
        manifest = pd.DataFrame(manifest_rows, columns=list(REQUIRED_SOURCE_COLUMNS))
        if tuple(manifest.columns) != REQUIRED_SOURCE_COLUMNS or not manifest.group_id.eq(manifest.source_corpus + "|" + manifest.speaker_id).all():
            raise RuntimeError("H9 materializer did not emit canonical trainer fields/group IDs")
        manifest_path, audit_path = stage / "h9_odss_source_manifest.csv", stage / "h9_odss_source_audio_audit.csv"
        manifest.to_csv(manifest_path, index=False)
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        trainer = load_source_manifest(manifest_path)
        if trainer.source_manifest_sha256 != sha256_file(manifest_path) or sum(map(len, trainer.paired_eligible_ids.values())) != _expectation.retained_trials:
            raise RuntimeError("H9 emitted source manifest fails the trainer's sealed-pool validator")
        provenance = {
            "artifact_kind": "h9_odss_paired_counterfactual_source_materialization", "version": H9_ODSS_MATERIALIZATION_VERSION,
            "claim_guard": "Source-only ODSS WAV extraction. No target data, model, score, training result, or checkpoint was read.",
            "source_access": {"sealed_source_freeze_read": True, "raw_odss_audio_opened": True, "source_feature_read": False, "source_detector_score_read": False, "model_loaded": False, "training_run": False, "target_data_read": False, "target_label_read": False, "target_prediction_read": False},
            "source": {"repo_id": H9_ODSS_REPO_ID, "revision": H9_ODSS_REVISION, "raw_shard_dir": str(raw_dir), "raw_shards": shard_records},
            "sealed_source_freeze": {"directory": str(frozen.freeze_dir), "provenance_sha256": sha256_file(frozen.provenance_path), "artifact_sha256": {name: sha256_file(path) for name, path in _artifact_paths(frozen.freeze_dir).items()}, "retained_trials": _expectation.retained_trials, "complete_groups": _expectation.complete_groups, "p_edges": _expectation.p_edges, "b2_edges": _expectation.p_edges, "excluded_unmatched": _expectation.excluded_unmatched},
            "canonical_audio_policy": CANONICAL_AUDIO_POLICY,
            "outputs": {"source_manifest_csv": {"path": str((target / manifest_path.name).resolve()), "sha256": sha256_file(manifest_path), "n_rows": len(manifest), "columns": list(REQUIRED_SOURCE_COLUMNS)}, "audio_audit_csv": {"path": str((target / audit_path.name).resolve()), "sha256": sha256_file(audit_path), "n_rows": len(audit_rows)}, "waveforms": {"directory": str((target / "waveforms").resolve()), "n_files": len(extracted), "payload_id_sha256": _sha256_json([{"sample_id": row["sample_id"], "raw_payload_sha256": row["raw_payload_sha256"]} for row in audit_rows])}},
            "trainer_manifest_validation": {"source_manifest_sha256": trainer.source_manifest_sha256, "required_columns": list(REQUIRED_SOURCE_COLUMNS), "paired_eligible_rows": sum(map(len, trainer.paired_eligible_ids.values())), "p_edges": sum(map(len, trainer.pair_edges.values()))},
        }
        provenance_path = stage / "h9_odss_source_materialization.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing H9 ODSS materialization directory: {target}")
        os.replace(stage, target)
        return SourceMaterialization(target, target / manifest_path.name, target / audit_path.name, target / provenance_path.name, target / "waveforms")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
