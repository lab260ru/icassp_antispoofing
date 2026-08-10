"""Freeze source-only ODSS content-matched pairs for H9-PCR.

This module is intentionally a *metadata-only* step.  It reads a pinned ODSS
label table (``utterance_id``, ``label``) and the two versioned ODSS documents
that define that identifier's path semantics.  It never opens an audio shard,
decodes a waveform, loads an H9 target, constructs a model, or computes a
detector score.

The ODSS packaging derives its identifier from the source-relative path:
``generator/corpus/speaker/stem.wav`` becomes
``generator__corpus__speaker__stem``.  A conservative H9 content group is the
triple ``(corpus, speaker, stem)``.  The group is retained only when it has
exactly one natural rendering and at least one distinct synthetic rendering.
This excludes the ODSS VCTK-only synthetic rows rather than inventing a
counterpart for them. Every retained content group is assigned with its whole
voice to train or development, preventing ODSS speaker/voice leakage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd


H9_ODSS_PAIRING_VERSION = "h9_odss_paired_counterfactual_source_freeze_v1"
H9_ODSS_REPO_ID = "SpeechAntiSpoofingBenchmarks/ODSS"
H9_ODSS_REVISION = "1968e6d0ef141c4572073695bdc1d17a8706177f"
H9_SPLIT_SEED = 2909
H9_DEV_FRACTION = 0.20
MIN_COMPLETE_GROUPS = 1000
MIN_COMPLETE_GROUPS_PER_SPLIT = 1000
HDD_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing")

GENERATOR_TO_LABEL = {"natural": 0, "vits": 1, "fastpitch-hifigan": 1}
CORPUS_TO_LANGUAGE = {
    "hifi-tts": "en",
    "hui-acg": "de",
    "openslr-es": "es",
    "vctk": "en",
}
REQUIRED_METADATA_COLUMNS = ("utterance_id", "label")
SEMANTICS_MARKERS = {
    "readme": (
        "each natural utterance is paired with TTS re-synthesis of the same text",
        "the bare stem repeats across the three generators",
        "source-relative path",
    ),
    "build_script": (
        "utterance_id = full source-relative path with '/' -> '__'",
        'parts = rel.split("/")',
        'label = "bonafide" if gen == "natural" else "spoof"',
        "natural/<corpus>/<speaker>/<stem>.wav",
        "vits/<corpus>/<speaker>/<stem>.wav",
        "fastpitch-hifigan/<corpus>/<speaker>/<stem>.wav",
    ),
}


@dataclass(frozen=True)
class H9ODSSPairingFreeze:
    """In-memory, target-free pairing and split freeze before artifact writing."""

    trials: pd.DataFrame
    pairs: pd.DataFrame
    random_pairs: pd.DataFrame
    excluded_unmatched: pd.DataFrame
    provenance: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest without parsing a second table."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _path_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required H9 ODSS source file is missing: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "byte_size": int(source.stat().st_size),
    }


def _normalize_semantics_whitespace(value: str) -> str:
    """Make Markdown line wrapping semantically inert without relaxing wording."""
    return re.sub(r"\s+", " ", value).strip()


def _validate_semantics_documents(semantics_documents: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Byte-bind the two ODSS documents that make UID pairing interpretable."""
    expected = set(SEMANTICS_MARKERS)
    if set(semantics_documents) != expected:
        raise ValueError(f"ODSS semantics documents must be exactly {sorted(expected)}")
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        record = _path_record(semantics_documents[name])
        content = _normalize_semantics_whitespace(Path(record["path"]).read_text(encoding="utf-8"))
        missing = [
            marker
            for marker in SEMANTICS_MARKERS[name]
            if _normalize_semantics_whitespace(marker) not in content
        ]
        if missing:
            raise ValueError(
                f"ODSS {name} no longer establishes the pinned path/label semantics; missing markers: {missing}"
            )
        records[name] = record
    return records


def _read_metadata_table(path: str | Path) -> pd.DataFrame:
    """Read only the compact ODSS label metadata table, never an audio shard."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ODSS label metadata is missing: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        rows = pd.read_parquet(source)
    elif suffix == ".csv":
        rows = pd.read_csv(source)
    else:
        raise ValueError(f"ODSS metadata must be .parquet or .csv, got {source}")
    forbidden = [column for column in rows.columns if str(column).casefold() in {"audio", "waveform", "bytes"}]
    if forbidden:
        raise ValueError(f"H9 pairing accepts metadata only, not audio-bearing columns: {forbidden}")
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"ODSS metadata is missing required columns: {missing}")
    return rows


def _nonempty_text(value: object, *, field: str, utterance_id: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: {utterance_id} has empty {field}")
    result = str(value).strip()
    if not result:
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: {utterance_id} has empty {field}")
    return result


def _parse_utterance_id(utterance_id: str, label: object, row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the documented ODSS UID, and reject any conflicting optional path/notes."""
    parts = utterance_id.split("__", 3)
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise ValueError(
            "SOURCE_PAIRING_UNAVAILABLE: ODSS utterance_id must encode "
            "generator__corpus__speaker__stem exactly, got "
            f"{utterance_id!r}"
        )
    generator, source_corpus, speaker, stem = (part.strip() for part in parts)
    if generator not in GENERATOR_TO_LABEL:
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: unsupported ODSS generator {generator!r}")
    if any("/" in field or "\\" in field for field in (generator, source_corpus, speaker, stem)):
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: path separator found inside ODSS utterance_id {utterance_id!r}")
    if source_corpus not in CORPUS_TO_LANGUAGE:
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: unsupported ODSS source corpus {source_corpus!r}")
    try:
        numeric_label = int(label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: non-binary label for {utterance_id!r}: {label!r}") from error
    expected_label = GENERATOR_TO_LABEL[generator]
    if numeric_label != expected_label:
        raise ValueError(
            "SOURCE_PAIRING_UNAVAILABLE: generator/label conflict for "
            f"{utterance_id!r}: generator {generator!r} requires {expected_label}, observed {numeric_label}"
        )
    relative_path = f"{generator}/{source_corpus}/{speaker}/{stem}.wav"
    if "path" in row and not pd.isna(row["path"]):
        observed_path = _nonempty_text(row["path"], field="path", utterance_id=utterance_id).replace("\\", "/")
        if observed_path != relative_path:
            raise ValueError(
                "SOURCE_PAIRING_UNAVAILABLE: optional ODSS path conflicts with documented UID semantics for "
                f"{utterance_id!r}: expected {relative_path!r}, observed {observed_path!r}"
            )
    if "notes" in row and not pd.isna(row["notes"]):
        notes_raw = _nonempty_text(row["notes"], field="notes", utterance_id=utterance_id)
        try:
            notes = json.loads(notes_raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: invalid JSON notes for {utterance_id!r}") from error
        if not isinstance(notes, Mapping):
            raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: JSON notes must be an object for {utterance_id!r}")
        expected_notes = {
            "utterance_id": utterance_id,
            "generator": generator,
            "source_corpus": source_corpus,
            "speaker": speaker,
            "language": CORPUS_TO_LANGUAGE[source_corpus],
        }
        mismatches = {
            field: {"expected": expected, "observed": notes.get(field)}
            for field, expected in expected_notes.items()
            if notes.get(field) != expected
        }
        if mismatches:
            raise ValueError(
                "SOURCE_PAIRING_UNAVAILABLE: optional ODSS notes conflict with UID semantics for "
                f"{utterance_id!r}: {mismatches}"
            )
    content_key = f"{source_corpus}/{speaker}/{stem}"
    voice_key = f"{source_corpus}/{speaker}"
    return {
        "utterance_id": utterance_id,
        "relative_path": relative_path,
        "label": numeric_label,
        "label_name": "bonafide" if numeric_label == 0 else "spoof",
        "generator": generator,
        "source_corpus": source_corpus,
        "language": CORPUS_TO_LANGUAGE[source_corpus],
        "speaker": speaker,
        "voice_key": voice_key,
        "content_key": content_key,
        # This explicit tuple spelling makes the plan's (voice, content) rule
        # inspectable even though content_key already includes the voice key.
        "group_key": f"{voice_key}::{content_key}",
    }


def _parse_metadata_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("SOURCE_PAIRING_UNAVAILABLE: ODSS metadata has no rows")
    parsed: list[dict[str, Any]] = []
    for record in rows.to_dict(orient="records"):
        utterance_id = _nonempty_text(record.get("utterance_id"), field="utterance_id", utterance_id="<unknown>")
        parsed.append(_parse_utterance_id(utterance_id, record.get("label"), record))
    result = pd.DataFrame(parsed)
    if result["utterance_id"].duplicated().any():
        duplicates = result.loc[result["utterance_id"].duplicated(keep=False), "utterance_id"].head(8).tolist()
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: duplicate ODSS utterance_id values: {duplicates}")
    duplicate_generator = result.duplicated(["content_key", "generator"], keep=False)
    if duplicate_generator.any():
        examples = result.loc[duplicate_generator, ["content_key", "generator", "utterance_id"]].head(8).to_dict("records")
        raise ValueError(
            "SOURCE_PAIRING_UNAVAILABLE: a content group has ambiguous duplicate generator renderings: "
            f"{examples}"
        )
    return result.sort_values(["content_key", "generator", "utterance_id"], kind="stable").reset_index(drop=True)


def _complete_content_groups(parsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate complete groups from an explicit unmatched-row exclusion ledger."""
    retained: list[pd.DataFrame] = []
    excluded: list[pd.DataFrame] = []
    for content_key, group in parsed.groupby("content_key", sort=True, dropna=False):
        bona = group.loc[group["label"] == 0]
        spoof = group.loc[group["label"] == 1]
        if bona.empty or spoof.empty:
            excluded.append(group.assign(exclusion_reason="missing_bonafide" if bona.empty else "missing_spoof").copy())
            continue
        if len(bona) != 1:
            raise ValueError(
                "SOURCE_PAIRING_UNAVAILABLE: complete ODSS content group must have exactly one natural rendering, "
                f"got {len(bona)} for {content_key!r}"
            )
        retained.append(group.copy())
    if not retained:
        raise ValueError("SOURCE_PAIRING_UNAVAILABLE: no ODSS content group has both natural and spoof recordings")
    result = pd.concat(retained, ignore_index=True)
    complete_groups = int(result["group_key"].nunique())
    if complete_groups < MIN_COMPLETE_GROUPS:
        raise ValueError(
            "SOURCE_PAIRING_UNAVAILABLE: fewer than the required "
            f"{MIN_COMPLETE_GROUPS} complete conservative ODSS groups ({complete_groups})"
        )
    excluded_rows = (
        pd.concat(excluded, ignore_index=True)
        if excluded
        else pd.DataFrame(columns=[*parsed.columns, "exclusion_reason"])
    )
    excluded_rows = excluded_rows.loc[
        :,
        [
            "utterance_id",
            "relative_path",
            "label",
            "label_name",
            "generator",
            "source_corpus",
            "language",
            "speaker",
            "voice_key",
            "content_key",
            "group_key",
            "exclusion_reason",
        ],
    ].sort_values(["exclusion_reason", "content_key", "generator", "utterance_id"], kind="stable").reset_index(drop=True)
    return result, excluded_rows


def _split_voices(complete: pd.DataFrame, *, split_seed: int, dev_fraction: float) -> pd.DataFrame:
    if split_seed != H9_SPLIT_SEED:
        raise ValueError(f"H9-PCR split seed is locked at {H9_SPLIT_SEED}, got {split_seed}")
    if dev_fraction != H9_DEV_FRACTION:
        raise ValueError(f"H9-PCR dev fraction is locked at {H9_DEV_FRACTION}, got {dev_fraction}")
    voice_records = (
        complete.loc[:, ["voice_key", "source_corpus", "language"]]
        .drop_duplicates()
        .sort_values("voice_key", kind="stable")
        .reset_index(drop=True)
    )
    if voice_records["voice_key"].duplicated().any():
        raise RuntimeError("H9 ODSS source metadata maps a voice to conflicting corpus/language records")
    assignments: dict[str, str] = {}
    for language, stratum in voice_records.groupby("language", sort=True):
        n_dev = int(math.ceil(len(stratum) * dev_fraction))
        if n_dev == 0 or n_dev == len(stratum):
            raise ValueError(
                "SOURCE_PAIRING_UNAVAILABLE: language stratum cannot populate both voice-disjoint splits: "
                f"{language!r} has {len(stratum)} voices"
            )
        ranked = stratum.assign(
            _rank=stratum["voice_key"].map(
                lambda key: hashlib.sha256(f"{split_seed}:{key}".encode("utf-8")).hexdigest()
            )
        ).sort_values(["_rank", "voice_key"], kind="stable")
        assignments.update({key: "dev" for key in ranked.iloc[:n_dev]["voice_key"]})
        assignments.update({key: "train" for key in ranked.iloc[n_dev:]["voice_key"]})
    if len(assignments) != len(voice_records):
        raise RuntimeError("H9 ODSS split assignment did not cover every complete voice")
    result = complete.copy()
    result["split"] = result["voice_key"].map(assignments)
    if result["split"].isna().any():
        raise RuntimeError("H9 ODSS split assignment produced unassigned trials")
    per_split = result.loc[:, ["group_key", "split"]].drop_duplicates().groupby("split", sort=True).size().to_dict()
    missing = [split for split in ("train", "dev") if split not in per_split]
    if missing:
        raise ValueError(f"SOURCE_PAIRING_UNAVAILABLE: missing voice-disjoint split(s): {missing}")
    too_small = {split: int(count) for split, count in per_split.items() if count < MIN_COMPLETE_GROUPS_PER_SPLIT}
    if too_small:
        raise ValueError(
            "SOURCE_PAIRING_UNAVAILABLE: each split must retain at least "
            f"{MIN_COMPLETE_GROUPS_PER_SPLIT} complete groups, observed {too_small}"
        )
    crossing = result.groupby("voice_key", sort=True)["split"].nunique()
    if (crossing != 1).any():
        bad = crossing.loc[crossing != 1].head(8).to_dict()
        raise RuntimeError(f"H9 voice-disjoint split violation: {bad}")
    language_splits = result.loc[:, ["language", "split"]].drop_duplicates().groupby("language", sort=True)["split"].agg(set)
    missing_languages = {language: sorted({"train", "dev"} - splits) for language, splits in language_splits.items() if splits != {"train", "dev"}}
    if missing_languages:
        raise ValueError(
            "SOURCE_PAIRING_UNAVAILABLE: every ODSS language must occur in both voice-disjoint splits, "
            f"missing {missing_languages}"
        )
    return result.sort_values(["split", "content_key", "generator", "utterance_id"], kind="stable").reset_index(drop=True)


def _build_pairs(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in trials.groupby("group_key", sort=True):
        bona = group.loc[group["label"] == 0]
        spoof = group.loc[group["label"] == 1].sort_values(["generator", "utterance_id"], kind="stable")
        if len(bona) != 1 or spoof.empty:
            raise RuntimeError(f"H9 complete group invariant broken for {group_key!r}")
        natural = bona.iloc[0]
        for _, synthetic in spoof.iterrows():
            rows.append(
                {
                    "pair_id": natural["content_key"],
                    "split": natural["split"],
                    "group_key": group_key,
                    "voice_key": natural["voice_key"],
                    "content_key": natural["content_key"],
                    "source_corpus": natural["source_corpus"],
                    "language": natural["language"],
                    "bona_utterance_id": natural["utterance_id"],
                    "bona_relative_path": natural["relative_path"],
                    "spoof_utterance_id": synthetic["utterance_id"],
                    "spoof_relative_path": synthetic["relative_path"],
                    "spoof_generator": synthetic["generator"],
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(["pair_id", "spoof_generator"], keep=False).any():
        raise RuntimeError("H9 pair construction yielded no pairs or ambiguous pair_id/spoof_generator edges")
    if (result["bona_utterance_id"] == result["spoof_utterance_id"]).any():
        raise RuntimeError("H9 pair construction paired a trial with itself")
    if result.groupby("voice_key", sort=True)["split"].nunique().gt(1).any():
        raise RuntimeError("H9 pair construction broke voice-disjoint split assignment")
    return result.sort_values(["split", "content_key", "spoof_generator", "pair_id"], kind="stable").reset_index(drop=True)


def _build_random_pairs(pairs: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Create B2 edges from the same source pool without a matched-content partner."""
    natural = trials.loc[trials["label"] == 0].copy()
    random_rows: list[dict[str, Any]] = []
    strata = ["split", "language", "source_corpus", "spoof_generator"]
    for values, p_stratum in pairs.groupby(strata, sort=True):
        split, language, source_corpus, spoof_generator = values
        candidates = natural.loc[
            (natural["split"] == split)
            & (natural["language"] == language)
            & (natural["source_corpus"] == source_corpus)
        ].sort_values("utterance_id", kind="stable")
        if len(candidates) < 2:
            raise ValueError(
                "SOURCE_PAIRING_UNAVAILABLE: B2 random-pair stratum has fewer than two bona-fide candidates: "
                f"{values}"
            )
        ranked_pairs = p_stratum.assign(
            _rank=p_stratum["pair_id"].map(
                lambda pair_id: hashlib.sha256(
                    f"{H9_SPLIT_SEED}:b2:{split}:{language}:{source_corpus}:{spoof_generator}:{pair_id}".encode("utf-8")
                ).hexdigest()
            )
        ).sort_values(["_rank", "pair_id"], kind="stable")
        ranked_candidates = candidates.assign(
            _rank=candidates["utterance_id"].map(
                lambda uid: hashlib.sha256(
                    f"{H9_SPLIT_SEED}:b2-candidate:{split}:{language}:{source_corpus}:{spoof_generator}:{uid}".encode("utf-8")
                ).hexdigest()
            )
        ).sort_values(["_rank", "utterance_id"], kind="stable").reset_index(drop=True)
        for position, (_, edge) in enumerate(ranked_pairs.iterrows()):
            chosen: pd.Series | None = None
            for offset in range(len(ranked_candidates)):
                candidate = ranked_candidates.iloc[(position + 1 + offset) % len(ranked_candidates)]
                if candidate["content_key"] != edge["content_key"]:
                    chosen = candidate
                    break
            if chosen is None:
                raise ValueError(
                    "SOURCE_PAIRING_UNAVAILABLE: B2 random-pair stratum has no content-unmatched bona-fide partner: "
                    f"{values}"
                )
            random_rows.append(
                {
                    "pair_id": edge["pair_id"],
                    "split": split,
                    "group_key": edge["group_key"],
                    "voice_key": edge["voice_key"],
                    "content_key": edge["content_key"],
                    "source_corpus": source_corpus,
                    "language": language,
                    "spoof_generator": spoof_generator,
                    "random_bona_utterance_id": chosen["utterance_id"],
                    "random_bona_relative_path": chosen["relative_path"],
                    "random_bona_content_key": chosen["content_key"],
                    "random_bona_voice_key": chosen["voice_key"],
                    "random_bona_split": chosen["split"],
                    "random_bona_language": chosen["language"],
                    "random_bona_source_corpus": chosen["source_corpus"],
                    "spoof_utterance_id": edge["spoof_utterance_id"],
                    "spoof_relative_path": edge["spoof_relative_path"],
                }
            )
    result = pd.DataFrame(random_rows).sort_values(["split", "content_key", "spoof_generator", "pair_id"], kind="stable").reset_index(drop=True)
    key_columns = ["pair_id", "spoof_generator"]
    if len(result) != len(pairs) or result.duplicated(key_columns, keep=False).any():
        raise RuntimeError("H9 B2 mapping does not contain one unambiguous edge for every P edge")
    expected = pairs.loc[:, key_columns].sort_values(key_columns, kind="stable").reset_index(drop=True)
    observed = result.loc[:, key_columns].sort_values(key_columns, kind="stable").reset_index(drop=True)
    if not expected.equals(observed):
        raise RuntimeError("H9 B2 mapping does not share the exact P pair_id/spoof_generator edge set")
    if result["content_key"].eq(result["random_bona_content_key"]).any():
        raise RuntimeError("H9 B2 mapping accidentally retained a content-matched bona-fide partner")
    if not result["split"].eq(result["random_bona_split"]).all():
        raise RuntimeError("H9 B2 mapping crossed the frozen voice-disjoint split")
    if not result["language"].eq(result["random_bona_language"]).all() or not result["source_corpus"].eq(
        result["random_bona_source_corpus"]
    ).all():
        raise RuntimeError("H9 B2 mapping left its language/source-corpus stratum")
    return result


def _with_partner_counts(trials: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    pair_counts = pd.concat(
        [
            pairs.loc[:, ["bona_utterance_id"]].rename(columns={"bona_utterance_id": "utterance_id"}),
            pairs.loc[:, ["spoof_utterance_id"]].rename(columns={"spoof_utterance_id": "utterance_id"}),
        ],
        ignore_index=True,
    ).groupby("utterance_id", sort=True).size()
    result = trials.copy()
    result["n_counterfactual_partners"] = result["utterance_id"].map(pair_counts).fillna(0).astype(int)
    result["source_pool"] = "shared_b1_b2_p_complete_matched_only"
    if (result["n_counterfactual_partners"] < 1).any():
        raise RuntimeError("H9 retained trial without a counterfactual partner")
    columns = [
        "utterance_id",
        "relative_path",
        "label",
        "label_name",
        "generator",
        "source_corpus",
        "language",
        "speaker",
        "voice_key",
        "content_key",
        "group_key",
        "split",
        "n_counterfactual_partners",
        "source_pool",
    ]
    return result.loc[:, columns].sort_values(["split", "content_key", "generator", "utterance_id"], kind="stable").reset_index(drop=True)


def _counts_by(rows: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    grouped = rows.groupby(list(columns), sort=True).size()
    return {
        "|".join(str(item) for item in (key if isinstance(key, tuple) else (key,)) if item is not None): int(value)
        for key, value in grouped.items()
    }


def build_h9_odss_pairing_freeze(
    metadata_rows: pd.DataFrame,
    *,
    metadata_record: Mapping[str, Any],
    semantics_records: Mapping[str, Mapping[str, Any]],
    protocol_records: Mapping[str, Mapping[str, Any]],
    split_seed: int = H9_SPLIT_SEED,
    dev_fraction: float = H9_DEV_FRACTION,
) -> H9ODSSPairingFreeze:
    """Create deterministic source-only trials/pairs from compact ODSS metadata."""
    parsed = _parse_metadata_rows(metadata_rows)
    complete, excluded_unmatched = _complete_content_groups(parsed)
    assigned = _split_voices(complete, split_seed=split_seed, dev_fraction=dev_fraction)
    pairs = _build_pairs(assigned)
    trials = _with_partner_counts(assigned, pairs)
    random_pairs = _build_random_pairs(pairs, trials)
    group_splits = trials.loc[:, ["group_key", "split"]].drop_duplicates()
    if group_splits["group_key"].duplicated().any():
        raise RuntimeError("H9 group-disjoint manifest has a group in both train and dev")
    provenance: dict[str, Any] = {
        "artifact_kind": "h9_odss_paired_counterfactual_source_freeze",
        "version": H9_ODSS_PAIRING_VERSION,
        "claim_guard": (
            "ODSS source metadata pairing/split freeze only. No audio was opened or decoded; no source feature, "
            "detector score, target file, target label, target prediction, model, or training result was read. "
            "This preliminary metadata freeze does not replace the later source-audio and source--target fingerprint audit."
        ),
        "source_access": {
            "odss_metadata_read": True,
            "audio_opened_or_decoded": False,
            "source_feature_read": False,
            "source_detector_score_read": False,
            "model_loaded": False,
            "training_run": False,
            "target_data_read": False,
            "target_label_read": False,
            "target_prediction_read": False,
        },
        "source": {
            "repo_id": H9_ODSS_REPO_ID,
            "revision": H9_ODSS_REVISION,
            "metadata": dict(metadata_record),
            "semantics_documents": {name: dict(record) for name, record in sorted(semantics_records.items())},
        },
        "protocol_documents": {name: dict(record) for name, record in sorted(protocol_records.items())},
        "pairing_rule": {
            "utterance_id_encoding": "generator__source_corpus__speaker__stem",
            "relative_path_reconstruction": "generator/source_corpus/speaker/stem.wav",
            "content_key": "source_corpus/speaker/stem",
            "voice_key": "source_corpus/speaker",
            "group_key": "voice_key::content_key",
            "pair_id": "source_corpus/speaker/stem",
            "retain_rule": "exactly one bonafide and at least one spoof rendering per content group",
            "incomplete_groups": "excluded; no pairing is inferred from audio similarity",
        },
        "same_source_pool_guard": {
            "conditions": ["B1", "B2", "P"],
            "eligible_trials": "only trials from complete natural+spoof content groups",
            "excluded_trials": "every unmatched ODSS row is recorded in excluded_unmatched and is unavailable to B1, B2, and P",
            "eligible_trial_ids_sha256": _sha256_json(trials["utterance_id"].tolist()),
            "excluded_trial_ids_sha256": _sha256_json(excluded_unmatched["utterance_id"].tolist()),
            "excluded_rows_sha256": _sha256_json(excluded_unmatched.to_dict(orient="records")),
            "p_edge_keys_sha256": _sha256_json(
                pairs.loc[:, ["pair_id", "spoof_generator"]]
                .sort_values(["pair_id", "spoof_generator"], kind="stable")
                .to_dict(orient="records")
            ),
            "b2_edge_keys_sha256": _sha256_json(
                random_pairs.loc[:, ["pair_id", "spoof_generator"]]
                .sort_values(["pair_id", "spoof_generator"], kind="stable")
                .to_dict(orient="records")
            ),
        },
        "split_rule": {
            "split_seed": split_seed,
            "dev_fraction": dev_fraction,
            "assignment": "SHA-256(seed:voice_key), stratified by documented language",
            "voice_disjoint": True,
            "group_disjoint": True,
            "minimum_complete_groups_total": MIN_COMPLETE_GROUPS,
            "minimum_complete_groups_per_split": MIN_COMPLETE_GROUPS_PER_SPLIT,
        },
        "counts": {
            "input_metadata_rows": int(len(metadata_rows)),
            "parsed_metadata_rows": int(len(parsed)),
            "retained_trials": int(len(trials)),
            "retained_pairs": int(len(pairs)),
            "random_control_pairs": int(len(random_pairs)),
            "excluded_unmatched_trials": int(len(excluded_unmatched)),
            "excluded_unmatched_by_reason": _counts_by(excluded_unmatched, ("exclusion_reason",)) if not excluded_unmatched.empty else {},
            "complete_groups": int(len(group_splits)),
            "groups_by_split": {str(key): int(value) for key, value in group_splits.groupby("split", sort=True).size().items()},
            "trials_by_split_and_label": _counts_by(trials, ("split", "label_name")),
            "pairs_by_split_and_generator": _counts_by(pairs, ("split", "spoof_generator")),
            "groups_by_corpus_and_split": _counts_by(group_splits.merge(trials.loc[:, ["group_key", "source_corpus"]].drop_duplicates(), on="group_key"), ("source_corpus", "split")),
            "voices_by_language_and_split": _counts_by(trials.loc[:, ["voice_key", "language", "split"]].drop_duplicates(), ("language", "split")),
            "b2_pairs_by_split_language_corpus_generator": _counts_by(random_pairs, ("split", "language", "source_corpus", "spoof_generator")),
        },
        "trials_rows_sha256": _sha256_json(trials.to_dict(orient="records")),
        "pairs_rows_sha256": _sha256_json(pairs.to_dict(orient="records")),
        "random_pairs_rows_sha256": _sha256_json(random_pairs.to_dict(orient="records")),
    }
    return H9ODSSPairingFreeze(
        trials=trials,
        pairs=pairs,
        random_pairs=random_pairs,
        excluded_unmatched=excluded_unmatched,
        provenance=provenance,
    )


def freeze_h9_odss_pairing_from_files(
    *,
    metadata_path: str | Path,
    semantics_documents: Mapping[str, str | Path],
    protocol_documents: Mapping[str, str | Path],
) -> H9ODSSPairingFreeze:
    """Read validated ODSS metadata/documents and return a source-only freeze."""
    expected_protocol_documents = {"TASK.md", "DATA.md", "PLAN.md", "BRAINSTORM.md"}
    if set(protocol_documents) != expected_protocol_documents:
        raise ValueError(f"H9 protocol documents must be exactly {sorted(expected_protocol_documents)}")
    rows = _read_metadata_table(metadata_path)
    semantics_records = _validate_semantics_documents(semantics_documents)
    protocol_records = {name: _path_record(path) for name, path in sorted(protocol_documents.items())}
    return build_h9_odss_pairing_freeze(
        rows,
        metadata_record=_path_record(metadata_path),
        semantics_records=semantics_records,
        protocol_records=protocol_records,
    )


def write_h9_odss_pairing_artifacts(
    freeze: H9ODSSPairingFreeze,
    *,
    output_dir: str | Path,
    allowed_output_root: str | Path = HDD_ROOT,
) -> dict[str, Path]:
    """Write a non-overwriting source-only freeze below the designated HDD root."""
    target = Path(output_dir).resolve()
    allowed_root = Path(allowed_output_root).resolve()
    try:
        target.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(f"H9 ODSS source artifacts must be below {allowed_root}, got {target}") from error
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing H9 ODSS freeze directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    trials_path = target / "h9_odss_source_trials.csv"
    pairs_path = target / "h9_odss_source_pairs.csv"
    random_pairs_path = target / "h9_odss_b2_random_pairs.csv"
    excluded_path = target / "h9_odss_excluded_unmatched.csv"
    provenance_path = target / "h9_odss_source_freeze.json"
    freeze.trials.to_csv(trials_path, index=False)
    freeze.pairs.to_csv(pairs_path, index=False)
    freeze.random_pairs.to_csv(random_pairs_path, index=False)
    freeze.excluded_unmatched.to_csv(excluded_path, index=False)
    provenance = dict(freeze.provenance)
    provenance["outputs"] = {
        "trials_csv": {"path": str(trials_path), "sha256": sha256_file(trials_path), "n_rows": int(len(freeze.trials))},
        "pairs_csv": {"path": str(pairs_path), "sha256": sha256_file(pairs_path), "n_rows": int(len(freeze.pairs))},
        "b2_random_pairs_csv": {
            "path": str(random_pairs_path),
            "sha256": sha256_file(random_pairs_path),
            "n_rows": int(len(freeze.random_pairs)),
        },
        "excluded_unmatched_csv": {
            "path": str(excluded_path),
            "sha256": sha256_file(excluded_path),
            "n_rows": int(len(freeze.excluded_unmatched)),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "trials": trials_path,
        "pairs": pairs_path,
        "random_pairs": random_pairs_path,
        "excluded_unmatched": excluded_path,
        "provenance": provenance_path,
    }
