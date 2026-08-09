"""Pinned Arena artifact loading and strict sample-ID handling."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf


def normalize_sample_id(value: str) -> str:
    """Return the documented cross-artifact ID: basename without one suffix."""
    return Path(str(value)).stem


def load_labels(dataset: dict[str, Any]) -> pd.DataFrame:
    """Load the Arena-pinned labels table without inferring labels from paths."""
    labels_path = Path(dataset["local_dir"]) / dataset["files"]["labels"]
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing pinned labels table: {labels_path}")
    table = pq.read_table(labels_path)
    frame = table.to_pandas()
    id_column = next((name for name in ("utterance_id", "sample_id", "path") if name in frame), None)
    if id_column is None or "label" not in frame:
        raise ValueError(f"Unsupported labels schema in {labels_path}: {table.schema}")
    result = frame.rename(columns={id_column: "source_id"})[["source_id", "label"]].copy()
    result["sample_id"] = result["source_id"].map(normalize_sample_id)
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype("int8")
    if result["sample_id"].duplicated().any():
        raise ValueError(f"Non-unique normalized IDs in {labels_path}")
    return result[["sample_id", "source_id", "label"]]


def deterministic_balanced_subset(labels: pd.DataFrame, maximum_full_examples: int, per_label: int, seed: int) -> pd.DataFrame:
    """Use all small datasets; otherwise select a stable, balanced H1 subset."""
    if len(labels) <= maximum_full_examples:
        return labels.sort_values("sample_id").reset_index(drop=True)
    groups: list[pd.DataFrame] = []
    for label, group in labels.groupby("label", sort=True):
        count = min(len(group), per_label)
        if count == 0:
            continue
        groups.append(group.sample(n=count, random_state=seed + int(label)))
    result = pd.concat(groups, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if result["label"].nunique() != labels["label"].nunique():
        raise ValueError("Balanced selection removed a class")
    return result


def parse_scores(path: Path) -> pd.DataFrame:
    """Parse standard Arena `utterance_id score` text without guessing columns."""
    rows: list[tuple[str, float]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"Expected two fields in {path}:{line_number}, got {len(fields)}")
        sample_id, score = fields
        rows.append((normalize_sample_id(sample_id), float(score)))
    output = pd.DataFrame(rows, columns=["sample_id", "raw_score"])
    if output.empty or output["sample_id"].duplicated().any():
        raise ValueError(f"Invalid or duplicate score IDs in {path}")
    return output


def load_model_scores(index: dict[str, Any], dataset_name: str, model_name: str) -> pd.DataFrame:
    """Load a score artifact pinned in the catalogue and join its authoritative labels."""
    dataset = index["datasets"][dataset_name]
    model = index["models"][model_name]
    artifact = model["score_artifacts"].get(dataset_name)
    if artifact is None:
        return pd.DataFrame(columns=["sample_id", "label", "raw_score", "score_spoof", "model", "dataset"])
    scores_path = Path(model["local_dir"]) / artifact["scores"]["path"]
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing pinned score artifact: {scores_path}")
    labels = load_labels(dataset)[["sample_id", "label"]]
    joined = parse_scores(scores_path).merge(labels, on="sample_id", how="inner", validate="one_to_one")
    if len(joined) == 0:
        raise ValueError(f"No score-label joins for {model_name} on {dataset_name}")
    # Arena logits can represent either bonafide or spoof. Canonicalize to
    # increasing spoof evidence solely from labels in the same pinned artifact.
    rank = joined["raw_score"].rank(method="average")
    corr = np.corrcoef(rank, joined["label"])[0, 1]
    if not np.isfinite(corr) or abs(corr) < 0.02:
        raise ValueError(f"Ambiguous score orientation for {model_name} on {dataset_name}: {corr}")
    joined["score_spoof"] = joined["raw_score"] if corr > 0 else -joined["raw_score"]
    joined["model"] = model_name
    joined["dataset"] = dataset_name
    joined["orientation"] = "raw_is_spoof" if corr > 0 else "negated_raw_is_spoof"
    return joined[["sample_id", "label", "raw_score", "score_spoof", "model", "dataset", "orientation"]]


def _audio_bytes(audio_field: Any) -> bytes:
    if isinstance(audio_field, dict):
        payload = audio_field.get("bytes")
    else:
        payload = audio_field
    if payload is None:
        raise ValueError("Audio bytes unavailable in local Parquet row")
    return bytes(payload)


@dataclass(frozen=True)
class AudioRecord:
    sample_id: str
    label: int
    source_id: str
    audio_bytes: bytes
    notes: dict[str, Any]


def iter_selected_audio(dataset: dict[str, Any], selected: pd.DataFrame, batch_size: int = 256) -> Iterator[AudioRecord]:
    """Stream selected rows from local Parquet shards without positional joins."""
    root = Path(dataset["local_dir"])
    shards = sorted((root / "data").glob("test-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No downloaded audio shards in {root / 'data'}")
    selected_map = selected.set_index("sample_id")[["source_id", "label"]].to_dict("index")
    seen: set[str] = set()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["path", "audio", "label", "notes"]):
            for row in batch.to_pylist():
                sample_id = normalize_sample_id(row["path"])
                if sample_id not in selected_map:
                    continue
                selected_row = selected_map[sample_id]
                if int(row["label"]) != int(selected_row["label"]):
                    raise ValueError(f"Label mismatch for {sample_id} in {shard}")
                notes_raw = row.get("notes")
                notes = json.loads(notes_raw) if isinstance(notes_raw, str) and notes_raw else {}
                seen.add(sample_id)
                yield AudioRecord(
                    sample_id=sample_id,
                    label=int(selected_row["label"]),
                    source_id=str(selected_row["source_id"]),
                    audio_bytes=_audio_bytes(row["audio"]),
                    notes=notes,
                )
    missing = set(selected_map).difference(seen)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Selected samples missing from local shards ({len(missing)}): {preview}")


def decode_audio(payload: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)
