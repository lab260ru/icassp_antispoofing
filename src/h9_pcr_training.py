"""Source-only training primitives for the locked H9-PCR study.

The module deliberately knows nothing about SONAR or ArAD.  It accepts a
*source* manifest whose paired-content eligibility has already been frozen,
starts Res2TCNGuard from fresh seeded initialization, and implements the three
predeclared objectives:

``B1``
    BCE on every paired-eligible source row.
``B2``
    BCE plus a deterministic, stratum-matched random opposite-class margin.
``P``
    BCE plus the matched-content natural-to-synthetic margin.

The input contract is intentionally strict.  A future source-manifest builder
must prove the pairing and voice-disjoint split before this code is allowed to
load waveforms.  No target path, label reader, metric, or checkpoint loader is
present here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from src.onnx_fixed_window import WINDOW_SAMPLES, fixed_first_window_tiled
from src.res2tcn_pytorch import sha256_file


H9_SEEDS: tuple[int, ...] = (9101, 9102, 9103, 9104)
H9_SEED_DEVICES: Mapping[int, str] = {
    9101: "cuda:0",
    9102: "cuda:1",
    9103: "cuda:2",
    9104: "cuda:3",
}
H9_LAMBDA_GRID: tuple[float, ...] = (0.10, 0.30, 1.00)
H9_MARGIN = 1.0
Method = Literal["B1", "B2", "P"]
REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "audio_path",
    "label",
    "split",
    "pair_id",
    "group_id",
    "language",
    "canonical_fingerprint",
    "source_corpus",
    "speaker_id",
    "spoof_generator",
)


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Encode a compact, stable JSON record for a run ledger."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_int(*parts: object) -> int:
    """Map immutable identifiers to an unbiased deterministic integer."""
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big", signed=False)


def _require_text(frame: pd.DataFrame, column: str) -> None:
    values = frame[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"H9 source manifest requires nonempty {column} values")


@dataclass(frozen=True)
class SourceRecord:
    """One source-only audio row from the immutable H9 input manifest."""

    sample_id: str
    audio_path: str
    label: int
    split: str
    pair_id: str
    group_id: str
    language: str
    canonical_fingerprint: str
    source_corpus: str
    speaker_id: str
    spoof_generator: str

    @property
    def voice_key(self) -> str:
        """The frozen split grouping: source corpus plus speaker/voice."""
        return f"{self.source_corpus}|{self.speaker_id}"


@dataclass(frozen=True)
class PairEdge:
    """A one-spoof to one-bona-fide margin edge from an eligible pair."""

    split: str
    pair_id: str
    spoof_id: str
    bona_id: str
    language: str
    source_corpus: str
    spoof_generator: str

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.language, self.source_corpus, self.spoof_generator)


@dataclass(frozen=True)
class SourceManifest:
    """Validated paired source manifest and all immutable matched edges."""

    records: Mapping[str, SourceRecord]
    paired_eligible_ids: Mapping[str, tuple[str, ...]]
    pair_edges: Mapping[str, tuple[PairEdge, ...]]
    source_manifest_sha256: str

    def records_for_split(self, split: str) -> tuple[SourceRecord, ...]:
        return tuple(self.records[sample_id] for sample_id in self.paired_eligible_ids[split])


def _read_manifest_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"H9 source manifest is unavailable: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError("H9 source manifest must be CSV or Parquet")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("H9 source manifest did not decode to a table")
    return frame


def load_source_manifest(path: str | Path) -> SourceManifest:
    """Read and hard-validate the H9 **source-only** pairing contract.

    Each ``pair_id`` must have exactly one bona-fide row and exactly two spoof
    rows from distinct named generators.  All members share a split, a voice
    key, corpus, and language.  ``group_id`` is verified to be the mandatory
    ``source_corpus|speaker_id`` voice key, so source development is
    voice-disjoint rather than merely content-disjoint.
    """
    manifest_path = Path(path)
    frame = _read_manifest_frame(manifest_path)
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"H9 source manifest lacks required columns: {missing}")
    frame = frame.loc[:, REQUIRED_SOURCE_COLUMNS].copy()
    for column in REQUIRED_SOURCE_COLUMNS:
        # Fingerprints may be populated only after the source audio audit. The
        # column must already exist so that a later source freeze cannot drift.
        if column != "canonical_fingerprint":
            _require_text(frame, column)
    if frame["sample_id"].duplicated().any():
        raise ValueError("H9 source manifest has duplicate sample_id values")
    try:
        frame["label"] = frame["label"].astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("H9 source labels must be integer binary values") from error
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("H9 source manifest needs both binary labels 0/1")
    if set(frame["split"].unique()) - {"train", "dev"}:
        raise ValueError("H9 source manifest split must be exactly train/dev rows")
    if set(frame["split"].unique()) != {"train", "dev"}:
        raise ValueError("H9 source manifest must contain both train and dev")
    expected_group = frame["source_corpus"].astype(str) + "|" + frame["speaker_id"].astype(str)
    if not frame["group_id"].astype(str).eq(expected_group).all():
        raise ValueError("H9 group_id must equal source_corpus|speaker_id (voice-disjoint split key)")
    split_counts = frame.groupby("group_id", sort=False)["split"].nunique()
    if (split_counts != 1).any():
        offenders = sorted(split_counts.index[split_counts != 1].astype(str).tolist())[:5]
        raise ValueError(f"H9 voice group crosses train/dev split: {offenders}")

    records: dict[str, SourceRecord] = {}
    for row in frame.itertuples(index=False):
        record = SourceRecord(
            sample_id=str(row.sample_id),
            audio_path=str(row.audio_path),
            label=int(row.label),
            split=str(row.split),
            pair_id=str(row.pair_id),
            group_id=str(row.group_id),
            language=str(row.language),
            canonical_fingerprint=str(row.canonical_fingerprint),
            source_corpus=str(row.source_corpus),
            speaker_id=str(row.speaker_id),
            spoof_generator=str(row.spoof_generator),
        )
        if record.label == 0 and record.spoof_generator not in {"bonafide", "bona_fide", "natural"}:
            raise ValueError("H9 bona-fide rows must mark spoof_generator as bonafide/bona_fide/natural")
        if record.label == 1 and record.spoof_generator in {"bonafide", "bona_fide", "natural"}:
            raise ValueError("H9 spoof rows require a named non-bona-fide generator")
        records[record.sample_id] = record

    edges_by_split: dict[str, list[PairEdge]] = {"train": [], "dev": []}
    eligible_ids_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    for pair_id, rows in frame.groupby("pair_id", sort=True):
        row_records = [records[str(sample_id)] for sample_id in rows["sample_id"].tolist()]
        bona = [record for record in row_records if record.label == 0]
        spoof = [record for record in row_records if record.label == 1]
        if len(bona) != 1 or len(spoof) != 2:
            raise ValueError(f"H9 pair_id {pair_id!r} requires one bona-fide and two spoof rows")
        if len({record.spoof_generator for record in spoof}) != 2:
            raise ValueError(f"H9 pair_id {pair_id!r} spoof rows must use two distinct generators")
        for column, values in {
            "split": {record.split for record in row_records},
            "group_id": {record.group_id for record in row_records},
            "language": {record.language for record in row_records},
            "source_corpus": {record.source_corpus for record in row_records},
        }.items():
            if len(values) != 1:
                raise ValueError(f"H9 pair_id {pair_id!r} crosses {column}")
        bona_record = bona[0]
        split = bona_record.split
        eligible_ids_by_split[split].update(record.sample_id for record in row_records)
        for spoof_record in sorted(spoof, key=lambda record: record.sample_id):
            edges_by_split[split].append(
                PairEdge(
                    split=split,
                    pair_id=str(pair_id),
                    spoof_id=spoof_record.sample_id,
                    bona_id=bona_record.sample_id,
                    language=spoof_record.language,
                    source_corpus=spoof_record.source_corpus,
                    spoof_generator=spoof_record.spoof_generator,
                )
            )
    for split in ("train", "dev"):
        if not edges_by_split[split]:
            raise ValueError(f"H9 source manifest has no complete matched pairs in {split}")
        labels = {records[sample_id].label for sample_id in eligible_ids_by_split[split]}
        if labels != {0, 1}:
            raise ValueError(f"H9 paired-eligible {split} rows do not contain both labels")
    digest = sha256_file(manifest_path)
    return SourceManifest(
        records=records,
        paired_eligible_ids={
            split: tuple(sorted(sample_ids)) for split, sample_ids in eligible_ids_by_split.items()
        },
        pair_edges={
            split: tuple(sorted(edges, key=lambda edge: (edge.pair_id, edge.spoof_id)))
            for split, edges in edges_by_split.items()
        },
        source_manifest_sha256=digest,
    )


def seed_to_device(seed: int, requested_device: str = "auto") -> str:
    """Return the predeclared one-seed-per-physical-GPU H9 assignment."""
    if seed not in H9_SEED_DEVICES:
        raise ValueError(f"H9 seed must be one of {H9_SEEDS}, got {seed}")
    assigned = H9_SEED_DEVICES[seed]
    if requested_device not in {"auto", assigned}:
        raise ValueError(f"H9 seed {seed} is frozen to {assigned}, not {requested_device}")
    return assigned


def seed_everything(seed: int) -> None:
    """Set all local initialization/sampler RNGs before a fresh model build."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_res2_architecture(bundle_dir: Path) -> ModuleType:
    """Load only the pinned architecture definition; never read a checkpoint."""
    network_path = bundle_dir / "_net.py"
    if not network_path.is_file():
        raise FileNotFoundError(f"Res2TCNGuard architecture file is unavailable: {network_path}")
    module_name = f"_h9_fresh_res2_arch_{sha256_file(network_path)[:16]}"
    existing = __import__("sys").modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, network_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pinned Res2TCNGuard architecture: {network_path}")
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_fresh_res2tcn_guard(
    bundle_dir: str | Path,
    *,
    seed: int,
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Create a fresh seeded Res2TCNGuard model without checkpoint loading.

    The public ``best_1.495.pth`` bundle is intentionally not consulted: it
    was trained outside H9 and would contaminate the fresh-start comparison.
    """
    bundle_path = Path(bundle_dir)
    seed_everything(seed)
    architecture = _load_res2_architecture(bundle_path)
    test_model = getattr(architecture, "TestModel", None)
    if not isinstance(test_model, type) or not issubclass(test_model, nn.Module):
        raise ValueError("Pinned Res2TCNGuard _net.py does not expose TestModel")
    model = test_model().to(device)
    return model, {
        "architecture_path": str(bundle_path / "_net.py"),
        "architecture_sha256": sha256_file(bundle_path / "_net.py"),
        "initialization": "fresh_seeded",
        "initialization_seed": int(seed),
        "checkpoint_loaded": False,
    }


def _load_audio_16k(path: str) -> np.ndarray:
    """Read one source waveform and apply the shared deterministic window."""
    try:
        import soundfile as sound_file
    except ImportError as error:  # pragma: no cover - dependency installed in runtime
        raise RuntimeError("H9 source waveform loading needs soundfile") from error
    waveform, sample_rate = sound_file.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    waveform = waveform.reshape(-1)
    if waveform.size == 0:
        raise ValueError(f"H9 source waveform is empty: {path}")
    if int(sample_rate) != 16000:
        try:
            from scipy.signal import resample_poly
        except ImportError as error:  # pragma: no cover - scipy is already available
            raise RuntimeError("H9 source resampling needs scipy") from error
        divisor = math.gcd(int(sample_rate), 16000)
        waveform = resample_poly(waveform, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    return np.ascontiguousarray(fixed_first_window_tiled(waveform), dtype=np.float32)


class SourceWaveformDataset(Dataset[dict[str, Any]]):
    """Waveform dataset restricted to already paired-eligible source rows."""

    def __init__(
        self,
        records: Sequence[SourceRecord],
        *,
        waveform_loader: Callable[[str], np.ndarray] = _load_audio_16k,
    ) -> None:
        self.records = tuple(records)
        self.waveform_loader = waveform_loader

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = np.asarray(self.waveform_loader(record.audio_path), dtype=np.float32)
        if waveform.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"H9 fixed waveform must have {WINDOW_SAMPLES} samples, got {waveform.shape}")
        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform)),
            "label": torch.tensor(record.label, dtype=torch.long),
            "sample_id": record.sample_id,
            "language": record.language,
        }


class PairWaveformDataset(Dataset[dict[str, Any]]):
    """A deterministic list of spoof→bona source ranking edges."""

    def __init__(
        self,
        edges: Sequence[PairEdge],
        records: Mapping[str, SourceRecord],
        *,
        waveform_loader: Callable[[str], np.ndarray] = _load_audio_16k,
    ) -> None:
        self.edges = tuple(edges)
        self.records = records
        self.waveform_loader = waveform_loader

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, index: int) -> dict[str, Any]:
        edge = self.edges[index]
        spoof = np.asarray(self.waveform_loader(self.records[edge.spoof_id].audio_path), dtype=np.float32)
        bona = np.asarray(self.waveform_loader(self.records[edge.bona_id].audio_path), dtype=np.float32)
        if spoof.shape != (WINDOW_SAMPLES,) or bona.shape != (WINDOW_SAMPLES,):
            raise ValueError("H9 paired waveforms must use the fixed shared window length")
        return {
            "spoof_waveform": torch.from_numpy(np.ascontiguousarray(spoof)),
            "bona_waveform": torch.from_numpy(np.ascontiguousarray(bona)),
            "spoof_id": edge.spoof_id,
            "bona_id": edge.bona_id,
            "pair_id": edge.pair_id,
            "stratum": "|".join(edge.stratum),
        }


class BalancedLanguageBatchSampler(Sampler[list[int]]):
    """Deterministic class-balanced batches with round-robin language draws."""

    def __init__(self, records: Sequence[SourceRecord], *, batch_size: int, seed: int, epoch: int) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("H9 BCE batch_size must be an even integer of at least two")
        self.records = tuple(records)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.by_label_language: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
        for index, record in enumerate(self.records):
            self.by_label_language[record.label].setdefault(record.language, []).append(index)
        if not self.by_label_language[0] or not self.by_label_language[1]:
            raise ValueError("H9 BCE batches require both classes")
        self.num_batches = math.ceil(max(sum(map(len, self.by_label_language[0].values())), sum(map(len, self.by_label_language[1].values()))) / (batch_size // 2))

    def __len__(self) -> int:
        return self.num_batches

    def _draws(self, label: int, total: int) -> list[int]:
        groups = self.by_label_language[label]
        languages = tuple(sorted(groups))
        per_language: dict[str, list[int]] = {}
        for language in languages:
            values = sorted(groups[language], key=lambda index: self.records[index].sample_id)
            rng = np.random.default_rng(_stable_int("h9-bce", self.seed, self.epoch, label, language))
            per_language[language] = [values[index] for index in rng.permutation(len(values)).tolist()]
        offsets = {language: 0 for language in languages}
        result: list[int] = []
        for draw in range(total):
            language = languages[draw % len(languages)]
            values = per_language[language]
            result.append(values[offsets[language] % len(values)])
            offsets[language] += 1
        return result

    def __iter__(self) -> Iterator[list[int]]:
        half = self.batch_size // 2
        class_zero = self._draws(0, self.num_batches * half)
        class_one = self._draws(1, self.num_batches * half)
        for batch_index in range(self.num_batches):
            batch = class_zero[batch_index * half : (batch_index + 1) * half] + class_one[batch_index * half : (batch_index + 1) * half]
            rng = np.random.default_rng(_stable_int("h9-bce-batch", self.seed, self.epoch, batch_index))
            order = rng.permutation(len(batch)).tolist()
            yield [batch[index] for index in order]


def _cycle_edges(edges: Sequence[PairEdge], *, wanted: int) -> tuple[PairEdge, ...]:
    if not edges or wanted <= 0:
        raise ValueError("H9 rank schedule requires a nonempty edge sequence")
    return tuple(edges[index % len(edges)] for index in range(wanted))


def build_random_pair_edges(
    matched_edges: Sequence[PairEdge],
    records: Mapping[str, SourceRecord],
    *,
    seed: int,
    epoch: int,
) -> tuple[tuple[PairEdge, ...], dict[str, Any]]:
    """Build B2 edges, preserving P's spoof IDs and stratum/count exactly.

    Candidate bona-fide clips are derived from matched P edges, then stratified
    by the spoof edge's language, source corpus, and generator.  The exact
    same-content ``pair_id`` is excluded, giving a label-valid but
    content-unmatched control.  Hash selection makes the partner independent
    of worker order and deterministic from ``(seed, epoch, spoof_id)``.
    """
    candidates: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    for edge in matched_edges:
        candidates.setdefault(edge.stratum, []).append((edge.pair_id, edge.bona_id))
    output: list[PairEdge] = []
    audit: dict[str, dict[str, int]] = {}
    for edge in matched_edges:
        valid = sorted(
            {bona_id for pair_id, bona_id in candidates[edge.stratum] if pair_id != edge.pair_id}
        )
        if not valid:
            raise ValueError(
                "H9 B2 random control lacks a same-language/corpus/generator "
                f"content-unmatched bona partner for pair_id={edge.pair_id!r}, stratum={edge.stratum}"
            )
        selected = valid[_stable_int("h9-b2", seed, epoch, edge.spoof_id) % len(valid)]
        if records[selected].label != 0:
            raise AssertionError("H9 B2 construction selected a non-bona-fide partner")
        output.append(
            PairEdge(
                split=edge.split,
                pair_id=f"random:{edge.pair_id}",
                spoof_id=edge.spoof_id,
                bona_id=selected,
                language=edge.language,
                source_corpus=edge.source_corpus,
                spoof_generator=edge.spoof_generator,
            )
        )
        key = "|".join(edge.stratum)
        audit.setdefault(key, {"edge_count": 0, "candidate_bona_count": len(valid)})["edge_count"] += 1
    return tuple(output), {
        "kind": "H9_B2_STRATIFIED_RANDOM_PAIR_AUDIT",
        "seed": int(seed),
        "epoch": int(epoch),
        "edge_count": len(output),
        "strata": audit,
        "all_bona_are_content_unmatched": True,
    }


def pairwise_margin_loss(spoof_logits: torch.Tensor, bona_logits: torch.Tensor, *, margin: float = H9_MARGIN) -> torch.Tensor:
    """The protocol's exact mean hinge: max(0, 1 - [z_spoof - z_bona])."""
    if spoof_logits.ndim != 1 or bona_logits.ndim != 1 or spoof_logits.shape != bona_logits.shape:
        raise ValueError("H9 pair margin requires same-shape one-dimensional logits")
    if not torch.isfinite(spoof_logits).all() or not torch.isfinite(bona_logits).all():
        raise FloatingPointError("H9 pair margin received non-finite logits")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("H9 pair margin must be finite and positive")
    return torch.relu(float(margin) - (spoof_logits - bona_logits)).mean()


def _spoof_logit(model_output: Any) -> torch.Tensor:
    logits = model_output[1] if isinstance(model_output, tuple) else model_output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("H9 Res2TCNGuard model must return (batch, 2) logits")
    # H9 fresh training defines index 1 as the spoof class throughout BCE,
    # ranking, EER, and emitted orientation provenance.
    return logits[:, 1]


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("H9 source EER needs both classes and finite one-score-per-row predictions")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-15))
    if len(exact):
        return float(fpr[int(exact[0])])
    crossings = np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:]))
    if not len(crossings):
        nearest = int(np.argmin(np.abs(difference)))
        return float((fpr[nearest] + fnr[nearest]) / 2.0)
    index = int(crossings[0])
    low, high = float(fpr[index]), float(fpr[index + 1])
    if math.isclose(low, high, abs_tol=1e-15):
        return low
    alpha = -float(difference[index]) / float(difference[index + 1] - difference[index])
    return float(low + alpha * (high - low))


@dataclass(frozen=True)
class TrainingConfig:
    """A completely recorded source-only H9 training recipe."""

    method: Method
    seed: int
    device: str
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    lambda_rank: float = 0.0
    margin: float = H9_MARGIN
    require_cuda: bool = True

    def validate(self) -> None:
        if self.method not in {"B1", "B2", "P"}:
            raise ValueError("H9 method must be B1, B2, or P")
        if self.seed not in H9_SEEDS:
            raise ValueError(f"H9 seed must be one of {H9_SEEDS}")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("H9 batch_size must be an even integer at least two")
        if self.num_workers < 0 or self.max_epochs <= 0:
            raise ValueError("H9 num_workers must be nonnegative and max_epochs positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("H9 learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("H9 weight_decay must be finite and nonnegative")
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("H9 margin must be finite and positive")
        if self.method == "B1" and self.lambda_rank != 0.0:
            raise ValueError("H9 B1 must use exactly zero ranking weight")
        if self.method in {"B2", "P"} and self.lambda_rank not in H9_LAMBDA_GRID:
            raise ValueError(f"H9 B2/P lambda_rank must be one of {H9_LAMBDA_GRID}")
        if self.require_cuda:
            assigned = seed_to_device(self.seed, self.device)
            if not torch.cuda.is_available():
                raise RuntimeError("H9 production training requires CUDA BF16; CPU is not a fallback")
            if assigned != self.device:
                raise AssertionError("H9 seed/device mapping drift")


@dataclass(frozen=True)
class TrainingResult:
    """Compact source-development output for a single method and seed."""

    method: Method
    seed: int
    device: str
    lambda_rank: float
    best_epoch: int
    source_dev_eer: float
    source_manifest_sha256: str
    architecture_provenance: Mapping[str, Any]
    pairing_audit: Sequence[Mapping[str, Any]]
    checkpoint_path: str | None
    checkpoint_sha256: str | None

    def jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        value["orientation"] = "logit_1_is_spoof_evidence"
        value["target_labels_read"] = False
        value["target_audio_read"] = False
        return value


class H9PCRTrainer:
    """One source-only BF16 H9 fit, with immutable dev checkpoint selection."""

    def __init__(
        self,
        manifest: SourceManifest,
        config: TrainingConfig,
        *,
        bundle_dir: str | Path,
        waveform_loader: Callable[[str], np.ndarray] = _load_audio_16k,
        model_factory: Callable[[int, str | torch.device], tuple[nn.Module, dict[str, Any]]] = build_fresh_res2tcn_guard,
    ) -> None:
        config.validate()
        self.manifest = manifest
        self.config = config
        self.bundle_dir = Path(bundle_dir)
        self.waveform_loader = waveform_loader
        self.model_factory = model_factory

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        device_type = torch.device(self.config.device).type
        if self.config.require_cuda:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # Synthetic tests can exercise an actual BF16 step on CPU. This is not
        # a production fallback and is intentionally unreachable from the CLI.
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    def _bce_loader(self, split: str, epoch: int) -> DataLoader[dict[str, Any]]:
        records = self.manifest.records_for_split(split)
        dataset = SourceWaveformDataset(records, waveform_loader=self.waveform_loader)
        sampler = BalancedLanguageBatchSampler(records, batch_size=self.config.batch_size, seed=self.config.seed, epoch=epoch)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.require_cuda,
            persistent_workers=bool(self.config.num_workers),
        )

    def _rank_loader(self, epoch: int, steps: int) -> tuple[DataLoader[dict[str, Any]], Mapping[str, Any]]:
        matched = self.manifest.pair_edges["train"]
        if self.config.method == "P":
            rank_edges = matched
            audit: Mapping[str, Any] = {
                "kind": "H9_P_MATCHED_PAIR_AUDIT",
                "epoch": int(epoch),
                "edge_count": len(matched),
                "all_matched_pair_edges_used_before_cycling": True,
            }
        elif self.config.method == "B2":
            rank_edges, audit = build_random_pair_edges(matched, self.manifest.records, seed=self.config.seed, epoch=epoch)
        else:
            raise AssertionError("B1 has no rank loader")
        half = self.config.batch_size // 2
        scheduled = _cycle_edges(rank_edges, wanted=steps * half)
        dataset = PairWaveformDataset(scheduled, self.manifest.records, waveform_loader=self.waveform_loader)
        return (
            DataLoader(
                dataset,
                batch_size=half,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.require_cuda,
                persistent_workers=bool(self.config.num_workers),
            ),
            audit,
        )

    def _evaluate_dev(self, model: nn.Module) -> float:
        model.eval()
        labels: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        records = self.manifest.records_for_split("dev")
        dataset = SourceWaveformDataset(records, waveform_loader=self.waveform_loader)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=self.config.require_cuda)
        with torch.no_grad():
            for batch in loader:
                waveforms = batch["waveform"].to(self.config.device, non_blocking=self.config.require_cuda)
                with self._autocast():
                    logits = _spoof_logit(model(waveforms))
                labels.append(batch["label"].detach().cpu().numpy())
                scores.append(logits.float().detach().cpu().numpy())
        return _eer(np.concatenate(labels), np.concatenate(scores))

    def fit(self, *, checkpoint_path: str | Path | None = None) -> TrainingResult:
        """Fit only source data and retain lower-dev-EER, lower-epoch checkpoint."""
        if self.config.require_cuda and not self.config.device.startswith("cuda:"):
            raise ValueError("H9 production BF16 run requires its assigned CUDA device")
        model, architecture_provenance = self.model_factory(self.bundle_dir, seed=self.config.seed, device=self.config.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        best_eer = math.inf
        best_epoch = -1
        audits: list[Mapping[str, Any]] = []
        final_checkpoint: Path | None = Path(checkpoint_path) if checkpoint_path is not None else None
        for epoch in range(1, self.config.max_epochs + 1):
            bce_loader = self._bce_loader("train", epoch)
            if self.config.method == "B1":
                iterator: Iterable[tuple[dict[str, Any], Mapping[str, Any] | None]] = ((batch, None) for batch in bce_loader)
            else:
                rank_loader, audit = self._rank_loader(epoch, len(bce_loader))
                audits.append(audit)
                iterator = zip(bce_loader, rank_loader, strict=True)
            for item in iterator:
                if self.config.method == "B1":
                    bce_batch, rank_batch = item
                else:
                    bce_batch, rank_batch = item
                optimizer.zero_grad(set_to_none=True)
                waveforms = bce_batch["waveform"].to(self.config.device, non_blocking=self.config.require_cuda)
                labels = bce_batch["label"].to(self.config.device, non_blocking=self.config.require_cuda)
                with self._autocast():
                    bce = functional.cross_entropy(_spoof_logit_to_logits(model(waveforms)), labels)
                    if rank_batch is None:
                        rank = torch.zeros((), device=waveforms.device, dtype=bce.dtype)
                    else:
                        spoof = rank_batch["spoof_waveform"].to(self.config.device, non_blocking=self.config.require_cuda)
                        bona = rank_batch["bona_waveform"].to(self.config.device, non_blocking=self.config.require_cuda)
                        rank = pairwise_margin_loss(_spoof_logit(model(spoof)), _spoof_logit(model(bona)), margin=self.config.margin)
                    loss = bce + float(self.config.lambda_rank) * rank
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"H9 non-finite {self.config.method} loss at epoch {epoch}")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise FloatingPointError(f"H9 non-finite {self.config.method} gradient at epoch {epoch}")
                optimizer.step()
            dev_eer = self._evaluate_dev(model)
            if dev_eer < best_eer:
                best_eer, best_epoch = dev_eer, epoch
                if final_checkpoint is not None:
                    _atomic_torch_save(
                        final_checkpoint,
                        {
                            "model_state_dict": model.state_dict(),
                            "training_config": asdict(self.config),
                            "source_manifest_sha256": self.manifest.source_manifest_sha256,
                            "architecture_provenance": architecture_provenance,
                            "selection_rule": "lowest_source_dev_eer_then_lower_epoch",
                            "best_epoch": epoch,
                            "source_dev_eer": dev_eer,
                        },
                    )
        if best_epoch < 0:
            raise RuntimeError("H9 training produced no source-development checkpoint")
        checkpoint_hash = sha256_file(final_checkpoint) if final_checkpoint is not None and final_checkpoint.is_file() else None
        return TrainingResult(
            method=self.config.method,
            seed=self.config.seed,
            device=self.config.device,
            lambda_rank=float(self.config.lambda_rank),
            best_epoch=best_epoch,
            source_dev_eer=float(best_eer),
            source_manifest_sha256=self.manifest.source_manifest_sha256,
            architecture_provenance=architecture_provenance,
            pairing_audit=tuple(audits),
            checkpoint_path=str(final_checkpoint) if final_checkpoint is not None else None,
            checkpoint_sha256=checkpoint_hash,
        )


def _spoof_logit_to_logits(model_output: Any) -> torch.Tensor:
    """Return the full two-class fresh-training logits for BCE."""
    logits = model_output[1] if isinstance(model_output, tuple) else model_output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("H9 Res2TCNGuard model must return (batch, 2) logits")
    return logits


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a caller-designated source checkpoint only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_p_lambda(source_dev_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select P's source-only lambda and require all four predeclared seeds.

    This selector deliberately accepts only ``P`` development outputs.  Its
    selected value is then supplied unchanged to both final P and B2 fits.
    """
    expected = {(lam, seed) for lam in H9_LAMBDA_GRID for seed in H9_SEEDS}
    observed: dict[tuple[float, int], float] = {}
    for row in source_dev_results:
        if row.get("method") != "P":
            raise ValueError("H9 lambda selection accepts P source-development records only")
        lam = float(row.get("lambda_rank"))
        seed = int(row.get("seed"))
        eer = float(row.get("source_dev_eer"))
        if (lam, seed) not in expected or not math.isfinite(eer):
            raise ValueError("H9 lambda selection record is outside the frozen grid/seeds or has invalid EER")
        if (lam, seed) in observed:
            raise ValueError("H9 lambda selection has duplicate lambda/seed records")
        observed[(lam, seed)] = eer
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise ValueError(f"H9 lambda selection requires every frozen P lambda/seed run; missing {missing}")
    rows = [
        {
            "lambda_rank": lam,
            "mean_source_dev_eer": float(np.mean([observed[(lam, seed)] for seed in H9_SEEDS])),
            "source_dev_eer_by_seed": {str(seed): observed[(lam, seed)] for seed in H9_SEEDS},
        }
        for lam in H9_LAMBDA_GRID
    ]
    selected = min(rows, key=lambda row: (float(row["mean_source_dev_eer"]), float(row["lambda_rank"])))
    return {
        "kind": "H9_P_SOURCE_ONLY_LAMBDA_SELECTION",
        "selected_lambda_rank": float(selected["lambda_rank"]),
        "selection_metric": "mean_source_dev_eer_over_four_predeclared_P_seeds",
        "tie_break": "lower_lambda_rank",
        "applies_unchanged_to": ["P", "B2"],
        "candidates": rows,
        "target_labels_read": False,
        "target_audio_read": False,
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a compact source-only ledger chosen explicitly by the caller."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) + "\n"
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", mode="w", encoding="utf-8", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_fit_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one source-only fresh-init H9-PCR Res2TCNGuard run.")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--res2-bundle", required=True, type=Path, help="Directory containing the pinned _net.py architecture only.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("B1", "B2", "P"))
    parser.add_argument("--seed", required=True, type=int, choices=H9_SEEDS)
    parser.add_argument("--device", default="auto", help="Must be auto or the frozen GPU for --seed.")
    parser.add_argument("--lambda-rank", type=float, default=None)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--max-epochs", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--weight-decay", required=True, type=float)
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; it exposes no target input and no checkpoint-loading flag."""
    args = _parse_fit_args(argv)
    device = seed_to_device(args.seed, args.device)
    lambda_rank = 0.0 if args.method == "B1" and args.lambda_rank is None else args.lambda_rank
    if lambda_rank is None:
        raise SystemExit("--lambda-rank is required for B2/P and must be the source-selected frozen value")
    config = TrainingConfig(
        method=args.method,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_rank=float(lambda_rank),
    )
    manifest = load_source_manifest(args.source_manifest)
    run_stem = f"h9_pcr_{args.method}_seed{args.seed}"
    result = H9PCRTrainer(manifest, config, bundle_dir=args.res2_bundle).fit(checkpoint_path=args.output_dir / f"{run_stem}.pt")
    write_json(args.output_dir / f"{run_stem}.json", result.jsonable())
    print(canonical_json(result.jsonable()))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through scripts/train_h9_pcr.py
    raise SystemExit(cli_main())
