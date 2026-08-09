#!/usr/bin/env python3
"""Parallel feature extraction from downloaded, pinned audio Parquet shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arena_io import AudioRecord, decode_audio, deterministic_balanced_subset, iter_selected_audio, load_labels
from src.audio_features import FEATURE_NAMES, FEATURE_VERSION, compute_features, waveform_views


def process_record(record: AudioRecord) -> list[dict[str, object]]:
    audio, sample_rate = decode_audio(record.audio_bytes)
    output: list[dict[str, object]] = []
    for view, waveform in waveform_views(audio, sample_rate).items():
        row: dict[str, object] = {
            "sample_id": record.sample_id,
            "source_id": record.source_id,
            "label": record.label,
            # Arena datasets do not use one uniform metadata spelling. The
            # InTheWild shards expose an explicit `speaker` field in notes;
            # preserve it as the registered speaker control rather than
            # silently degrading to an utterance-only bootstrap cluster.
            "speaker_id": record.notes.get("speaker_id") or record.notes.get("speaker"),
            "attack_id": record.notes.get("attack_id") or record.notes.get("attack"),
            "view": view,
            "sample_rate_hz": sample_rate,
            "duration_seconds": len(audio) / max(sample_rate, 1),
        }
        row.update(compute_features(waveform))
        output.append(row)
    return output


def flatten(chunks: Iterable[list[dict[str, object]]]) -> Iterable[dict[str, object]]:
    for chunk in chunks:
        yield from chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--pilot-per-label", type=int, default=None)
    args = parser.parse_args()

    index = yaml.safe_load(Path(args.index).read_text())
    config = yaml.safe_load(Path("configs/study.yaml").read_text())
    output_root = Path(args.output_root)
    for dataset_name in args.dataset:
        dataset = index["datasets"][dataset_name]
        labels = load_labels(dataset)
        if args.pilot_per_label is None:
            selected = deterministic_balanced_subset(
                labels,
                config["datasets"]["maximum_full_examples"],
                config["datasets"]["subset_per_label"],
                config["study"]["seed"],
            )
            sample_mode = "confirmatory"
        else:
            selected = deterministic_balanced_subset(labels, 0, args.pilot_per_label, config["study"]["seed"])
            sample_mode = "pilot"
        artifact_version = FEATURE_VERSION if args.pilot_per_label is None else f"{FEATURE_VERSION}_pilot_{args.pilot_per_label}_per_label"
        target_dir = output_root / artifact_version / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(target_dir / "selected_manifest.parquet", index=False)
        records = iter_selected_audio(dataset, selected)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(flatten(executor.map(process_record, records, chunksize=4)))
        table = pd.DataFrame(rows)
        expected_columns = {"sample_id", "label", "view", *FEATURE_NAMES}
        if not expected_columns.issubset(table.columns):
            missing = expected_columns.difference(table.columns)
            raise RuntimeError(f"Feature extraction missing columns: {sorted(missing)}")
        table.to_parquet(target_dir / "features_wide.parquet", index=False)
        report = {
            "feature_version": FEATURE_VERSION,
            "dataset": dataset_name,
            "dataset_revision": dataset["revision"],
            "sample_mode": sample_mode,
            "n_selected_samples": len(selected),
            "n_feature_rows": len(table),
            "n_unique_samples": int(table["sample_id"].nunique()),
            "views": sorted(table["view"].unique().tolist()),
            "missing_fraction": {name: float(table[name].isna().mean()) for name in FEATURE_NAMES},
        }
        (target_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {target_dir / 'features_wide.parquet'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
