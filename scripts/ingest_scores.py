#!/usr/bin/env python3
"""Create compact, canonical score tables from pinned Arena artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arena_io import load_model_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/scores")
    parser.add_argument("--summary", default="experiments/h1_feature_association/results/score_catalog.csv")
    args = parser.parse_args()

    index = yaml.safe_load(Path(args.index).read_text())
    output_root = Path(args.output_root)
    summaries: list[dict[str, object]] = []
    for dataset_name in args.dataset:
        if dataset_name not in index["datasets"]:
            raise KeyError(f"Unknown dataset: {dataset_name}")
        frames: list[pd.DataFrame] = []
        for model_name in index["models"]:
            if dataset_name not in index["models"][model_name]["score_artifacts"]:
                continue
            frame = load_model_scores(index, dataset_name, model_name)
            frames.append(frame)
            summaries.append({
                "dataset": dataset_name,
                "model": model_name,
                "n_scores_joined": len(frame),
                "score_orientation": frame["orientation"].iloc[0],
                "score_revision": index["models"][model_name]["revision"],
                "dataset_revision": index["datasets"][dataset_name]["revision"],
            })
        if not frames:
            raise RuntimeError(f"No downloaded score artifacts for {dataset_name}")
        output = pd.concat(frames, ignore_index=True)
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output.to_parquet(dataset_dir / "score_panel.parquet", index=False)
        (dataset_dir / "summary.json").write_text(json.dumps([row for row in summaries if row["dataset"] == dataset_name], indent=2))
        print(f"wrote {dataset_dir / 'score_panel.parquet'} ({len(output)} rows)")
    summary = pd.DataFrame(summaries)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
