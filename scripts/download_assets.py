#!/usr/bin/env python3
"""Resumably download pinned public data or Arena score artifacts to the HDD."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


GROUPS = {
    "discovery": ["ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF"],
    "confirmation": ["InTheWild", "ASVspoof5"],
    "fallback": ["DeepVoice", "LibriSeVoc", "XMAD", "CFAD", "CVoiceFake_small", "DECRO"],
}


def read_index(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def select_datasets(index: dict, group: str) -> list[str]:
    names = GROUPS[group] if group != "all" else list(index["datasets"])
    return [name for name in names if name in index["datasets"]]


def download_dataset(index: dict, name: str, workers: int) -> None:
    item = index["datasets"][name]
    print(f"[dataset] {name} @ {item['revision']}", flush=True)
    snapshot_download(
        repo_id=item["repo_id"],
        repo_type="dataset",
        revision=item["revision"],
        allow_patterns=["data/*.parquet", "README.md", "LICENSE*", "eval.yaml", "protocols/*.txt"],
        local_dir=item["local_dir"],
        max_workers=workers,
    )
    print(f"[dataset] complete {name}", flush=True)


def download_scores(index: dict, model_name: str, workers: int) -> None:
    item = index["models"][model_name]
    patterns = ["README.md", "meta.yaml", "*.onnx", "config.json", "model.py"]
    for artifact in item["score_artifacts"].values():
        patterns.append(artifact["scores"]["path"])
        if artifact.get("result"):
            patterns.append(artifact["result"]["path"])
    print(f"[scores] {model_name} @ {item['revision']}", flush=True)
    snapshot_download(
        repo_id=item["repo_id"],
        revision=item["revision"],
        allow_patterns=patterns,
        local_dir=item["local_dir"],
        max_workers=workers,
    )
    print(f"[scores] complete {model_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["datasets", "scores"])
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--group", choices=["discovery", "confirmation", "fallback", "all"], default="discovery")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    index = read_index(Path(args.index))
    if args.kind == "datasets":
        for name in select_datasets(index, args.group):
            download_dataset(index, name, args.workers)
    else:
        model_names = args.models or list(index["models"])
        for name in model_names:
            download_scores(index, name, args.workers)


if __name__ == "__main__":
    main()
