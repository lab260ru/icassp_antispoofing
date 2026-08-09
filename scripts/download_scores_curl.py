#!/usr/bin/env python3
"""Download only pinned Arena score files via curl with identity encoding and resume."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess
from urllib.parse import quote

import yaml


CORE_DATASETS = ["ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF", "InTheWild", "ASVspoof5"]


@dataclass(frozen=True)
class Task:
    model: str
    repo_id: str
    revision: str
    artifact_path: str
    expected_bytes: int | None
    destination: Path


def download(task: Task) -> str:
    task.destination.parent.mkdir(parents=True, exist_ok=True)
    if task.expected_bytes is not None and task.destination.exists() and task.destination.stat().st_size == task.expected_bytes:
        return f"skip {task.model} {task.artifact_path}"
    url = f"https://huggingface.co/{task.repo_id}/resolve/{task.revision}/{quote(task.artifact_path, safe='/')}"
    command = [
        "curl", "-L", "--fail", "--silent", "--show-error", "--retry", "5", "--retry-all-errors",
        "--continue-at", "-", "-H", "Accept-Encoding: identity", url, "-o", str(task.destination),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"curl failed for {task.model}/{task.artifact_path}: {completed.stderr.strip()}")
    if task.expected_bytes is not None and task.destination.stat().st_size != task.expected_bytes:
        raise RuntimeError(f"size mismatch for {task.destination}: got {task.destination.stat().st_size}, expected {task.expected_bytes}")
    return f"done {task.model} {task.artifact_path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    index = yaml.safe_load(Path(args.index).read_text())
    datasets = args.datasets or CORE_DATASETS
    models = args.models or list(index["models"])
    tasks: list[Task] = []
    for model_name in models:
        model = index["models"][model_name]
        for dataset in datasets:
            artifact = model["score_artifacts"].get(dataset)
            if not artifact:
                continue
            for kind in ("scores", "result"):
                file_info = artifact.get(kind)
                if not file_info:
                    continue
                tasks.append(Task(
                    model=model_name,
                    repo_id=model["repo_id"],
                    revision=model["revision"],
                    artifact_path=file_info["path"],
                    expected_bytes=file_info.get("size_bytes"),
                    destination=Path(model["local_dir"]) / file_info["path"],
                ))
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(download, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(f"failed {task.model} {task.artifact_path}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} score artifacts failed")
    print(f"complete: {len(tasks)} score/result artifacts", flush=True)


if __name__ == "__main__":
    main()
