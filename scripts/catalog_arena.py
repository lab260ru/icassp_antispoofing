#!/usr/bin/env python3
"""Pin public Arena datasets, models, and score artifacts in one YAML registry."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download


MODEL_REPOS = {
    "Spectra-AASIST": "lab260/Spectra-AASIST",
    "AASIST": "SpeechAntiSpoofingBenchmarks/AASIST",
    "Res2TCNGuard": "SpeechAntiSpoofingBenchmarks/Res2TCNGuard",
    "RawTFNet": "SpeechAntiSpoofingBenchmarks/RawTFNet",
    "WhisperMFCCMesoNet": "SpeechAntiSpoofingBenchmarks/WhisperMFCCMesoNet",
    "W2V2-AASIST": "SpeechAntiSpoofingBenchmarks/W2V2-AASIST",
    "XLSR-SLS": "SpeechAntiSpoofingBenchmarks/XLSR-SLS",
    "RawBMamba": "SpeechAntiSpoofingBenchmarks/RawBMamba",
}


def file_metadata(sibling: Any) -> dict[str, Any]:
    lfs = getattr(sibling, "lfs", None)
    return {
        "path": sibling.rfilename,
        "size_bytes": getattr(sibling, "size", None),
        "sha256": getattr(lfs, "sha256", None) if lfs else None,
    }


def load_manifest(api: HfApi) -> tuple[dict[str, Any], str]:
    local_path = hf_hub_download(
        repo_id="SpeechAntiSpoofingBenchmarks/arena-manifest",
        repo_type="dataset",
        filename="manifest.yaml",
    )
    manifest = yaml.safe_load(Path(local_path).read_text())
    info = api.dataset_info("SpeechAntiSpoofingBenchmarks/arena-manifest")
    return manifest, info.sha


def index_dataset(api: HfApi, entry: dict[str, Any], hdd_root: Path) -> tuple[str, dict[str, Any]]:
    repo_id = entry["id"]
    revision = entry["revision"]
    info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    name = repo_id.rsplit("/", 1)[-1]
    siblings = [file_metadata(item) for item in info.siblings]
    return name, {
        "repo_id": repo_id,
        "revision": revision,
        "n_trials": entry["n_trials"],
        "category": entry.get("category", "binary"),
        "metric": entry.get("metric", "eer_percent"),
        "paper_url": entry.get("paper_url"),
        "local_dir": str(hdd_root / "datasets" / name / revision),
        "files": {
            "labels": "data/labels.parquet",
            "audio_glob": "data/test-*.parquet",
            "eval": "eval.yaml",
        },
        "pinned_files": [
            item for item in siblings if item["path"] in {"data/labels.parquet", "eval.yaml", "README.md", "LICENSE.txt"}
        ],
        "source_revision_resolved": info.sha,
    }


def index_model(api: HfApi, name: str, repo_id: str, datasets: dict[str, Any], hdd_root: Path) -> dict[str, Any]:
    info = api.model_info(repo_id, files_metadata=True)
    file_by_path = {item.rfilename: item for item in info.siblings}
    score_artifacts: dict[str, Any] = {}
    for dataset_name, dataset in datasets.items():
        dataset_id = dataset["repo_id"]
        prefix = f".eval_results/{dataset_id}/"
        score_path = f"{prefix}scores.txt"
        result_path = f"{prefix}result.yaml"
        if score_path in file_by_path:
            score_artifacts[dataset_name] = {
                "scores": file_metadata(file_by_path[score_path]),
                "result": file_metadata(file_by_path[result_path]) if result_path in file_by_path else None,
            }
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "local_dir": str(hdd_root / "models" / name / info.sha),
        "runtime_files": [
            file_metadata(file_by_path[path])
            for path in ("README.md", "meta.yaml", "config.json", "model.py", "spectra-aasist.onnx", "aasist.onnx", "rawtfnet.onnx")
            if path in file_by_path
        ],
        "score_artifacts": score_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/arena-index.yaml")
    parser.add_argument("--hdd-root", default="/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing")
    args = parser.parse_args()

    api = HfApi()
    manifest, manifest_sha = load_manifest(api)
    hdd_root = Path(args.hdd_root)
    selected_names = {"ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF", "InTheWild", "ASVspoof5", "DeepVoice", "LibriSeVoc", "XMAD", "CFAD", "CVoiceFake_small", "DECRO"}
    entries = [entry for entry in manifest["core_set"] if entry["id"].rsplit("/", 1)[-1] in selected_names]
    datasets = dict(index_dataset(api, entry, hdd_root) for entry in entries)
    models = {name: index_model(api, name, repo_id, datasets, hdd_root) for name, repo_id in MODEL_REPOS.items()}
    index = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "arena_manifest": {
            "repo_id": "SpeechAntiSpoofingBenchmarks/arena-manifest",
            "revision": manifest_sha,
            "ranking_version": manifest.get("ranking_version"),
            "ranking_metric": manifest["ranking"]["metric"],
            "source_manifest_revision": "manifest.yaml at current Arena repository head",
        },
        "datasets": datasets,
        "models": models,
        "join_policy": "Join only by exact utterance/sample identifier after documented normalization; never use ordinal row position or filename heuristics.",
        "score_policy": "Arena per-sample scores and published metrics are authoritative. This project does not rerun baseline inference to reproduce them.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(index, sort_keys=False, allow_unicode=True))
    print(f"wrote {output} with {len(datasets)} datasets and {len(models)} models")


if __name__ == "__main__":
    main()
