"""Synthetic-only contract tests for the sealed H9-PCR terminal evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch import nn

from src.h9_pcr_evaluation import (
    H9_EVALUATION_VERSION,
    METHODS,
    TARGET_REVISIONS,
    bootstrap_macro_eer_differences,
    load_frozen_checkpoint_ledger,
    terminal_evaluate,
)
from src.h9_pcr_training import H9_SEEDS
from src.res2tcn_pytorch import sha256_file


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _TinyTerminalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = waveforms.mean(dim=1, keepdim=True)
        return pooled, self.linear(pooled)


def _tiny_factory(_bundle_dir: Path, *, seed: int, device: str) -> tuple[nn.Module, dict[str, object]]:
    return _TinyTerminalModel().to(device), {
        "architecture_sha256": _sha(b"synthetic-architecture"),
        "initialization": "fresh_seeded",
        "initialization_seed": seed,
        "checkpoint_loaded": False,
    }


def _waveform_loader(path: str) -> np.ndarray:
    # The synthetic waveform is determined from its filename, never from the
    # separately sealed target label artifact.
    return np.full(12, float(Path(path).stem.rsplit("_", 1)[1]), dtype=np.float32)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _target_manifest(tmp_path: Path, dataset: str, labels: list[int]) -> Path:
    records: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        audio = tmp_path / f"{dataset}_audio_{index}.wav"
        sf.write(audio, np.full(16, float(index), dtype=np.float32), 16000)
        records.append(
            {
                "sample_id": f"{dataset}_{index}",
                "audio_path": str(audio),
                "audio_sha256": sha256_file(audio),
                "audio_bytes": audio.stat().st_size,
                "canonical_fingerprint": _sha(f"{dataset}-fp-{index}".encode()),
            }
        )
        label_rows.append({"sample_id": f"{dataset}_{index}", "label": label})
    records_path = tmp_path / f"{dataset}_records.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)
    labels_path = tmp_path / f"{dataset}_labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    manifest_path = tmp_path / f"{dataset}_manifest.json"
    _write_json(
        manifest_path,
        {
            "artifact_kind": "h9_pcr_canonical_target_manifest",
            "version": H9_EVALUATION_VERSION,
            "dataset": dataset,
            "dataset_revision": TARGET_REVISIONS[dataset],
            "records_path": str(records_path),
            "records_sha256": sha256_file(records_path),
            "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            "target_labels_read": False,
            "target_metrics_read": False,
        },
    )
    return manifest_path


def _source_binding(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    source_path = tmp_path / "source_manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["source_a", "source_b"],
            "canonical_fingerprint": [_sha(b"source-a"), _sha(b"source-b")],
        }
    ).to_csv(source_path, index=False)
    p_pairs = tmp_path / "p_pairs.csv"
    b2_pairs = tmp_path / "b2_pairs.csv"
    pd.DataFrame({"edge": ["p"]}).to_csv(p_pairs, index=False)
    pd.DataFrame({"edge": ["b2"]}).to_csv(b2_pairs, index=False)
    return source_path, p_pairs, b2_pairs, {
        "source_manifest_sha256": sha256_file(source_path),
        "p_pairs_sha256": sha256_file(p_pairs),
        "b2_pairs_sha256": sha256_file(b2_pairs),
    }


def _ledger(tmp_path: Path, *, plan: Path, data: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    architecture = bundle / "_net.py"
    architecture.write_text("# synthetic architecture provenance only\n", encoding="utf-8")
    architecture_sha = _sha(b"synthetic-architecture")
    # Keep the byte-bound file's hash identical to the tiny factory's declared
    # synthetic provenance only through an explicit monkey-free file payload.
    architecture.write_bytes(b"synthetic-architecture")
    source, p_pairs, b2_pairs, hashes = _source_binding(tmp_path)
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "selected_lambda_rank": 0.1,
            "target_labels_read": False,
            "target_audio_read": False,
        },
    )
    runs: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        for seed in H9_SEEDS:
            model = _TinyTerminalModel()
            with torch.no_grad():
                model.linear.weight.copy_(torch.tensor([[0.0], [1.0 + method_index * 0.1]], dtype=torch.float32))
                model.linear.bias.copy_(torch.tensor([0.0, -0.25 * method_index], dtype=torch.float32))
            checkpoint = tmp_path / f"{method}_{seed}.pt"
            lambda_rank = 0.0 if method == "B1" else 0.1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "training_config": {
                        "method": method,
                        "seed": seed,
                        "device": f"cuda:{H9_SEEDS.index(seed)}",
                        "batch_size": 24,
                        "num_workers": 4,
                        "max_epochs": 6,
                        "learning_rate": 1e-4,
                        "weight_decay": 1e-2,
                        "lambda_rank": lambda_rank,
                        "margin": 1.0,
                        "require_cuda": True,
                    },
                    "source_manifest_sha256": hashes["source_manifest_sha256"],
                    "architecture_provenance": {
                        "architecture_sha256": architecture_sha,
                        "initialization": "fresh_seeded",
                        "initialization_seed": seed,
                        "checkpoint_loaded": False,
                    },
                    "selection_rule": "lowest_source_dev_eer_then_lower_epoch",
                    "best_epoch": 1,
                    "source_dev_eer": 0.5,
                },
                checkpoint,
            )
            sidecar = tmp_path / f"{method}_{seed}.json"
            _write_json(
                sidecar,
                {
                    "method": method,
                    "seed": seed,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "target_labels_read": False,
                    "target_audio_read": False,
                    "architecture_provenance": {
                        "architecture_sha256": architecture_sha,
                        "initialization": "fresh_seeded",
                        "initialization_seed": seed,
                        "checkpoint_loaded": False,
                    },
                    "source_artifact_hashes": hashes,
                },
            )
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "training_record_path": str(sidecar),
                    "training_record_sha256": sha256_file(sidecar),
                    "source_artifact_hashes": hashes,
                }
            )
    ledger = tmp_path / "ledger.json"
    _write_json(
        ledger,
        {
            "artifact_kind": "h9_pcr_frozen_checkpoint_ledger",
            "version": H9_EVALUATION_VERSION,
            "target_labels_read": False,
            "target_audio_read": False,
            "target_metrics_read": False,
            "protocol": {"plan_sha256": sha256_file(plan), "data_contract_sha256": sha256_file(data)},
            "source_artifacts": {
                "source_manifest_path": str(source),
                "source_manifest_sha256": hashes["source_manifest_sha256"],
                "p_pairs_path": str(p_pairs),
                "p_pairs_sha256": hashes["p_pairs_sha256"],
                "b2_pairs_path": str(b2_pairs),
                "b2_pairs_sha256": hashes["b2_pairs_sha256"],
            },
            "architecture": {"bundle_dir": str(bundle), "architecture_sha256": architecture_sha},
            "source_selection": {"selected_lambda_rank": 0.1, "path": str(selection), "sha256": sha256_file(selection)},
            "training_envelope": {
                "batch_size": 24,
                "num_workers": 4,
                "max_epochs": 6,
                "learning_rate": 1e-4,
                "weight_decay": 1e-2,
                "margin": 1.0,
                "require_cuda": True,
            },
            "checkpoints": runs,
        },
    )
    return ledger


def test_terminal_switch_refuses_before_any_target_manifest_is_read(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="terminal-evaluation"):
        terminal_evaluate(
            sonar_manifest_path=tmp_path / "not_read.json",
            arad_manifest_path=tmp_path / "not_read.json",
            checkpoint_ledger_path=tmp_path / "not_read.json",
            plan_path=tmp_path / "not_read.md",
            data_contract_path=tmp_path / "not_read.md",
            output_dir=tmp_path / "out",
            device="cpu",
            terminal_evaluation=False,
        )


def test_checkpoint_ledger_requires_complete_unique_method_seed_matrix(tmp_path: Path) -> None:
    plan = Path("experiments/h9_paired_counterfactual/PLAN.md").resolve()
    data = Path("experiments/h9_paired_counterfactual/DATA.md").resolve()
    ledger_path = _ledger(tmp_path, plan=plan, data=data)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["checkpoints"].pop()
    _write_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="exactly all B1/B2/P"):
        load_frozen_checkpoint_ledger(ledger_path, plan_path=plan, data_contract_path=data)


def test_synthetic_terminal_evaluator_writes_all_frozen_outputs(tmp_path: Path) -> None:
    plan = Path("experiments/h9_paired_counterfactual/PLAN.md").resolve()
    data = Path("experiments/h9_paired_counterfactual/DATA.md").resolve()
    sonar = _target_manifest(tmp_path, "SONAR", [0, 0, 1, 1])
    arad = _target_manifest(tmp_path, "ArAD", [0, 0, 1, 1])
    ledger = _ledger(tmp_path, plan=plan, data=data)
    outputs = terminal_evaluate(
        sonar_manifest_path=sonar,
        arad_manifest_path=arad,
        checkpoint_ledger_path=ledger,
        plan_path=plan,
        data_contract_path=data,
        output_dir=tmp_path / "terminal-output",
        device="cpu",
        terminal_evaluation=True,
        synthetic_test_mode=True,
        waveform_loader=_waveform_loader,
        model_factory=_tiny_factory,
    )
    raw = pd.read_parquet(outputs["raw_predictions"])
    metrics = pd.read_csv(outputs["metrics"])
    seed_metrics = pd.read_csv(outputs["seed_metrics"])
    bootstrap = pd.read_csv(outputs["bootstrap"])
    assert len(raw) == 2 * 4 * len(METHODS) * len(H9_SEEDS)
    assert "label" not in raw.columns
    assert len(metrics) == 2 * len(METHODS)
    assert len(seed_metrics) == 2 * len(METHODS) * len(H9_SEEDS)
    assert len(bootstrap) == 2_000
    provenance = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
    assert provenance["target_labels_read"] is True
    assert provenance["source_target_canonical_fingerprint_collision_count"] == 0


def test_bootstrap_is_fixed_and_shared_for_both_comparators() -> None:
    rows = []
    for dataset in TARGET_REVISIONS:
        for label in (0, 1):
            for index in range(3):
                rows.append({"dataset": dataset, "sample_id": f"{dataset}-{label}-{index}", "label": label, "P": float(label), "B1": float(1 - label), "B2": float(1 - label)})
    first = bootstrap_macro_eer_differences(pd.DataFrame(rows))
    second = bootstrap_macro_eer_differences(pd.DataFrame(rows))
    assert first.equals(second)
    assert len(first) == 2_000
