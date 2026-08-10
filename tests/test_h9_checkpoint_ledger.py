"""Synthetic-only tests for the H9 source checkpoint handoff builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.h9_pcr_checkpoint_ledger import build_frozen_checkpoint_ledger
from src.h9_pcr_evaluation import load_frozen_checkpoint_ledger
from src.h9_pcr_training import B2_PAIR_COLUMNS, H9_LAMBDA_GRID, H9_SEEDS, P_PAIR_COLUMNS
from src.res2tcn_pytorch import sha256_file


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _source_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    rows: list[dict[str, object]] = []
    p_rows: list[dict[str, str]] = []
    b2_rows: list[dict[str, str]] = []
    pairs_by_split: dict[str, list[dict[str, str]]] = {"train": [], "dev": []}
    for split in ("train", "dev"):
        for number in (1, 2):
            pair_id = f"{split}_pair_{number}"
            corpus, speaker = "synthetic", f"{split}_voice_{number}"
            voice_key, content_key = f"{corpus}/{speaker}", f"content_{pair_id}"
            group_key = f"{voice_key}::{content_key}"
            bona_id = f"bona_{pair_id}"
            rows.append(
                {
                    "sample_id": bona_id,
                    "audio_path": str(tmp_path / f"{bona_id}.wav"),
                    "label": 0,
                    "split": split,
                    "pair_id": pair_id,
                    "group_id": f"{corpus}|{speaker}",
                    "language": "en",
                    "canonical_fingerprint": _sha(bona_id),
                    "source_corpus": corpus,
                    "speaker_id": speaker,
                    "spoof_generator": "bonafide",
                }
            )
            pair = {
                "pair_id": pair_id,
                "split": split,
                "group_key": group_key,
                "voice_key": voice_key,
                "content_key": content_key,
                "source_corpus": corpus,
                "language": "en",
                "bona_utterance_id": bona_id,
                "bona_relative_path": f"{bona_id}.wav",
            }
            pairs_by_split[split].append(pair)
            for generator in ("vits", "fastpitch-hifigan"):
                spoof_id = f"{generator}_{pair_id}"
                rows.append(
                    {
                        "sample_id": spoof_id,
                        "audio_path": str(tmp_path / f"{spoof_id}.wav"),
                        "label": 1,
                        "split": split,
                        "pair_id": pair_id,
                        "group_id": f"{corpus}|{speaker}",
                        "language": "en",
                        "canonical_fingerprint": _sha(spoof_id),
                        "source_corpus": corpus,
                        "speaker_id": speaker,
                        "spoof_generator": generator,
                    }
                )
                p_rows.append(
                    {
                        **pair,
                        "spoof_utterance_id": spoof_id,
                        "spoof_relative_path": f"{spoof_id}.wav",
                        "spoof_generator": generator,
                    }
                )
    for split, pairs in pairs_by_split.items():
        for pair, alternate in ((pairs[0], pairs[1]), (pairs[1], pairs[0])):
            for generator in ("vits", "fastpitch-hifigan"):
                spoof_id = f"{generator}_{pair['pair_id']}"
                b2_rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "split": split,
                        "group_key": pair["group_key"],
                        "voice_key": pair["voice_key"],
                        "content_key": pair["content_key"],
                        "source_corpus": "synthetic",
                        "language": "en",
                        "spoof_generator": generator,
                        "random_bona_utterance_id": alternate["bona_utterance_id"],
                        "random_bona_relative_path": alternate["bona_relative_path"],
                        "random_bona_content_key": alternate["content_key"],
                        "random_bona_voice_key": alternate["voice_key"],
                        "random_bona_split": split,
                        "random_bona_language": "en",
                        "random_bona_source_corpus": "synthetic",
                        "spoof_utterance_id": spoof_id,
                        "spoof_relative_path": f"{spoof_id}.wav",
                    }
                )
    manifest = tmp_path / "source_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    p_pairs = tmp_path / "p_pairs.csv"
    pd.DataFrame(p_rows, columns=P_PAIR_COLUMNS).to_csv(p_pairs, index=False)
    b2_pairs = tmp_path / "b2_pairs.csv"
    pd.DataFrame(b2_rows, columns=B2_PAIR_COLUMNS).to_csv(b2_pairs, index=False)
    freeze = tmp_path / "source_freeze.json"
    _write_json(
        freeze,
        {
            "artifact_kind": "h9_odss_paired_counterfactual_source_freeze",
            "source_access": {"target_data_read": False, "target_label_read": False, "target_prediction_read": False},
            "outputs": {
                "pairs_csv": {"path": str(p_pairs), "sha256": sha256_file(p_pairs)},
                "b2_random_pairs_csv": {"path": str(b2_pairs), "sha256": sha256_file(b2_pairs)},
            },
        },
    )
    materialization = tmp_path / "materialization.json"
    _write_json(
        materialization,
        {
            "artifact_kind": "h9_odss_paired_counterfactual_source_materialization",
            "source_access": {"target_data_read": False, "target_label_read": False, "target_prediction_read": False},
            "outputs": {"source_manifest_csv": {"path": str(manifest), "sha256": sha256_file(manifest)}},
            "sealed_source_freeze": {
                "provenance_sha256": sha256_file(freeze),
                "artifact_sha256": {p_pairs.name: sha256_file(p_pairs), b2_pairs.name: sha256_file(b2_pairs)},
            },
            "trainer_manifest_validation": {"source_manifest_sha256": sha256_file(manifest)},
        },
    )
    return manifest, p_pairs, b2_pairs, freeze, materialization


def _selection(tmp_path: Path) -> Path:
    candidates = []
    for number, lam in enumerate(H9_LAMBDA_GRID, start=1):
        by_seed = {str(seed): 0.1 * number for seed in H9_SEEDS}
        candidates.append(
            {
                "lambda_rank": lam,
                "mean_source_dev_eer": 0.1 * number,
                "source_dev_eer_by_seed": by_seed,
            }
        )
    path = tmp_path / "selection.json"
    _write_json(
        path,
        {
            "kind": "H9_P_SOURCE_ONLY_LAMBDA_SELECTION",
            "selected_lambda_rank": 0.1,
            "selection_metric": "mean_source_dev_eer_over_four_predeclared_P_seeds",
            "tie_break": "lower_lambda_rank",
            "applies_unchanged_to": ["P", "B2"],
            "candidates": candidates,
            "target_labels_read": False,
            "target_audio_read": False,
        },
    )
    return path


def _final_sidecars(tmp_path: Path, hashes: dict[str, str], architecture_sha256: str) -> list[Path]:
    sidecars: list[Path] = []
    for method in ("B1", "B2", "P"):
        for index, seed in enumerate(H9_SEEDS):
            checkpoint = tmp_path / f"{method}_{seed}.pt"
            lambda_rank = 0.0 if method == "B1" else 0.1
            torch.save(
                {
                    "model_state_dict": {"synthetic": torch.tensor([index])},
                    "training_config": {
                        "method": method,
                        "seed": seed,
                        "device": f"cuda:{index}",
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
                    "p_pair_csv_sha256": hashes["p_pairs_sha256"],
                    "b2_pair_csv_sha256": hashes["b2_pairs_sha256"] if method == "B2" else None,
                    "architecture_provenance": {
                        "architecture_sha256": architecture_sha256,
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
                    "device": f"cuda:{index}",
                    "lambda_rank": lambda_rank,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "target_labels_read": False,
                    "target_audio_read": False,
                    "precision": "cuda_bfloat16_autocast",
                    "architecture_provenance": {
                        "architecture_sha256": architecture_sha256,
                        "initialization": "fresh_seeded",
                        "initialization_seed": seed,
                        "checkpoint_loaded": False,
                    },
                    "source_artifact_hashes": hashes,
                },
            )
            sidecars.append(sidecar)
    return sidecars


def _inputs(tmp_path: Path) -> dict[str, object]:
    manifest, p_pairs, b2_pairs, freeze, materialization = _source_artifacts(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    architecture = bundle / "_net.py"
    architecture.write_text("# synthetic architecture\n", encoding="utf-8")
    hashes = {
        "source_manifest_sha256": sha256_file(manifest),
        "p_pairs_sha256": sha256_file(p_pairs),
        "b2_pairs_sha256": sha256_file(b2_pairs),
    }
    return {
        "source_manifest": manifest,
        "p_pairs": p_pairs,
        "b2_pairs": b2_pairs,
        "source_freeze_provenance": freeze,
        "source_materialization_provenance": materialization,
        "source_selection": _selection(tmp_path),
        "architecture_bundle": bundle,
        "training_records": _final_sidecars(tmp_path, hashes, sha256_file(architecture)),
        "plan": Path("experiments/h9_paired_counterfactual/PLAN.md").resolve(),
        "data_contract": Path("experiments/h9_paired_counterfactual/DATA.md").resolve(),
    }


def test_checkpoint_ledger_builder_creates_exact_terminal_contract_once(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "frozen_checkpoint_ledger.json"
    written = build_frozen_checkpoint_ledger(**inputs, output=output)
    assert written == output
    loaded = load_frozen_checkpoint_ledger(output, plan_path=inputs["plan"], data_contract_path=inputs["data_contract"])
    assert len(loaded.checkpoints) == 12
    assert loaded.selected_lambda_rank == 0.1
    with pytest.raises(FileExistsError, match="already exists"):
        build_frozen_checkpoint_ledger(**inputs, output=output)


def test_checkpoint_ledger_builder_rejects_target_firewall_breach_before_publication(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    breached = Path(inputs["training_records"][0])
    payload = json.loads(breached.read_text(encoding="utf-8"))
    payload["target_audio_read"] = True
    _write_json(breached, payload)
    output = tmp_path / "should_not_exist.json"
    with pytest.raises(ValueError, match="target firewall"):
        build_frozen_checkpoint_ledger(**inputs, output=output)
    assert not output.exists()
