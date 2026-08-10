from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch import nn

from src.h9_pcr_training import (
    H9_LAMBDA_GRID,
    H9_SEEDS,
    H9PCRTrainer,
    TrainingConfig,
    build_random_pair_edges,
    load_source_manifest,
    pairwise_margin_loss,
    seed_to_device,
    select_p_lambda,
)


def _manifest_rows(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Two pair IDs per split make the B2 stratum-matched, content-unmatched
    # partner requirement testable for each of the two spoof generators.
    for split, pair_numbers in (("train", (0, 1)), ("dev", (2, 3))):
        for number in pair_numbers:
            speaker = f"speaker_{number}"
            pair_id = f"content_{number}"
            waveform_path = tmp_path / f"{pair_id}_bona.wav"
            sf.write(waveform_path, np.full(64, number + 1, dtype=np.float32), 16000)
            common = {
                "split": split,
                "pair_id": pair_id,
                "group_id": f"odss_a|{speaker}",
                "language": "en",
                "canonical_fingerprint": "",
                "source_corpus": "odss_a",
                "speaker_id": speaker,
            }
            rows.append(
                {
                    **common,
                    "sample_id": f"{pair_id}_bona",
                    "audio_path": str(waveform_path),
                    "label": 0,
                    "spoof_generator": "bonafide",
                }
            )
            for generator, offset in (("vits", 0.1), ("fastpitch", 0.2)):
                spoof_path = tmp_path / f"{pair_id}_{generator}.wav"
                sf.write(spoof_path, np.full(64, number + offset, dtype=np.float32), 16000)
                rows.append(
                    {
                        **common,
                        "sample_id": f"{pair_id}_{generator}",
                        "audio_path": str(spoof_path),
                        "label": 1,
                        "spoof_generator": generator,
                    }
                )
    return rows


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_source_manifest.csv"
    pd.DataFrame(_manifest_rows(tmp_path)).to_csv(path, index=False)
    return path


def test_source_manifest_requires_voice_disjoint_complete_pairs_and_filters_to_pairs(tmp_path: Path) -> None:
    manifest = load_source_manifest(_write_manifest(tmp_path))
    assert len(manifest.paired_eligible_ids["train"]) == 6
    assert len(manifest.paired_eligible_ids["dev"]) == 6
    assert len(manifest.pair_edges["train"]) == 4
    assert {edge.spoof_generator for edge in manifest.pair_edges["train"]} == {"vits", "fastpitch"}

    broken = pd.DataFrame(_manifest_rows(tmp_path))
    broken.loc[broken.index == 0, "split"] = "dev"
    broken_path = tmp_path / "voice_crosses_split.csv"
    broken.to_csv(broken_path, index=False)
    with pytest.raises(ValueError, match="voice group crosses"):
        load_source_manifest(broken_path)


def test_random_control_is_deterministic_stratified_and_content_unmatched(tmp_path: Path) -> None:
    manifest = load_source_manifest(_write_manifest(tmp_path))
    first, audit = build_random_pair_edges(manifest.pair_edges["train"], manifest.records, seed=9101, epoch=3)
    second, _ = build_random_pair_edges(manifest.pair_edges["train"], manifest.records, seed=9101, epoch=3)
    assert first == second
    assert len(first) == len(manifest.pair_edges["train"])
    assert audit["all_bona_are_content_unmatched"] is True
    for matched, random_edge in zip(manifest.pair_edges["train"], first, strict=True):
        assert random_edge.spoof_id == matched.spoof_id
        assert random_edge.bona_id != matched.bona_id
        assert manifest.records[random_edge.bona_id].label == 0
        assert random_edge.stratum == matched.stratum


def test_margin_and_seed_assignment_are_exact() -> None:
    loss = pairwise_margin_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.5]))
    assert float(loss) == pytest.approx((0.0 + 1.5) / 2.0)
    assert seed_to_device(9101) == "cuda:0"
    assert seed_to_device(9104) == "cuda:3"
    with pytest.raises(ValueError, match="frozen"):
        seed_to_device(9101, "cuda:1")


def test_p_lambda_selection_requires_complete_grid_and_lower_lambda_tie_break() -> None:
    rows = []
    for lam in H9_LAMBDA_GRID:
        for seed in H9_SEEDS:
            rows.append({"method": "P", "lambda_rank": lam, "seed": seed, "source_dev_eer": 0.2 if lam != 0.1 else 0.1})
    selection = select_p_lambda(rows)
    assert selection["selected_lambda_rank"] == 0.1
    assert selection["applies_unchanged_to"] == ["P", "B2"]
    with pytest.raises(ValueError, match="requires every"):
        select_p_lambda(rows[:-1])


class _TinyRes2Like(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = waveforms.mean(dim=1, keepdim=True)
        return pooled, self.linear(pooled)


def _tiny_factory(_bundle_dir: Path, *, seed: int, device: str) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    return _TinyRes2Like().to(device), {
        "initialization": "fresh_seeded",
        "initialization_seed": seed,
        "checkpoint_loaded": False,
        "architecture_sha256": "synthetic-test-only",
    }


def test_synthetic_bf16_p_training_step_is_finite_and_saves_best_source_checkpoint(tmp_path: Path) -> None:
    manifest = load_source_manifest(_write_manifest(tmp_path))
    config = TrainingConfig(
        method="P",
        seed=9101,
        device="cpu",
        batch_size=2,
        num_workers=0,
        max_epochs=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        lambda_rank=0.1,
        require_cuda=False,
    )
    result = H9PCRTrainer(manifest, config, bundle_dir=tmp_path, model_factory=_tiny_factory).fit(checkpoint_path=tmp_path / "synthetic.pt")
    assert result.best_epoch == 1
    assert np.isfinite(result.source_dev_eer)
    assert result.checkpoint_sha256 is not None
    assert result.architecture_provenance["checkpoint_loaded"] is False
    assert result.pairing_audit[0]["kind"] == "H9_P_MATCHED_PAIR_AUDIT"
