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
    load_frozen_b2_pairs,
    load_frozen_p_pairs,
    load_source_manifest,
    pairwise_margin_loss,
    seed_to_device,
    select_p_lambda,
)
from src.res2tcn_pytorch import sha256_file


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


def _write_frozen_pair_csvs(tmp_path: Path, manifest_path: Path) -> tuple[Path, Path]:
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    p_rows: list[dict[str, str]] = []
    b2_rows: list[dict[str, str]] = []
    for split in ("train", "dev"):
        split_frame = frame.loc[frame["split"].eq(split)]
        pair_ids = sorted(split_frame["pair_id"].unique())
        for position, pair_id in enumerate(pair_ids):
            rows = split_frame.loc[split_frame["pair_id"].eq(pair_id)]
            bona = rows.loc[rows["label"].eq("0")].iloc[0]
            other_pair = pair_ids[(position + 1) % len(pair_ids)]
            random_bona = split_frame.loc[(split_frame["pair_id"].eq(other_pair)) & (split_frame["label"].eq("0"))].iloc[0]
            for spoof in rows.loc[rows["label"].eq("1")].itertuples(index=False):
                p_rows.append(
                    {
                        "pair_id": str(pair_id),
                        "split": split,
                        "group_key": str(spoof.group_id),
                        "voice_key": str(spoof.group_id),
                        "content_key": str(pair_id),
                        "source_corpus": str(spoof.source_corpus),
                        "language": str(spoof.language),
                        "bona_utterance_id": str(bona.sample_id),
                        "bona_relative_path": Path(str(bona.audio_path)).name,
                        "spoof_utterance_id": str(spoof.sample_id),
                        "spoof_relative_path": Path(str(spoof.audio_path)).name,
                        "spoof_generator": str(spoof.spoof_generator),
                    }
                )
                b2_rows.append(
                    {
                        "pair_id": str(pair_id),
                        "split": split,
                        "group_key": str(spoof.group_id),
                        "voice_key": str(spoof.group_id),
                        "content_key": str(pair_id),
                        "source_corpus": str(spoof.source_corpus),
                        "language": str(spoof.language),
                        "spoof_generator": str(spoof.spoof_generator),
                        "random_bona_utterance_id": str(random_bona.sample_id),
                        "random_bona_relative_path": Path(str(random_bona.audio_path)).name,
                        "random_bona_content_key": str(other_pair),
                        "random_bona_voice_key": str(random_bona.group_id),
                        "random_bona_split": split,
                        "random_bona_language": str(random_bona.language),
                        "random_bona_source_corpus": str(random_bona.source_corpus),
                        "spoof_utterance_id": str(spoof.sample_id),
                        "spoof_relative_path": Path(str(spoof.audio_path)).name,
                    }
                )
    p_path = tmp_path / "h9_odss_source_pairs.csv"
    b2_path = tmp_path / "h9_odss_b2_random_pairs.csv"
    pd.DataFrame(p_rows).to_csv(p_path, index=False)
    pd.DataFrame(b2_rows).to_csv(b2_path, index=False)
    return p_path, b2_path


def _load_frozen_artifacts(tmp_path: Path) -> tuple[object, object, object]:
    manifest_path = _write_manifest(tmp_path)
    manifest = load_source_manifest(manifest_path)
    p_path, b2_path = _write_frozen_pair_csvs(tmp_path, manifest_path)
    p_pairs = load_frozen_p_pairs(p_path, manifest, expected_sha256=sha256_file(p_path))
    b2_pairs = load_frozen_b2_pairs(b2_path, manifest, p_pairs, expected_sha256=sha256_file(b2_path))
    return manifest, p_pairs, b2_pairs


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


def test_frozen_pair_artifacts_are_hash_validated_stratified_and_content_unmatched(tmp_path: Path) -> None:
    manifest, p_pairs, b2_pairs = _load_frozen_artifacts(tmp_path)
    assert p_pairs.pairing_audit["all_manifest_matched_edges_present_exactly_once"] is True
    assert b2_pairs.pairing_audit["all_bona_are_content_unmatched"] is True
    for p_edge, b2_edge in zip(p_pairs.edges_by_split["train"], b2_pairs.edges_by_split["train"], strict=True):
        assert b2_edge.spoof_id == p_edge.spoof_id
        assert b2_edge.pair_id == p_edge.pair_id
        assert b2_edge.bona_id != p_edge.bona_id
        assert manifest.records[b2_edge.bona_id].label == 0
        assert b2_edge.stratum == p_edge.stratum

    p_path = Path(p_pairs.path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_p_pairs(p_path, manifest, expected_sha256="0" * 64)

    b2_frame = pd.read_csv(b2_pairs.path, dtype=str, keep_default_na=False)
    b2_frame.loc[0, "random_bona_content_key"] = b2_frame.loc[0, "content_key"]
    invalid_path = tmp_path / "b2_same_content.csv"
    b2_frame.to_csv(invalid_path, index=False)
    with pytest.raises(ValueError, match="random_bona_content_key"):
        load_frozen_b2_pairs(invalid_path, manifest, p_pairs)


def test_margin_and_seed_assignment_are_exact() -> None:
    loss = pairwise_margin_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.5]))
    assert float(loss) == pytest.approx((0.0 + 1.5) / 2.0)
    assert seed_to_device(9101) == "cuda:0"
    assert seed_to_device(9104) == "cuda:3"
    with pytest.raises(ValueError, match="frozen"):
        seed_to_device(9101, "cuda:1")


def test_b2_rank_loader_consumes_the_frozen_partner_list_without_epoch_remapping(tmp_path: Path) -> None:
    manifest, p_pairs, b2_pairs = _load_frozen_artifacts(tmp_path)
    config = TrainingConfig(
        method="B2",
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
    trainer = H9PCRTrainer(manifest, config, bundle_dir=tmp_path, p_pairs=p_pairs, b2_pairs=b2_pairs, model_factory=_tiny_factory)
    scheduled = trainer._rank_loader(4).dataset.edges
    assert scheduled == b2_pairs.edges_by_split["train"]
    assert trainer._rank_loader(4).dataset.edges == scheduled
    assert b2_pairs.pairing_audit["strata"]["en|odss_a|vits"]["eligible_p_bona_count"] == 4


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
    manifest, p_pairs, _ = _load_frozen_artifacts(tmp_path)
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
    trainer = H9PCRTrainer(manifest, config, bundle_dir=tmp_path, p_pairs=p_pairs, model_factory=_tiny_factory)
    # P consumes the exact same frozen list when each epoch's common batch
    # schedule has spare rank slots; its content partners are never remapped.
    assert trainer._rank_loader(4).dataset.edges == trainer._rank_loader(4).dataset.edges
    result = trainer.fit(checkpoint_path=tmp_path / "synthetic.pt")
    assert result.best_epoch == 1
    assert np.isfinite(result.source_dev_eer)
    assert result.checkpoint_sha256 is not None
    assert result.architecture_provenance["checkpoint_loaded"] is False
    assert result.pairing_audit[0]["kind"] == "H9_P_FROZEN_MATCHED_PAIR_AUDIT"
    assert result.p_pair_csv_sha256 == p_pairs.sha256
