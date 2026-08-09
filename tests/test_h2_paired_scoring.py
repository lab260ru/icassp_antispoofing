from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from src.arena_io import AudioRecord, decode_audio
from src.h2_paired_scoring import (
    SCORER_SPECS,
    run_paired_scoring,
    score_run_paths,
    validate_pinned_runtime_assignment,
    validate_score_eligibility,
)
from src.h2_pre_score_pairs import apply_registered_arm, arm_definitions_frame, freeze_input_manifest, waveform_sha256
from src.h2_quality_runner import pair_id_for, validate_frozen_inputs


SAMPLE_RATE = 16_000


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _audio_bytes() -> bytes:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    waveform = (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    waveform[1_000] = 0.5
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _write_pre_score_inputs(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    source = pd.DataFrame(
        [{"dataset": "toy", "sample_id": "clip_a", "source_id": "source_a", "label": 0, "dataset_revision": "rev-toy"}]
    )
    manifest = freeze_input_manifest(source, per_label=1, seed=2609, input_csv_sha256="a" * 64).rows
    manifest_path = root / "input_manifest.csv"
    ledger_path = root / "arm_ledger.json"
    index_path = root / "arena-index.yaml"
    manifest.to_csv(manifest_path, index=False)
    ledger_path.write_text(json.dumps(arm_definitions_frame().to_dict(orient="records"), indent=2), encoding="utf-8")
    index_path.write_text(yaml.safe_dump({"datasets": {"toy": {"repo_id": "example/toy", "revision": "rev-toy"}}}), encoding="utf-8")
    return manifest_path, ledger_path, index_path, manifest.iloc[0].to_dict()


def _write_quality_freeze(root: Path, *, transformed_hash: str | None = None) -> tuple[Path, Path, Path, Path, Path, bytes]:
    manifest_path, ledger_path, index_path, source = _write_pre_score_inputs(root)
    frozen = validate_frozen_inputs(manifest_path, ledger_path, index_path)
    arm = next(item for item in frozen.arms if item.arm_id == "polarity")
    audio_bytes = _audio_bytes()
    original, sample_rate = decode_audio(audio_bytes)
    transformed = apply_registered_arm(original, sample_rate, arm).waveform
    pair_id = pair_id_for(source, arm)
    row = {
        "quality_row_version": "h2_quality_row_v1",
        "pair_id": pair_id,
        "dataset": source["dataset"],
        "sample_id": source["sample_id"],
        "label": source["label"],
        "source_id": source["source_id"],
        "selection_key_sha256": source["selection_key_sha256"],
        "arm": arm.arm_id,
        "arm_definition_sha256": arm.as_record()["definition_sha256"],
        "detector_stage": "eligible_after_quality_freeze",
        "retained": True,
        "quality_status": "passed",
        "failure_reasons_json": "[]",
        "sample_rate_hz": sample_rate,
        "original_samples": len(original),
        "transformed_samples": len(transformed),
        "original_wave_sha256": waveform_sha256(original),
        "transformed_wave_sha256": transformed_hash or waveform_sha256(transformed),
    }
    quality_path = root / "score_eligible_pairs.csv"
    pd.DataFrame([row]).to_csv(quality_path, index=False)
    details = {
        arm_id: {
            "gate_status": "passed",
            "retained_fraction": 0.9,
        }
        for arm_id in (item.arm_id for item in frozen.arms)
    }
    report = {
        "artifact_kind": "h2_score_eligible_quality_freeze",
        "version": "h2_quality_freeze_v1",
        "freeze_status": "score_eligible",
        "detector_scoring_allowed": True,
        "detector_scores_read": False,
        "detector_scoring_performed": False,
        "quality_run_id": "quality_full_001",
        "input_manifest_sha256": frozen.manifest_sha256,
        "arm_ledger_sha256": frozen.arm_ledger_sha256,
        "arena_index_sha256": frozen.arena_index_sha256,
        "quality_provenance_sha256": _sha256("provenance"),
        "quality_table_sha256": _sha256("table"),
        "quality_checkpoint_aggregate_sha256": _sha256("checkpoints"),
        "predeclared_minimum_retained_fraction": 0.9,
        "score_eligible_pairs": 1,
        "score_eligible_manifest_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        "per_arm": details,
    }
    freeze_path = root / "quality_freeze.json"
    freeze_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return quality_path, freeze_path, manifest_path, ledger_path, index_path, audio_bytes


def _iterator(_dataset: dict[str, object], selected: pd.DataFrame, audio_bytes: bytes):
    for row in selected.to_dict(orient="records"):
        yield AudioRecord(
            sample_id=str(row["sample_id"]),
            source_id=str(row["source_id"]),
            label=int(row["label"]),
            audio_bytes=audio_bytes,
            notes={},
        )


@dataclass
class FakeScorer:
    calls: int = 0

    @property
    def provenance(self) -> dict[str, str]:
        return {"runner": "fake"}

    def score(self, waveforms: list[np.ndarray]) -> SimpleNamespace:
        self.calls += 1
        detector = np.stack([np.asarray(item, dtype=np.float32) for item in waveforms])
        raw = detector.mean(axis=1).astype(np.float32)
        logits = np.stack((-raw, raw), axis=1)
        return SimpleNamespace(raw_score=raw, logits=logits, detector_waveforms=detector)


def test_quality_freeze_hash_and_regeneration_gate_before_any_scorer(tmp_path: Path) -> None:
    quality, freeze, manifest, ledger, index, audio = _write_quality_freeze(tmp_path)
    eligibility = validate_score_eligibility(
        quality_manifest_path=quality,
        quality_freeze_path=freeze,
        input_manifest_path=manifest,
        arm_ledger_path=ledger,
        arena_index_path=index,
    )
    scorer = FakeScorer()
    paths = score_run_paths(tmp_path / "hdd", "score_001", "Res2TCNGuard")
    summary = run_paired_scoring(
        eligibility,
        paths=paths,
        run_id="score_001",
        spec=SCORER_SPECS["Res2TCNGuard"],
        parity={"test": "parity"},
        model_directory=tmp_path / "model",
        scorer_factory=lambda _spec, _directory: scorer,
        audio_iterator=lambda dataset, selected: _iterator(dataset, selected, audio),
    )
    assert scorer.calls == 1
    assert summary["completed_pairs"] == 1
    table = pd.read_parquet(paths.scores_path)
    assert table.loc[0, "score_delta_spoof"] != pytest.approx(0.0)
    assert table.loc[0, "quality_manifest_sha256"] == hashlib.sha256(quality.read_bytes()).hexdigest()

    # A matching resume validates the prior immutable segment and never calls a scorer again.
    resumed = run_paired_scoring(
        eligibility,
        paths=paths,
        run_id="score_001",
        spec=SCORER_SPECS["Res2TCNGuard"],
        parity={"test": "parity"},
        model_directory=tmp_path / "model",
        scorer_factory=lambda _spec, _directory: (_ for _ in ()).throw(AssertionError("scorer must not run on resume")),
        audio_iterator=lambda dataset, selected: _iterator(dataset, selected, audio),
    )
    assert resumed["newly_written_pairs"] == 0

    # Editing the CSV after the quality freeze is rejected before a scorer can be built.
    changed = pd.read_csv(quality)
    changed.loc[0, "sample_id"] = "tampered"
    changed.to_csv(quality, index=False)
    with pytest.raises(ValueError, match="score_eligible_manifest_sha256"):
        validate_score_eligibility(
            quality_manifest_path=quality,
            quality_freeze_path=freeze,
            input_manifest_path=manifest,
            arm_ledger_path=ledger,
            arena_index_path=index,
        )


def test_regenerated_transformed_hash_mismatch_prevents_scoring(tmp_path: Path) -> None:
    quality, freeze, manifest, ledger, index, audio = _write_quality_freeze(tmp_path, transformed_hash="a" * 64)
    eligibility = validate_score_eligibility(
        quality_manifest_path=quality,
        quality_freeze_path=freeze,
        input_manifest_path=manifest,
        arm_ledger_path=ledger,
        arena_index_path=index,
    )
    with pytest.raises(RuntimeError, match="transformed waveform hash"):
        run_paired_scoring(
            eligibility,
            paths=score_run_paths(tmp_path / "hdd", "score_002", "Res2TCNGuard"),
            run_id="score_002",
            spec=SCORER_SPECS["Res2TCNGuard"],
            parity={"test": "parity"},
            model_directory=tmp_path / "model",
            scorer_factory=lambda _spec, _directory: (_ for _ in ()).throw(AssertionError("must not build scorer")),
            audio_iterator=lambda dataset, selected: _iterator(dataset, selected, audio),
        )


def test_runtime_assignment_requires_the_pinned_single_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SCORER_SPECS["AASIST"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="physical GPU 1"):
        validate_pinned_runtime_assignment(spec)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    validate_pinned_runtime_assignment(spec)
