"""Synthetic-only contracts for the H10 CD-ADD adapter and terminal call.

These fixtures deliberately construct a temporary revision-shaped directory.
They never resolve, list, or open the real CD-ADD cache.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from src.h10_cdadd_evaluation import (
    H10TargetManifest,
    bootstrap_eer_differences,
    load_h10_cdadd_target_manifest,
    terminal_evaluate_h10_cdadd,
)
from src.h10_cdadd_target_materialize import (
    H10_CDADD_CARD_URL,
    H10_CDADD_DATASET,
    H10_CDADD_LICENSE,
    H10_CDADD_REPOSITORY,
    H10_CDADD_REVISION,
    TARGET_AUDIO_AUDIT_COLUMNS,
    TARGET_LABEL_COLUMNS,
    TARGET_RECORD_COLUMNS,
    materialize_h10_cdadd_target,
)
from src.h9_pcr_evaluation import FrozenCheckpoint, FrozenCheckpointLedger, METHODS
from src.h9_pcr_training import H9_SEEDS
from src.res2tcn_pytorch import sha256_file


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _audio_bytes(value: float, *, format: str = "FLAC") -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.full(64, value, dtype=np.float32), 16_000, format=format, subtype="PCM_16")
    return stream.getvalue()


def _synthetic_raw_target(tmp_path: Path, *, labels: list[int] | None = None) -> Path:
    values = [0, 0, 1, 1] if labels is None else labels
    raw = tmp_path / "datasets" / "CD-ADD" / H10_CDADD_REVISION / "data"
    raw.mkdir(parents=True)
    rows = [
        {
            "path": f"synthetic/cdadd_{index}.flac",
            "audio": {"bytes": _audio_bytes(float(index) / 8.0), "path": None},
            "label": label,
        }
        for index, label in enumerate(values)
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw / "test-00000.parquet")
    # The conventional standalone index is not an audio-bearing trial shard.
    # The adapter must not mistake it for one and must keep its own two-pass
    # label artifact separate from this raw cache convenience file.
    pq.write_table(
        pa.Table.from_pydict({"path": [str(row["path"]) for row in rows], "label": values}),
        raw / "labels.parquet",
    )
    return raw


def _metadata_audit(tmp_path: Path) -> Path:
    path = tmp_path / "metadata_audit.json"
    _write_json(
        path,
        {
            "artifact_kind": "h10_cdadd_target_metadata_audit",
            "repository": H10_CDADD_REPOSITORY,
            "revision": H10_CDADD_REVISION,
            "license": H10_CDADD_LICENSE,
            "card_url": H10_CDADD_CARD_URL,
            "card_sha256": _sha(b"synthetic pinned public card"),
            "target_rows_read": False,
            "target_audio_read": False,
            "target_labels_read": False,
            "target_scores_read": False,
            "target_predictions_read": False,
            "odss_overlap_audit": {
                "known_recording_source_overlap": False,
                "known_speaker_list_overlap": False,
                "known_released_item_overlap": False,
            },
        },
    )
    return path


def _source_ledger(tmp_path: Path) -> tuple[Path, FrozenCheckpointLedger]:
    source = tmp_path / "source_manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["source-a", "source-b"],
            "canonical_fingerprint": [_sha(b"source-a"), _sha(b"source-b")],
        }
    ).to_csv(source, index=False)
    pairs = tmp_path / "p_pairs.csv"
    b2_pairs = tmp_path / "b2_pairs.csv"
    pairs.write_text("edge\np\n", encoding="utf-8")
    b2_pairs.write_text("edge\nb2\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    architecture = bundle / "_net.py"
    architecture.write_bytes(b"synthetic architecture")
    ledger_path = tmp_path / "synthetic_ledger.json"
    ledger_path.write_bytes(b"synthetic H9 checkpoint ledger only")
    checkpoints = {
        (method, seed): FrozenCheckpoint(
            method,
            seed,
            tmp_path / f"{method}_{seed}.pt",
            _sha(f"{method}-{seed}".encode()),
            {},
        )
        for method in METHODS
        for seed in H9_SEEDS
    }
    ledger = FrozenCheckpointLedger(
        ledger_path=ledger_path,
        ledger_sha256=sha256_file(ledger_path),
        source_manifest_path=source,
        source_manifest_sha256=sha256_file(source),
        source_p_pairs_path=pairs,
        source_p_pairs_sha256=sha256_file(pairs),
        source_b2_pairs_path=b2_pairs,
        source_b2_pairs_sha256=sha256_file(b2_pairs),
        architecture_bundle=bundle,
        architecture_sha256=sha256_file(architecture),
        selected_lambda_rank=1.0,
        checkpoints=checkpoints,
    )
    return ledger_path, ledger


def _materialize(tmp_path: Path) -> tuple[Path, Path, FrozenCheckpointLedger]:
    raw = _synthetic_raw_target(tmp_path)
    audit = _metadata_audit(tmp_path)
    ledger_path, ledger = _source_ledger(tmp_path)
    hdd = tmp_path / "hdd"
    hdd.mkdir()
    result = materialize_h10_cdadd_target(
        raw_shard_dir=raw,
        checkpoint_ledger=ledger_path,
        metadata_audit=audit,
        output_dir=hdd / "materialization",
        allowed_output_root=hdd,
        _source_ledger_validator=lambda _path: ledger,
    )
    return result.manifest, ledger_path, ledger


def test_h10_materializer_writes_two_separate_synthetic_artifacts(tmp_path: Path) -> None:
    manifest_path, _, _ = _materialize(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = pd.read_csv(manifest["records_path"])
    labels = pd.read_csv(manifest["labels"]["path"])
    provenance = json.loads(Path(manifest["materialization_provenance"]["path"]).read_text(encoding="utf-8"))
    audit = pd.read_csv(provenance["outputs"]["audio_audit_csv"]["path"])
    assert tuple(records.columns) == TARGET_RECORD_COLUMNS
    assert tuple(labels.columns) == TARGET_LABEL_COLUMNS
    assert tuple(audit.columns) == TARGET_AUDIO_AUDIT_COLUMNS
    assert "label" not in records.columns
    assert len(records) == len(labels) == len(audit) == 4
    assert records["sample_id"].tolist() == labels["sample_id"].tolist()
    assert set(labels["label"]) == {0, 1}
    assert manifest["target_labels_read"] is False
    assert set(manifest["locked_protocols"]) == {"h10_protocol", "h9_plan", "h9_data"}
    assert manifest["target_construction"] == {
        "all_audio_bearing_parquet_trials": True,
        "target_subset": None,
        "generator_stratum": None,
        "path_based_exclusions": None,
        "raw_shard_count": 1,
        "raw_trial_count": 4,
    }
    assert provenance["access_boundary"]["record_phase_opened_columns"] == ["path", "audio"]
    assert provenance["access_boundary"]["record_phase_raw_label_values_read"] is False
    assert provenance["access_boundary"]["label_artifact_phase_opened_columns"] == ["path", "label"]
    parsed = load_h10_cdadd_target_manifest(manifest_path)
    assert isinstance(parsed, H10TargetManifest)
    assert len(parsed.records) == 4


def test_h10_materializer_validates_h9_handoff_before_target_path(tmp_path: Path) -> None:
    hdd = tmp_path / "hdd"
    hdd.mkdir()
    calls: list[Path] = []

    def reject(ledger: str | Path) -> object:
        calls.append(Path(ledger))
        raise RuntimeError("synthetic H9 ledger rejection")

    with pytest.raises(RuntimeError, match="ledger rejection"):
        materialize_h10_cdadd_target(
            raw_shard_dir=tmp_path / "not-a-real-cdadd-root",
            checkpoint_ledger=tmp_path / "ledger.json",
            metadata_audit=tmp_path / "not-read-audit.json",
            output_dir=hdd / "must-not-exist",
            allowed_output_root=hdd,
            _source_ledger_validator=reject,
        )
    assert len(calls) == 1
    assert not (hdd / "must-not-exist").exists()


def _synthetic_predictions(target: H10TargetManifest, checkpoint: FrozenCheckpoint) -> pd.DataFrame:
    # These deterministic fixture scores derive only from record order.  They
    # do not read the separately sealed target-label artifact.
    index = np.arange(len(target.records), dtype=float)
    if checkpoint.method == "P":
        probability = np.where(index >= 2.0, 0.95, 0.05)
    elif checkpoint.method == "B1":
        probability = np.where(index >= 2.0, 0.60, 0.40)
    else:
        probability = np.where(index >= 2.0, 0.55, 0.45)
    probability = np.clip(probability + (checkpoint.seed - H9_SEEDS[0]) * 1e-7, 1e-5, 1.0 - 1e-5)
    return pd.DataFrame(
        {
            "dataset": H10_CDADD_DATASET,
            "sample_id": target.records["sample_id"].astype(str).tolist(),
            "method": checkpoint.method,
            "seed": checkpoint.seed,
            "checkpoint_sha256": checkpoint.sha256,
            "spoof_logit": np.log(probability / (1.0 - probability)),
            "spoof_probability": probability,
        }
    )


def test_h10_terminal_evaluator_writes_complete_label_free_matrix_before_labels(tmp_path: Path) -> None:
    manifest, ledger_path, ledger = _materialize(tmp_path)
    output = tmp_path / "terminal"
    outputs = terminal_evaluate_h10_cdadd(
        target_manifest_path=manifest,
        checkpoint_ledger_path=ledger_path,
        output_dir=output,
        device="cpu",
        terminal_evaluation=True,
        synthetic_test_mode=True,
        _source_ledger_loader=lambda _path: ledger,
        _prediction_runner=_synthetic_predictions,
    )
    raw = pd.read_parquet(outputs["raw_predictions"])
    metrics = pd.read_csv(outputs["metrics"])
    seed_metrics = pd.read_csv(outputs["seed_metrics"])
    bootstrap = pd.read_csv(outputs["bootstrap"])
    provenance = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
    decision = json.loads(outputs["decision"].read_text(encoding="utf-8"))
    assert len(raw) == 4 * len(METHODS) * len(H9_SEEDS)
    assert "label" not in raw.columns
    assert len(metrics) == len(METHODS)
    assert len(seed_metrics) == len(METHODS) * len(H9_SEEDS)
    assert len(bootstrap) == 2_000
    assert provenance["target_labels_read"] is True
    assert provenance["source_target_canonical_fingerprint_collision_count"] == 0
    assert set(provenance["locked_protocols"]) == {"h10_protocol", "h9_plan", "h9_data"}
    assert provenance["target_construction"]["all_audio_bearing_parquet_trials"] is True
    assert decision["rules"]["complete_label_free_prediction_matrix_written_before_labels"] is True


def test_h10_terminal_interlock_returns_before_target_manifest_is_read(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="labels are sealed"):
        terminal_evaluate_h10_cdadd(
            target_manifest_path=tmp_path / "not-read-manifest.json",
            checkpoint_ledger_path=tmp_path / "not-read-ledger.json",
            output_dir=tmp_path / "not-written",
            device="cpu",
            terminal_evaluation=False,
        )


def test_h10_bootstrap_is_fixed_and_shared() -> None:
    frame = pd.DataFrame(
        {
            "dataset": [H10_CDADD_DATASET] * 6,
            "sample_id": [f"id-{index}" for index in range(6)],
            "label": [0, 0, 0, 1, 1, 1],
            "P": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "B1": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            "B2": [0.8, 0.7, 0.6, 0.4, 0.3, 0.2],
        }
    )
    assert bootstrap_eer_differences(frame).equals(bootstrap_eer_differences(frame))
