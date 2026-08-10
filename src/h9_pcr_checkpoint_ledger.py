"""Build the immutable source-only H9-PCR checkpoint handoff ledger.

This module sits between the source-training loop and the terminal evaluator.
It deliberately has no target-manifest argument, target dataset constant, or
metric implementation.  Its only output is the contract consumed by
``load_frozen_checkpoint_ledger`` in :mod:`src.h9_pcr_evaluation`.

The builder is intentionally stricter than a convenient experiment summary:
it replays the source manifest and frozen P/B2 table validation, checks the
source-only lambda selection, and fails before publication if any of the
twelve final sidecars or checkpoints is stale, incomplete, or unambiguously
incompatible with the sealed H9 terminal-evaluation contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.h9_pcr_evaluation import (
    EVALUATION_BATCH_SIZE,
    H9_EVALUATION_VERSION,
    METHODS,
    load_frozen_checkpoint_ledger,
)
from src.h9_pcr_training import (
    H9_LAMBDA_GRID,
    H9_MARGIN,
    H9_SEEDS,
    H9_SEED_DEVICES,
    load_frozen_b2_pairs,
    load_frozen_p_pairs,
    load_source_manifest,
)
from src.res2tcn_pytorch import sha256_file


_HEX = frozenset("0123456789abcdef")
_SOURCE_FREEZE_KIND = "h9_odss_paired_counterfactual_source_freeze"
_SOURCE_MATERIALIZATION_KIND = "h9_odss_paired_counterfactual_source_materialization"
_SELECTION_KIND = "H9_P_SOURCE_ONLY_LAMBDA_SELECTION"
_FROZEN_SELECTION_ARTIFACT_KIND = "h9_pcr_frozen_source_only_lambda_selection"
_SELECTION_METRIC = "mean_source_dev_eer_over_four_predeclared_P_seeds"
_SELECTION_TIE_BREAK = "lower_lambda_rank"
_SOURCE_BINDING_KEYS: tuple[str, ...] = (
    "source_manifest_sha256",
    "p_pairs_sha256",
    "b2_pairs_sha256",
)
_FROZEN_ENVELOPE: Mapping[str, object] = {
    "batch_size": 24,
    "num_workers": 4,
    "max_epochs": 6,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "margin": H9_MARGIN,
    "require_cuda": True,
}


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"H9 {description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H9 {description} is not a valid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"H9 {description} must be a JSON object")
    return value


def _path(value: str | Path, *, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"H9 {description} is unavailable: {path}")
    return path


def _sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"H9 {description} must be a lowercase SHA-256 digest")
    return value


def _hash(path: Path, *, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"H9 {description} is unavailable: {path}")
    return sha256_file(path)


def _require_hash(path: Path, expected: object, *, description: str) -> str:
    expected_digest = _sha256(expected, description=f"{description} expected hash")
    observed = _hash(path, description=description)
    if observed != expected_digest:
        raise ValueError(f"H9 {description} SHA-256 mismatch")
    return observed


def _require_path_hash(payload: Mapping[str, Any], *, path_key: str, hash_key: str, description: str) -> tuple[Path, str]:
    raw_path = payload.get(path_key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"H9 {description} lacks {path_key}")
    path = _path(raw_path, description=description)
    return path, _require_hash(path, payload.get(hash_key), description=description)


def _require_false(payload: Mapping[str, Any], key: str, *, description: str) -> None:
    if payload.get(key) is not False:
        raise ValueError(f"H9 {description} breached the target firewall in {key}")


def _as_float(value: object, *, description: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"H9 {description} must be finite numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"H9 {description} must be finite numeric")
    return parsed


def _same_path(left: Path, right: Path, *, description: str) -> None:
    if left != right:
        raise ValueError(f"H9 {description} path disagrees with its sealed provenance")


def _validate_source_provenance(
    *,
    source_manifest: Path,
    source_manifest_sha256: str,
    p_pairs: Path,
    p_pairs_sha256: str,
    b2_pairs: Path,
    b2_pairs_sha256: str,
    source_freeze_provenance: Path,
    source_materialization_provenance: Path,
) -> dict[str, Any]:
    """Require that the byte-pinned source inputs originate from both seals."""
    freeze = _read_json(source_freeze_provenance, description="source freeze provenance")
    if freeze.get("artifact_kind") != _SOURCE_FREEZE_KIND:
        raise ValueError("H9 source freeze provenance artifact kind drift")
    source_access = freeze.get("source_access")
    if not isinstance(source_access, Mapping):
        raise ValueError("H9 source freeze provenance lacks source-access guard")
    for key in ("target_data_read", "target_label_read", "target_prediction_read"):
        _require_false(source_access, key, description="source freeze provenance")
    outputs = freeze.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("H9 source freeze provenance lacks sealed outputs")
    output_specs = (
        ("pairs_csv", p_pairs, p_pairs_sha256, "P pairs"),
        ("b2_random_pairs_csv", b2_pairs, b2_pairs_sha256, "B2 pairs"),
    )
    for name, expected_path, expected_hash, description in output_specs:
        value = outputs.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"H9 source freeze provenance lacks {name}")
        path, digest = _require_path_hash(value, path_key="path", hash_key="sha256", description=f"sealed source {description}")
        _same_path(path, expected_path, description=f"sealed source {description}")
        if digest != expected_hash:
            raise ValueError(f"H9 sealed source {description} hash disagrees with input")

    materialization = _read_json(source_materialization_provenance, description="source materialization provenance")
    if materialization.get("artifact_kind") != _SOURCE_MATERIALIZATION_KIND:
        raise ValueError("H9 source materialization provenance artifact kind drift")
    materialization_access = materialization.get("source_access")
    if not isinstance(materialization_access, Mapping):
        raise ValueError("H9 source materialization provenance lacks source-access guard")
    for key in ("target_data_read", "target_label_read", "target_prediction_read"):
        _require_false(materialization_access, key, description="source materialization provenance")
    materialized_outputs = materialization.get("outputs")
    if not isinstance(materialized_outputs, Mapping) or not isinstance(materialized_outputs.get("source_manifest_csv"), Mapping):
        raise ValueError("H9 source materialization provenance lacks source manifest output")
    output_manifest_path, output_manifest_hash = _require_path_hash(
        materialized_outputs["source_manifest_csv"],
        path_key="path",
        hash_key="sha256",
        description="materialized source manifest",
    )
    _same_path(output_manifest_path, source_manifest, description="materialized source manifest")
    if output_manifest_hash != source_manifest_sha256:
        raise ValueError("H9 materialized source manifest hash disagrees with input")
    sealed_freeze = materialization.get("sealed_source_freeze")
    if not isinstance(sealed_freeze, Mapping):
        raise ValueError("H9 source materialization provenance lacks upstream freeze binding")
    if _sha256(sealed_freeze.get("provenance_sha256"), description="materialization upstream freeze hash") != _hash(
        source_freeze_provenance, description="source freeze provenance"
    ):
        raise ValueError("H9 source materialization provenance does not bind the supplied source freeze")
    artifact_hashes = sealed_freeze.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("H9 source materialization provenance lacks upstream pair hashes")
    if artifact_hashes.get(p_pairs.name) != p_pairs_sha256 or artifact_hashes.get(b2_pairs.name) != b2_pairs_sha256:
        raise ValueError("H9 source materialization provenance pair hashes disagree with sealed inputs")
    trainer_validation = materialization.get("trainer_manifest_validation")
    if not isinstance(trainer_validation, Mapping) or trainer_validation.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("H9 source materialization provenance lacks trainer manifest hash binding")
    return {
        "source_freeze_provenance_path": str(source_freeze_provenance),
        "source_freeze_provenance_sha256": _hash(source_freeze_provenance, description="source freeze provenance"),
        "source_materialization_provenance_path": str(source_materialization_provenance),
        "source_materialization_provenance_sha256": _hash(source_materialization_provenance, description="source materialization provenance"),
    }


def _validate_selection(
    path: Path,
    *,
    source_hashes: Mapping[str, str],
    architecture_sha256: str,
) -> tuple[dict[str, Any], float]:
    """Replay all 12 hashed P-selection fits before accepting their aggregate.

    The final source-training ledger must not trust a hand-assembled list of
    EER values.  Each P grid result is bound to a sidecar and checkpoint whose
    source hashes, fresh initialization, CUDA-BF16 provenance, and frozen
    optimization configuration are replayed here, before the selected lambda
    can enter the final B1/B2/P fit matrix.
    """
    selection = _read_json(path, description="source-only lambda selection")
    if selection.get("artifact_kind") != _FROZEN_SELECTION_ARTIFACT_KIND or selection.get("kind") != _SELECTION_KIND:
        raise ValueError("H9 lambda selection artifact kind drift")
    for key in ("target_labels_read", "target_audio_read", "target_metrics_read"):
        _require_false(selection, key, description="source-only lambda selection")
    if selection.get("selection_metric") != _SELECTION_METRIC or selection.get("tie_break") != _SELECTION_TIE_BREAK:
        raise ValueError("H9 lambda selection rule drift")
    if selection.get("applies_unchanged_to") != ["P", "B2"]:
        raise ValueError("H9 lambda selection does not bind both P and B2")
    selected = _as_float(selection.get("selected_lambda_rank"), description="selected lambda rank")
    if selected not in H9_LAMBDA_GRID:
        raise ValueError("H9 selected lambda is outside the locked grid")
    hashes = selection.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping) or dict(hashes) != dict(source_hashes):
        raise ValueError("H9 lambda selection source-artifact binding drift")
    sidecars = selection.get("input_sidecars")
    if not isinstance(sidecars, list) or len(sidecars) != len(H9_LAMBDA_GRID) * len(H9_SEEDS):
        raise ValueError("H9 lambda selection requires exactly twelve hashed P sidecars")
    expected_identities = {(lam, seed) for lam in H9_LAMBDA_GRID for seed in H9_SEEDS}
    observed_sidecars: dict[tuple[float, int], float] = {}
    seen_paths: set[Path] = set()
    for input_record in sidecars:
        if not isinstance(input_record, Mapping):
            raise ValueError("H9 lambda selection sidecar record must be an object")
        raw_path = input_record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("H9 lambda selection sidecar record lacks a path")
        sidecar_path = _path(raw_path, description="lambda-selection P sidecar")
        if sidecar_path in seen_paths:
            raise ValueError("H9 lambda selection sidecar paths must be unique")
        seen_paths.add(sidecar_path)
        _require_hash(sidecar_path, input_record.get("sha256"), description="lambda-selection P sidecar")
        sidecar = _read_json(sidecar_path, description="lambda-selection P sidecar")
        if sidecar.get("method") != "P":
            raise ValueError("H9 lambda selection sidecar method must be P")
        try:
            lambda_rank, seed = float(sidecar.get("lambda_rank")), int(sidecar.get("seed"))
        except (TypeError, ValueError) as error:
            raise ValueError("H9 lambda selection sidecar has invalid lambda/seed identity") from error
        identity = (lambda_rank, seed)
        if identity not in expected_identities or identity in observed_sidecars:
            raise ValueError("H9 lambda selection sidecars do not form the locked P lambda/seed grid")
        for firewall_key in ("target_labels_read", "target_audio_read"):
            _require_false(sidecar, firewall_key, description="lambda-selection P sidecar")
        if sidecar.get("precision") != "cuda_bfloat16_autocast":
            raise ValueError("H9 lambda selection sidecar does not prove CUDA BF16 training")
        if sidecar.get("device") != H9_SEED_DEVICES[seed]:
            raise ValueError("H9 lambda selection sidecar seed/device binding drift")
        if sidecar.get("source_artifact_hashes") != dict(source_hashes):
            raise ValueError("H9 lambda selection sidecar source-artifact binding drift")
        architecture = sidecar.get("architecture_provenance")
        if not isinstance(architecture, Mapping):
            raise ValueError("H9 lambda selection sidecar lacks architecture provenance")
        for name, value in {
            "architecture_sha256": architecture_sha256,
            "initialization": "fresh_seeded",
            "initialization_seed": seed,
            "checkpoint_loaded": False,
        }.items():
            if architecture.get(name) != value:
                raise ValueError(f"H9 lambda selection sidecar architecture provenance drift in {name}")
        raw_checkpoint = sidecar.get("checkpoint_path")
        if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
            raise ValueError("H9 lambda selection sidecar lacks checkpoint_path")
        checkpoint_path = _path(raw_checkpoint, description="lambda-selection P checkpoint")
        _require_hash(checkpoint_path, sidecar.get("checkpoint_sha256"), description="lambda-selection P checkpoint")
        _validate_checkpoint(
            _checkpoint_payload(checkpoint_path),
            method="P",
            seed=seed,
            selected_lambda_rank=lambda_rank,
            source_hashes=source_hashes,
            architecture_sha256=architecture_sha256,
        )
        checkpoint = _checkpoint_payload(checkpoint_path)
        sidecar_eer = _as_float(sidecar.get("source_dev_eer"), description="lambda-selection sidecar source development EER")
        checkpoint_eer = _as_float(checkpoint.get("source_dev_eer"), description="lambda-selection checkpoint source development EER")
        if not 0.0 <= sidecar_eer <= 1.0 or sidecar_eer != checkpoint_eer:
            raise ValueError("H9 lambda selection sidecar/checkpoint source development EER drift")
        observed_sidecars[identity] = sidecar_eer
    if set(observed_sidecars) != expected_identities:
        raise ValueError("H9 lambda selection sidecars do not cover the full locked P grid")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(H9_LAMBDA_GRID):
        raise ValueError("H9 lambda selection requires exactly one candidate per locked lambda")
    observed: dict[float, dict[int, float]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("H9 lambda selection candidate must be an object")
        lam = _as_float(candidate.get("lambda_rank"), description="lambda selection candidate lambda")
        if lam not in H9_LAMBDA_GRID or lam in observed:
            raise ValueError("H9 lambda selection candidates drift from the locked grid")
        values = candidate.get("source_dev_eer_by_seed")
        if not isinstance(values, Mapping) or set(values) != {str(seed) for seed in H9_SEEDS}:
            raise ValueError("H9 lambda selection candidate lacks the four locked seed EERs")
        by_seed = {seed: _as_float(values[str(seed)], description="lambda selection source EER") for seed in H9_SEEDS}
        if any(not 0.0 <= value <= 1.0 for value in by_seed.values()):
            raise ValueError("H9 lambda selection source EER is outside [0, 1]")
        mean = sum(by_seed.values()) / len(by_seed)
        if not math.isclose(
            _as_float(candidate.get("mean_source_dev_eer"), description="lambda selection mean source EER"),
            mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("H9 lambda selection mean does not reconstruct from its seed results")
        if any(by_seed[seed] != observed_sidecars[(lam, seed)] for seed in H9_SEEDS):
            raise ValueError("H9 lambda selection candidates disagree with hashed P sidecars")
        observed[lam] = by_seed
    recomputed = min(H9_LAMBDA_GRID, key=lambda lam: (sum(observed[lam].values()) / len(H9_SEEDS), lam))
    if selected != recomputed:
        raise ValueError("H9 selected lambda does not follow the sealed source-only selection rule")
    return selection, selected


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch releases
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"H9 cannot read checkpoint payload: {path}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("H9 final checkpoint lacks a model state dictionary")
    return payload


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    selected_lambda_rank: float,
    source_hashes: Mapping[str, str],
    architecture_sha256: str,
) -> None:
    if payload.get("source_manifest_sha256") != source_hashes["source_manifest_sha256"]:
        raise ValueError("H9 final checkpoint source manifest binding drift")
    if payload.get("p_pair_csv_sha256") != source_hashes["p_pairs_sha256"]:
        raise ValueError("H9 final checkpoint P-pair binding drift")
    expected_b2 = source_hashes["b2_pairs_sha256"] if method == "B2" else None
    if payload.get("b2_pair_csv_sha256") != expected_b2:
        raise ValueError("H9 final checkpoint B2-pair binding drift")
    if payload.get("selection_rule") != "lowest_source_dev_eer_then_lower_epoch":
        raise ValueError("H9 final checkpoint selection rule drift")
    epoch = payload.get("best_epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= int(_FROZEN_ENVELOPE["max_epochs"]):
        raise ValueError("H9 final checkpoint best epoch violates the frozen source stop")
    eer = _as_float(payload.get("source_dev_eer"), description="final checkpoint source development EER")
    if not 0.0 <= eer <= 1.0:
        raise ValueError("H9 final checkpoint source development EER is outside [0, 1]")
    architecture = payload.get("architecture_provenance")
    if not isinstance(architecture, Mapping):
        raise ValueError("H9 final checkpoint lacks architecture provenance")
    expected_architecture = {
        "architecture_sha256": architecture_sha256,
        "initialization": "fresh_seeded",
        "initialization_seed": seed,
        "checkpoint_loaded": False,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"H9 final checkpoint architecture provenance drift in {key}")
    config = payload.get("training_config")
    if not isinstance(config, Mapping):
        raise ValueError("H9 final checkpoint lacks training configuration")
    expected_lambda = 0.0 if method == "B1" else selected_lambda_rank
    expected = {"method": method, "seed": seed, "device": H9_SEED_DEVICES[seed], **_FROZEN_ENVELOPE}
    for key, expected_value in expected.items():
        observed = config.get(key)
        if isinstance(expected_value, float):
            if _as_float(observed, description=f"final checkpoint config {key}") != expected_value:
                raise ValueError(f"H9 final checkpoint training config drift in {key}")
        elif observed != expected_value:
            raise ValueError(f"H9 final checkpoint training config drift in {key}")
    if _as_float(config.get("lambda_rank"), description="final checkpoint lambda rank") != expected_lambda:
        raise ValueError("H9 final checkpoint lambda rank disagrees with the locked condition")


def _validate_sidecars(
    sidecar_paths: Sequence[Path],
    *,
    source_hashes: Mapping[str, str],
    selected_lambda_rank: float,
    architecture_sha256: str,
) -> list[dict[str, Any]]:
    expected = {(method, seed) for method in METHODS for seed in H9_SEEDS}
    records: list[dict[str, Any]] = []
    observed: set[tuple[str, int]] = set()
    for path in sidecar_paths:
        sidecar = _read_json(path, description="final source training sidecar")
        method, seed = sidecar.get("method"), sidecar.get("seed")
        if method not in METHODS or not isinstance(seed, int) or seed not in H9_SEEDS:
            raise ValueError("H9 final source training sidecar has an invalid method/seed identity")
        key = (str(method), seed)
        if key in observed:
            raise ValueError("H9 final source training sidecars duplicate a method/seed condition")
        observed.add(key)
        for firewall_key in ("target_labels_read", "target_audio_read"):
            _require_false(sidecar, firewall_key, description="final source training sidecar")
        if sidecar.get("precision") != "cuda_bfloat16_autocast":
            raise ValueError("H9 final source training sidecar does not prove CUDA BF16 training")
        if sidecar.get("device") != H9_SEED_DEVICES[seed]:
            raise ValueError("H9 final source training sidecar seed/device binding drift")
        if _as_float(sidecar.get("lambda_rank"), description="final source sidecar lambda rank") != (0.0 if method == "B1" else selected_lambda_rank):
            raise ValueError("H9 final source training sidecar lambda rank drift")
        sidecar_hashes = sidecar.get("source_artifact_hashes")
        if not isinstance(sidecar_hashes, Mapping) or dict(sidecar_hashes) != dict(source_hashes):
            raise ValueError("H9 final source training sidecar source-artifact binding drift")
        architecture = sidecar.get("architecture_provenance")
        if not isinstance(architecture, Mapping):
            raise ValueError("H9 final source training sidecar lacks architecture provenance")
        for name, value in {
            "architecture_sha256": architecture_sha256,
            "initialization": "fresh_seeded",
            "initialization_seed": seed,
            "checkpoint_loaded": False,
        }.items():
            if architecture.get(name) != value:
                raise ValueError(f"H9 final source training sidecar architecture provenance drift in {name}")
        raw_checkpoint = sidecar.get("checkpoint_path")
        if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
            raise ValueError("H9 final source training sidecar lacks checkpoint_path")
        checkpoint_path = _path(raw_checkpoint, description="final source checkpoint")
        checkpoint_hash = _require_hash(
            checkpoint_path,
            sidecar.get("checkpoint_sha256"),
            description="final source checkpoint",
        )
        _validate_checkpoint(
            _checkpoint_payload(checkpoint_path),
            method=str(method),
            seed=seed,
            selected_lambda_rank=selected_lambda_rank,
            source_hashes=source_hashes,
            architecture_sha256=architecture_sha256,
        )
        records.append(
            {
                "method": str(method),
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "training_record_path": str(path),
                "training_record_sha256": _hash(path, description="final source training sidecar"),
                "source_artifact_hashes": dict(source_hashes),
            }
        )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"H9 final source training sidecars must contain exactly all B1/B2/P × four seeds; missing={missing}, extra={extra}")
    return sorted(records, key=lambda record: (str(record["method"]), int(record["seed"])))


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _publish_new_file(path: Path, payload: bytes) -> None:
    """Atomically publish once; an existing path is a hard failure."""
    if path.exists():
        raise FileExistsError(f"H9 frozen checkpoint ledger already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"H9 frozen checkpoint ledger parent does not exist: {path.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        # ``link`` is create-only, unlike replace; it protects the ledger from
        # both accidental reuse and a concurrent process racing this builder.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_frozen_checkpoint_ledger(
    *,
    source_manifest: str | Path,
    p_pairs: str | Path,
    b2_pairs: str | Path,
    source_freeze_provenance: str | Path,
    source_materialization_provenance: str | Path,
    source_selection: str | Path,
    architecture_bundle: str | Path,
    training_records: Sequence[str | Path],
    plan: str | Path,
    data_contract: str | Path,
    output: str | Path,
) -> Path:
    """Validate and publish H9's one immutable source-training handoff.

    The function never opens a target artifact and rejects an existing output
    path before it considers any source sidecar/checkpoint.
    """
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"H9 frozen checkpoint ledger already exists: {output_path}")
    if len(training_records) != len(METHODS) * len(H9_SEEDS):
        raise ValueError("H9 frozen checkpoint ledger requires exactly twelve final source training sidecars")
    source_manifest_path = _path(source_manifest, description="source manifest")
    p_pairs_path = _path(p_pairs, description="source P pairs")
    b2_pairs_path = _path(b2_pairs, description="source B2 pairs")
    freeze_path = _path(source_freeze_provenance, description="source freeze provenance")
    materialization_path = _path(source_materialization_provenance, description="source materialization provenance")
    selection_path = _path(source_selection, description="source-only lambda selection")
    bundle_path = _path(architecture_bundle, description="architecture bundle")
    if not bundle_path.is_dir():
        raise ValueError("H9 architecture bundle must be a directory")
    architecture_path = bundle_path / "_net.py"
    architecture_sha256 = _hash(architecture_path, description="architecture definition")
    plan_path = _path(plan, description="locked H9 plan")
    data_path = _path(data_contract, description="locked H9 data contract")

    # Replay source-pool and pair constraints before admitting any final model.
    manifest = load_source_manifest(source_manifest_path)
    p_artifact = load_frozen_p_pairs(p_pairs_path, manifest)
    b2_artifact = load_frozen_b2_pairs(b2_pairs_path, manifest, p_artifact)
    source_hashes = {
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "p_pairs_sha256": p_artifact.sha256,
        "b2_pairs_sha256": b2_artifact.sha256,
    }
    source_provenance = _validate_source_provenance(
        source_manifest=source_manifest_path,
        source_manifest_sha256=manifest.source_manifest_sha256,
        p_pairs=p_pairs_path,
        p_pairs_sha256=p_artifact.sha256,
        b2_pairs=b2_pairs_path,
        b2_pairs_sha256=b2_artifact.sha256,
        source_freeze_provenance=freeze_path,
        source_materialization_provenance=materialization_path,
    )
    _, selected_lambda_rank = _validate_selection(
        selection_path,
        source_hashes=source_hashes,
        architecture_sha256=architecture_sha256,
    )
    records = _validate_sidecars(
        [Path(path).expanduser().resolve() for path in training_records],
        source_hashes=source_hashes,
        selected_lambda_rank=selected_lambda_rank,
        architecture_sha256=architecture_sha256,
    )
    ledger: dict[str, Any] = {
        "artifact_kind": "h9_pcr_frozen_checkpoint_ledger",
        "version": H9_EVALUATION_VERSION,
        "target_labels_read": False,
        "target_audio_read": False,
        "target_metrics_read": False,
        "protocol": {
            "plan_sha256": _hash(plan_path, description="locked H9 plan"),
            "data_contract_sha256": _hash(data_path, description="locked H9 data contract"),
        },
        "source_artifacts": {
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "p_pairs_path": str(p_pairs_path),
            "p_pairs_sha256": p_artifact.sha256,
            "b2_pairs_path": str(b2_pairs_path),
            "b2_pairs_sha256": b2_artifact.sha256,
        },
        "source_provenance": source_provenance,
        "architecture": {"bundle_dir": str(bundle_path), "architecture_sha256": architecture_sha256},
        "source_selection": {
            "selected_lambda_rank": selected_lambda_rank,
            "path": str(selection_path),
            "sha256": _hash(selection_path, description="source-only lambda selection"),
        },
        "training_envelope": dict(_FROZEN_ENVELOPE),
        "checkpoints": records,
    }
    payload = _canonical_bytes(ledger)
    # Validate the emitted schema through the exact terminal-evaluator loader
    # before its one-time create-only publication.  This reads source only.
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.validation.",
        suffix=".json",
        delete=False,
    ) as handle:
        validation_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        load_frozen_checkpoint_ledger(validation_path, plan_path=plan_path, data_contract_path=data_path)
    finally:
        if validation_path.exists():
            validation_path.unlink()
    _publish_new_file(output_path, payload)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one fail-closed source-only H9-PCR checkpoint ledger.")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--p-pairs", required=True, type=Path)
    parser.add_argument("--b2-pairs", required=True, type=Path)
    parser.add_argument("--source-freeze-provenance", required=True, type=Path)
    parser.add_argument("--source-materialization-provenance", required=True, type=Path)
    parser.add_argument("--source-selection", required=True, type=Path)
    parser.add_argument("--res2-bundle", required=True, type=Path)
    parser.add_argument("--training-record", action="append", required=True, type=Path, help="One final source-only training sidecar; provide exactly 12.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON path; existing paths are refused.")
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = build_frozen_checkpoint_ledger(
        source_manifest=args.source_manifest,
        p_pairs=args.p_pairs,
        b2_pairs=args.b2_pairs,
        source_freeze_provenance=args.source_freeze_provenance,
        source_materialization_provenance=args.source_materialization_provenance,
        source_selection=args.source_selection,
        architecture_bundle=args.res2_bundle,
        training_records=args.training_record,
        plan=args.plan,
        data_contract=args.data_contract,
        output=args.output,
    )
    print(json.dumps({"ledger_path": str(output), "ledger_sha256": sha256_file(output)}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through script wrapper
    raise SystemExit(cli_main())
