"""Freeze H2B-Q0 source rows without audio, scores, ASR, or detectors.

H2B begins with an independently declared waveform-quality calibration source.
This module accepts only a byte-bound score-free CSV/provenance pair emitted by
``src.h2_input_csv`` and creates an immutable class-balanced Q0 selection.  It
deliberately contains no audio-shard, feature, ASR, transform, or detector
import: decoding is downstream of a committed Q0 freeze.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.h2_input_csv import sha256_file
from src.h2_pre_score_pairs import FrozenInputManifest, freeze_input_manifest, validate_score_independent_input


H2B_Q0_MANIFEST_VERSION = "h2b_q0_score_blind_manifest_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"H2B Q0 input provenance is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"H2B Q0 input provenance is invalid JSON: {source}") from error
    if not isinstance(value, Mapping):
        raise ValueError("H2B Q0 input provenance must be a JSON object")
    return value


def _assert_score_free_provenance(
    provenance: Mapping[str, Any],
    *,
    input_csv: Path,
    dataset: str,
    revision: str,
) -> None:
    if provenance.get("artifact_kind") != "h2_score_free_input_csv":
        raise ValueError("H2B Q0 requires h2_score_free_input_csv provenance")
    source_access = provenance.get("source_access")
    expected_access = {
        "audio_read": False,
        "feature_read": False,
        "score_read": False,
        "asr_loaded": False,
        "detector_loaded": False,
    }
    if source_access != expected_access:
        raise ValueError("H2B Q0 input provenance does not prove score-free label-only source access")
    source_dataset = provenance.get("dataset")
    if not isinstance(source_dataset, Mapping):
        raise ValueError("H2B Q0 input provenance lacks dataset identity")
    if source_dataset.get("name") != dataset or source_dataset.get("revision") != revision:
        raise ValueError("H2B Q0 input provenance dataset/revision does not match the declared Q0 source")
    output_csv = provenance.get("output_csv")
    if not isinstance(output_csv, Mapping) or not isinstance(output_csv.get("sha256"), str):
        raise ValueError("H2B Q0 input provenance lacks byte-bound output_csv SHA-256")
    observed = sha256_file(input_csv)
    if output_csv["sha256"] != observed:
        raise RuntimeError(
            "H2B Q0 score-free input CSV SHA-256 mismatch: "
            f"expected {output_csv['sha256']}, observed {observed}"
        )


def freeze_h2b_q0_manifest(
    input_csv: str | Path,
    input_provenance: str | Path,
    *,
    dataset: str,
    revision: str,
    per_label: int,
    seed: int,
) -> FrozenInputManifest:
    """Freeze Q0 metadata rows from a byte-bound score-free source CSV.

    ``dataset`` and ``revision`` are explicit to prevent using an arbitrary
    Arena input table merely because it contains binary labels.  The resulting
    rows are still detector-ineligible and must be used only by a later,
    separately committed quality-calibration runner.
    """
    csv_path = Path(input_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"H2B Q0 input CSV is missing: {csv_path}")
    if csv_path.suffix.casefold() != ".csv":
        raise ValueError(f"H2B Q0 input must be CSV, got: {csv_path}")
    provenance_path = Path(input_provenance)
    provenance = _read_json(provenance_path)
    _assert_score_free_provenance(provenance, input_csv=csv_path, dataset=dataset, revision=revision)

    rows = validate_score_independent_input(pd.read_csv(csv_path))
    if set(rows["dataset"].astype(str)) != {dataset}:
        raise ValueError(f"H2B Q0 CSV must contain only declared dataset {dataset!r}")
    if "dataset_revision" not in rows.columns or set(rows["dataset_revision"].astype(str)) != {revision}:
        raise ValueError(f"H2B Q0 CSV must contain only declared revision {revision!r}")

    frozen = freeze_input_manifest(rows, per_label=per_label, seed=seed, input_csv_sha256=sha256_file(csv_path))
    frozen_rows = frozen.rows.copy()
    frozen_rows["h2b_stage"] = "q0_frozen_score_blind"
    frozen_rows["h2b_manifest_version"] = H2B_Q0_MANIFEST_VERSION
    q0_provenance: dict[str, Any] = {
        "artifact_kind": "h2b_q0_score_blind_input_freeze",
        "manifest_version": H2B_Q0_MANIFEST_VERSION,
        "claim_guard": (
            "Q0 metadata selection only; no audio was decoded and no feature, ASR, transform, score, or detector was read. "
            "The selected rows remain detector-ineligible."
        ),
        "h2b_stage": "q0_frozen_score_blind",
        "detector_scoring_allowed": False,
        "declared_source": {"dataset": dataset, "revision": revision},
        "input_csv": {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)},
        "input_provenance": {"path": str(provenance_path.resolve()), "sha256": sha256_file(provenance_path)},
        "input_score_free_provenance_sha256": _sha256_json(provenance),
        "selection": dict(frozen.provenance),
        "frozen_rows_sha256": _sha256_json(frozen_rows.to_dict(orient="records")),
    }
    return FrozenInputManifest(rows=frozen_rows, provenance=q0_provenance)


def write_h2b_q0_artifacts(
    frozen: FrozenInputManifest,
    *,
    output_manifest: str | Path,
    output_provenance: str | Path,
) -> None:
    """Write immutable Q0 CSV/JSON artifacts without replacing any prior run."""
    manifest_path = Path(output_manifest).resolve()
    provenance_path = Path(output_provenance).resolve()
    if manifest_path == provenance_path:
        raise ValueError("H2B Q0 manifest and provenance output paths must differ")
    existing = [str(path) for path in (manifest_path, provenance_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite H2B Q0 artifact(s): {existing}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frozen.rows.to_csv(manifest_path, index=False)
    provenance = dict(frozen.provenance)
    provenance["output_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "n_rows": int(len(frozen.rows)),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
