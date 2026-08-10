"""Strict, score-free H7 fixed-vector cross-corpus transfer audit."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_features import FEATURE_NAMES


CORPORA = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
METADATA_COLUMNS = ("sample_id", "source_id", "label", "view")
FEATURE_COLUMNS = tuple(FEATURE_NAMES)
FREEZE_COLUMNS = ("dataset", "label", "source_id", "sample_id", "selection_rank", "selection_key_sha256")
EXPECTED_VIEWS = {"full_waveform", "deterministic_crop", "preemphasized_crop"}
RESPONSE_TOKENS = ("score", "logit", "detector", "model", "eer")
FREEZE_SEED = 2612
MAX_PER_LABEL = 5000
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_CONFIDENCE = 0.95


def locked_paths(feature_root: Path) -> dict[str, Path]:
    """Return the protocol-pinned H7 feature table paths."""
    return {corpus: feature_root / corpus / "features_wide.parquet" for corpus in CORPORA}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_like(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in RESPONSE_TOKENS)


def _nonempty(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().fillna("")


def _require_corpus_paths(paths: Mapping[str, Path]) -> None:
    if tuple(paths) != CORPORA:
        raise ValueError(f"H7 requires exact ordered corpora {CORPORA}, got {tuple(paths)}")


def _read_metadata(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    path = Path(path).resolve()
    if _response_like(str(path)):
        raise ValueError(f"H7 forbids response-like input path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing H7 input: {path}")
    schema = tuple(pq.ParquetFile(path).schema.names)
    bad = [column for column in schema if _response_like(column)]
    if bad:
        raise ValueError(f"H7 forbids response-like source columns: {bad}")
    missing = [column for column in METADATA_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"H7 source lacks metadata columns: {missing}")
    frame = pd.read_parquet(path, columns=list(METADATA_COLUMNS))
    views = set(frame["view"].astype("string").dropna().tolist())
    if views != EXPECTED_VIEWS:
        raise ValueError(f"Unexpected H7 source views: {sorted(views)}")
    full = frame.loc[frame["view"].eq("full_waveform")].copy()
    if full.empty:
        raise ValueError("No full_waveform rows")
    for column in ("sample_id", "source_id"):
        if _nonempty(full[column]).eq("").any():
            raise ValueError(f"H7 empty {column}")
    labels = pd.to_numeric(full["label"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int)) != {0, 1}:
        raise ValueError("H7 requires binary labels 0/1")
    full["label"] = labels.astype(int)
    if full["sample_id"].duplicated().any() or full["source_id"].duplicated().any():
        raise ValueError("H7 full_waveform sample_id/source_id must be unique")
    return full, schema


def _selection_key(dataset: str, label: int, source_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{label}|{source_id}".encode("utf-8")).hexdigest()


def freeze_inputs(paths: Mapping[str, Path], *, seed: int = FREEZE_SEED, max_per_label: int = MAX_PER_LABEL) -> tuple[pd.DataFrame, dict[str, object]]:
    """Freeze score-free H7 source IDs from projected metadata only."""
    _require_corpus_paths(paths)
    if seed != FREEZE_SEED or max_per_label != MAX_PER_LABEL:
        raise ValueError("Production H7 freeze requires the locked seed and cap")
    selected_rows: list[dict[str, object]] = []
    source_manifest: dict[str, object] = {}
    for dataset in CORPORA:
        path = Path(paths[dataset]).resolve()
        full, schema = _read_metadata(path)
        source_manifest[dataset] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "schema": list(schema),
            "full_waveform_rows": int(len(full)),
        }
        for label in (0, 1):
            subset = full.loc[full["label"].eq(label), ["sample_id", "source_id"]].copy()
            subset["selection_key_sha256"] = subset["source_id"].map(lambda value: _selection_key(dataset, label, str(value), seed))
            subset = subset.sort_values(["selection_key_sha256", "source_id"], kind="mergesort").head(max_per_label).reset_index(drop=True)
            if len(subset) < max_per_label:
                raise ValueError(f"H7 {dataset}/label={label} has fewer than {max_per_label} source IDs")
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                selected_rows.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "source_id": str(row.source_id),
                        "sample_id": str(row.sample_id),
                        "selection_rank": rank,
                        "selection_key_sha256": str(row.selection_key_sha256),
                    }
                )
    manifest = pd.DataFrame(selected_rows, columns=list(FREEZE_COLUMNS))
    expected_rows = len(CORPORA) * 2 * max_per_label
    if len(manifest) != expected_rows or manifest.duplicated(["dataset", "sample_id"]).any():
        raise RuntimeError("H7 freeze output incomplete or duplicate")
    provenance: dict[str, object] = {
        "protocol": "experiments/h7_feature_transfer/protocol.md",
        "purpose": "score-free fixed-vector cross-corpus transfer freeze",
        "corpora": list(CORPORA),
        "seed": seed,
        "max_per_label": max_per_label,
        "view": "full_waveform",
        "projection_columns": list(METADATA_COLUMNS),
        "selection_key": "sha256('2612|dataset|label|source_id') ascending",
        "source_manifest": source_manifest,
        "reads": {"score": False, "feature_values": False, "audio": False, "asr": False, "model": False},
    }
    return manifest, provenance


def write_freeze(output_dir: Path, manifest: pd.DataFrame, provenance: Mapping[str, object]) -> None:
    """Write immutable freeze artifacts."""
    output_dir = Path(output_dir)
    targets = {
        "csv": output_dir / "h7_selected_sources.csv",
        "parquet": output_dir / "h7_selected_sources.parquet",
        "provenance": output_dir / "h7_input_freeze_provenance.json",
    }
    if output_dir.exists() or any(path.exists() for path in targets.values()):
        raise FileExistsError(f"Refusing to overwrite H7 input freeze: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest.to_csv(targets["csv"], index=False)
    manifest.to_parquet(targets["parquet"], index=False)
    targets["provenance"].write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_freeze(freeze_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    freeze_dir = Path(freeze_dir)
    manifest_path = freeze_dir / "h7_selected_sources.csv"
    provenance_path = freeze_dir / "h7_input_freeze_provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("H7 analysis requires complete freeze CSV and provenance")
    manifest = pd.read_csv(manifest_path, dtype={"source_id": "string", "sample_id": "string", "selection_key_sha256": "string"})
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if tuple(manifest.columns) != FREEZE_COLUMNS:
        raise ValueError("H7 freeze manifest schema drift")
    if provenance.get("seed") != FREEZE_SEED or provenance.get("max_per_label") != MAX_PER_LABEL:
        raise ValueError("H7 freeze seed/cap drift")
    if tuple(provenance.get("corpora", [])) != CORPORA or provenance.get("view") != "full_waveform":
        raise ValueError("H7 freeze corpus/view drift")
    if len(manifest) != len(CORPORA) * 2 * MAX_PER_LABEL or manifest.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("H7 freeze manifest is incomplete or duplicate")
    if set(manifest["dataset"]) != set(CORPORA) or set(manifest["label"].astype(int)) != {0, 1}:
        raise ValueError("H7 freeze manifest identities drift")
    return manifest, provenance


def _feature_matrix_for_dataset(dataset: str, selected: pd.DataFrame, source: Mapping[str, object]) -> pd.DataFrame:
    path = Path(str(source["path"])).resolve()
    if _response_like(str(path)) or not path.is_file():
        raise ValueError(f"Invalid H7 feature source path: {path}")
    if int(path.stat().st_size) != int(source["size_bytes"]) or _sha256_file(path) != str(source["sha256"]):
        raise ValueError(f"H7 source hash mismatch for {dataset}")
    schema = tuple(pq.ParquetFile(path).schema.names)
    if tuple(source["schema"]) != schema:
        raise ValueError(f"H7 source schema mismatch for {dataset}")
    bad = [column for column in schema if _response_like(column)]
    if bad:
        raise ValueError(f"H7 source unexpectedly contains response-like columns: {bad}")
    required = [*METADATA_COLUMNS, *FEATURE_COLUMNS]
    if any(column not in schema for column in required):
        raise ValueError(f"H7 source misses metadata/features for {dataset}")
    frame = pd.read_parquet(path, columns=required)
    full = frame.loc[frame["view"].eq("full_waveform")].copy()
    if full["sample_id"].duplicated().any() or full["source_id"].duplicated().any():
        raise ValueError(f"H7 source IDs duplicate for {dataset}")
    joined = selected.merge(full, on=["sample_id", "source_id"], how="left", validate="one_to_one", suffixes=("_freeze", "_source"))
    if joined["label_source"].isna().any() or not joined["label_freeze"].eq(joined["label_source"].astype(int)).all():
        raise ValueError(f"H7 selected identity/label mismatch for {dataset}")
    if not joined["view"].eq("full_waveform").all():
        raise ValueError(f"H7 selected view mismatch for {dataset}")
    result = joined[["dataset", "sample_id", "source_id", "label_freeze", *FEATURE_COLUMNS]].rename(columns={"label_freeze": "label"})
    if len(result) != 2 * MAX_PER_LABEL:
        raise ValueError(f"H7 selected count mismatch for {dataset}")
    return result


def _eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    differences = fpr - fnr
    zero = np.flatnonzero(np.isclose(differences, 0.0, atol=1e-15))
    if len(zero):
        return float(fpr[int(zero[0])])
    crossing = np.flatnonzero(np.signbit(differences[:-1]) != np.signbit(differences[1:]))
    if not len(crossing):
        return float((fpr[np.argmin(np.abs(differences))] + fnr[np.argmin(np.abs(differences))]) / 2.0)
    index = int(crossing[0])
    fpr_low, fpr_high = float(fpr[index]), float(fpr[index + 1])
    difference_low, difference_high = float(differences[index]), float(differences[index + 1])
    # A vertical ROC segment has identical FPRs but can cross the FNR line.
    # Its interpolated EER is that shared FPR; bracketing a root over a
    # zero-width interval would fail despite a valid empirical crossing.
    if np.isclose(fpr_low, fpr_high, rtol=0.0, atol=1e-15):
        return fpr_low
    weight = -difference_low / (difference_high - difference_low)
    return float(fpr_low + weight * (fpr_high - fpr_low))


def _cell_seed(dataset: str) -> int:
    digest = hashlib.sha256(f"{FREEZE_SEED}|H7|{dataset}|auroc_bootstrap".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _bootstrap_auroc(labels: np.ndarray, scores: np.ndarray, clusters: np.ndarray, *, seed: int) -> dict[str, object]:
    unique = np.array(sorted(set(clusters.astype(str))))
    positions = [np.flatnonzero(clusters.astype(str) == cluster) for cluster in unique]
    generator = np.random.default_rng(seed)
    values = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = generator.integers(0, len(positions), size=len(positions))
        indices = np.concatenate([positions[index] for index in chosen])
        if len(np.unique(labels[indices])) < 2:
            continue
        values[replicate] = roc_auc_score(labels[indices], scores[indices])
    valid = values[np.isfinite(values)]
    if len(valid) < int(BOOTSTRAP_REPLICATES * 0.95):
        raise RuntimeError("H7 AUROC bootstrap has insufficient valid replicates")
    low, high = np.quantile(valid, [(1.0 - BOOTSTRAP_CONFIDENCE) / 2.0, 1.0 - (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0])
    return {
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid": int(len(valid)),
        "bootstrap_replicates_invalid": int(BOOTSTRAP_REPLICATES - len(valid)),
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_cluster_count": int(len(unique)),
        "bootstrap_cluster_rule": "source_id",
        "bootstrap_seed": seed,
        "auroc_ci_low": float(low),
        "auroc_ci_high": float(high),
    }


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)],
        ]
    )


def run_analysis(freeze_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit exactly five locked LOO feature-only classifiers from a sealed freeze."""
    manifest, freeze = _load_freeze(freeze_dir)
    source_manifest = freeze.get("source_manifest")
    if not isinstance(source_manifest, dict) or set(source_manifest) != set(CORPORA):
        raise ValueError("H7 freeze source manifest drift")
    panels = []
    for dataset in CORPORA:
        selected = manifest.loc[manifest["dataset"].eq(dataset)].copy()
        panels.append(_feature_matrix_for_dataset(dataset, selected, source_manifest[dataset]))
    all_rows = pd.concat(panels, ignore_index=True)
    result_rows: list[dict[str, object]] = []
    for test_dataset in CORPORA:
        train = all_rows.loc[~all_rows["dataset"].eq(test_dataset)].copy()
        test = all_rows.loc[all_rows["dataset"].eq(test_dataset)].copy()
        if len(train) != 8 * MAX_PER_LABEL or len(test) != 2 * MAX_PER_LABEL:
            raise RuntimeError("H7 train/test cardinality drift")
        x_train = train.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        x_test = test.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=FREEZE_SEED)),
            ]
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(x_train, y_train)
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise RuntimeError(f"H7 logistic regression failed to converge for held-out {test_dataset}")
        scores = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
        auroc = float(roc_auc_score(y_test, scores))
        eer = _eer(y_test, scores)
        transformed_feature_count = int(pipeline.named_steps["imputer"].transform(x_train).shape[1])
        bootstrap = _bootstrap_auroc(y_test, scores, test["source_id"].astype(str).to_numpy(), seed=_cell_seed(test_dataset))
        result_rows.append(
            {
                "held_out_dataset": test_dataset,
                "train_datasets": ";".join(dataset for dataset in CORPORA if dataset != test_dataset),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_train_bonafide": int((y_train == 0).sum()),
                "n_train_spoof": int((y_train == 1).sum()),
                "n_test_bonafide": int((y_test == 0).sum()),
                "n_test_spoof": int((y_test == 1).sum()),
                "raw_feature_count": len(FEATURE_COLUMNS),
                "post_imputation_feature_count": transformed_feature_count,
                "auroc": auroc,
                "eer": eer,
                "fit_converged": True,
                **bootstrap,
            }
        )
    table = pd.DataFrame(result_rows)
    if len(table) != len(CORPORA) or table["held_out_dataset"].tolist() != list(CORPORA) or not np.isfinite(table[["auroc", "eer", "auroc_ci_low", "auroc_ci_high"]].to_numpy(dtype=float)).all():
        raise RuntimeError("H7 result matrix incomplete or non-finite")
    summary = {
        "protocol": "experiments/h7_feature_transfer/protocol.md",
        "freeze_dir": str(Path(freeze_dir)),
        "freeze_manifest_sha256": _sha256_file(Path(freeze_dir) / "h7_selected_sources.csv"),
        "freeze_provenance_sha256": _sha256_file(Path(freeze_dir) / "h7_input_freeze_provenance.json"),
        "corpora": list(CORPORA),
        "representation": "all_28_full_waveform_features",
        "classifier": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "random_state": FREEZE_SEED},
        "preprocessing": "train-only median imputation plus missing indicators and standardization",
        "complete_matrix_cells": int(len(table)),
        "median_held_out_auroc": float(table["auroc"].median()),
        "median_held_out_eer": float(table["eer"].median()),
        "claims_not_supported": ["detector_reliance", "causality", "cue_selection", "model_ranking", "mitigation_performance"],
    }
    return table, summary


def write_analysis(output_dir: Path, table: pd.DataFrame, summary: Mapping[str, object]) -> None:
    """Write immutable H7 result artifacts and a compact interpretation note."""
    output_dir = Path(output_dir)
    targets = {
        "csv": output_dir / "h7_leave_one_corpus_out.csv",
        "parquet": output_dir / "h7_leave_one_corpus_out.parquet",
        "summary": output_dir / "h7_analysis_provenance.json",
        "note": output_dir / "H7_ANALYSIS_001.md",
    }
    if output_dir.exists() or any(path.exists() for path in targets.values()):
        raise FileExistsError(f"Refusing to overwrite H7 analysis output: {output_dir}")
    output_dir.mkdir(parents=True)
    table.to_csv(targets["csv"], index=False)
    table.to_parquet(targets["parquet"], index=False)
    targets["summary"].write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    display = table.loc[:, ["held_out_dataset", "n_train", "n_test", "auroc", "auroc_ci_low", "auroc_ci_high", "eer", "post_imputation_feature_count"]].copy()
    for column in ("auroc", "auroc_ci_low", "auroc_ci_high", "eer"):
        display[column] = display[column].map(lambda value: f"{float(value):.6f}")
    lines = [
        "# H7 fixed-vector cross-corpus transfer audit — analysis 001",
        "",
        "This complete five-cell, score-free leave-one-corpus-out matrix uses one locked all-28-feature logistic baseline. It is descriptive feature-label transfer context only: it does not test a detector, select a cue, establish causality, rank models, or evaluate mitigation.",
        "",
        "## Complete matrix",
        "",
        _markdown_table(display),
        "",
        f"The locked unweighted median held-out AUROC is {float(summary['median_held_out_auroc']):.6f}; the median held-out EER is {float(summary['median_held_out_eer']):.6f}. These are summary descriptors without an acceptance threshold.",
        "",
        "Every cell uses 40,000 frozen training rows, 10,000 frozen held-out rows, all 28 registered full-waveform features, train-only median imputation plus missing indicators and standardization, and the fixed L2 logistic recipe. AUROC intervals use 500 source-ID cluster bootstrap replicates; source IDs are singleton in these cohorts, so this is an utterance-level cluster bootstrap. The adjacent provenance JSON binds the two freeze artifacts, the model recipe, and the complete matrix. No score artifact, detector/model code, audio, ASR, feature coefficient, or H1/H4/H5/H6 result was used.",
    ]
    targets["note"].write_text("\n".join(lines) + "\n", encoding="utf-8")
