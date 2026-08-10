"""Score-free disclosure audit for the completed H1 covariate design.

This module intentionally reports only metadata availability.  It does not
load any H1 feature values or score artifacts and cannot compute an
association, a bootstrap interval, or an intervention decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq


CORPORA = (
    "ASVspoof2019_LA",
    "ASVspoof2021_LA",
    "ASVspoof2021_DF",
    "InTheWild",
    "ASVspoof5",
)
AUDIT_COLUMNS = (
    "sample_id",
    "source_id",
    "label",
    "view",
    "duration_seconds",
    "integrated_lufs",
    "speaker_id",
    "attack_id",
)
EXPECTED_VIEWS = {"full_waveform", "deterministic_crop", "preemphasized_crop"}
RESPONSE_TOKENS = ("score", "logit", "detector", "model", "eer", "auroc", "audio", "asr")
BOOTSTRAP_CONTEXT = {
    "method": "stratified_clustered_percentile",
    "cluster_rule": "speaker_id_when_nonempty_else_source_id",
    "strata": "label",
    "replicates": 2000,
    "seed": 2609,
    "confirmation_selected_spoof_cluster_counts": {
        "InTheWild": 54,
        "ASVspoof5": 367,
    },
}


def locked_paths(feature_root: Path) -> dict[str, Path]:
    """Return the exact five H1 feature-table locations under ``feature_root``."""
    return {corpus: feature_root / corpus / "features_wide.parquet" for corpus in CORPORA}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_response_like(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in RESPONSE_TOKENS)


def _nonempty_strings(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().fillna("")


def _finite_count(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce")
    return int(pd.Series(numeric).map(math.isfinite).sum())


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional formatting package."""
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def validate_and_summarize(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate fixed H1 metadata paths and return per-corpus/label coverage.

    ``paths`` is injectable only for synthetic tests.  Production callers use
    the five protocol-pinned paths returned by :func:`locked_paths`.
    """
    if tuple(paths) != CORPORA:
        raise ValueError(f"Expected exact ordered corpus keys {CORPORA}, got {tuple(paths)}")

    rows: list[dict[str, object]] = []
    source_manifest: dict[str, object] = {}
    for corpus in CORPORA:
        path = Path(paths[corpus]).resolve()
        if _is_response_like(str(path)):
            raise ValueError(f"Response-like source path is forbidden: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned feature table: {path}")

        schema_names = tuple(pq.ParquetFile(path).schema.names)
        bad_columns = [name for name in schema_names if _is_response_like(name)]
        if bad_columns:
            raise ValueError(f"Response-like source columns are forbidden: {bad_columns}")
        missing = [column for column in AUDIT_COLUMNS if column not in schema_names]
        if missing:
            raise ValueError(f"H1 metadata source is missing required columns: {missing}")

        # Explicit projection is the firewall that keeps registry feature
        # columns from entering memory even though the source Parquet stores
        # them beside the H1 metadata.
        frame = pd.read_parquet(path, columns=list(AUDIT_COLUMNS))
        if tuple(frame.columns) != AUDIT_COLUMNS:
            raise ValueError("Metadata projection drifted from the locked audit columns")
        observed_views = set(frame["view"].astype("string").dropna().tolist())
        if observed_views != EXPECTED_VIEWS:
            raise ValueError(f"Unexpected views for {corpus}: {sorted(observed_views)}")
        selected = frame.loc[frame["view"].eq("full_waveform")].copy()
        if selected.empty:
            raise ValueError(f"No full_waveform metadata rows for {corpus}")
        if selected["sample_id"].isna().any() or _nonempty_strings(selected["sample_id"]).eq("").any():
            raise ValueError(f"Invalid sample_id in {corpus}")
        if selected["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate full_waveform sample_id in {corpus}")
        labels = pd.to_numeric(selected["label"], errors="coerce")
        if labels.isna().any() or set(labels.astype(int)) != {0, 1}:
            raise ValueError(f"Expected binary labels 0/1 in {corpus}")
        selected["label"] = labels.astype(int)

        source_manifest[corpus] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
            "schema": list(schema_names),
            "full_waveform_samples": int(len(selected)),
        }
        for label in (0, 1):
            subset = selected.loc[selected["label"].eq(label)]
            if subset.empty:
                raise ValueError(f"No label {label} rows in {corpus}")
            row: dict[str, object] = {
                "dataset": corpus,
                "label": label,
                "n_samples": int(len(subset)),
            }
            for column in ("duration_seconds", "integrated_lufs"):
                row[f"{column}_finite_n"] = _finite_count(subset[column])
            for column in ("speaker_id", "attack_id", "source_id"):
                values = _nonempty_strings(subset[column])
                present = values.loc[values.ne("")]
                row[f"{column}_nonempty_n"] = int(len(present))
                row[f"{column}_unique_n"] = int(present.nunique())
            rows.append(row)

    table = pd.DataFrame(rows)
    expected_rows = len(CORPORA) * 2
    if len(table) != expected_rows or table.duplicated(["dataset", "label"]).any():
        raise RuntimeError("Covariate-availability table is incomplete or non-unique")
    provenance: dict[str, object] = {
        "purpose": "H1 covariate-availability reporting audit; no score-dependent statistic",
        "corpora": list(CORPORA),
        "projection_columns": list(AUDIT_COLUMNS),
        "source_manifest": source_manifest,
        "partial_spearman_controls": {
            "numeric": ["duration_seconds", "integrated_lufs"],
            "categorical": ["speaker_id", "attack_id"],
            "intercept": True,
            "numeric_missing_policy": "median_fill",
            "categorical_missing_policy": "explicit_missing_category",
            "tested_variable_control_policy": "remove_tested_variable_from_controls",
        },
        "confirmation_bootstrap_context": BOOTSTRAP_CONTEXT,
        "forbidden_operations": [
            "score_or_model_read",
            "feature_value_read",
            "association_or_p_value",
            "bootstrap_rerun",
            "candidate_or_intervention_selection",
        ],
    }
    return table, provenance


def write_outputs(output_dir: Path, table: pd.DataFrame, provenance: Mapping[str, object]) -> None:
    """Write the audit's three immutable compact reporting artifacts."""
    output_dir = Path(output_dir)
    targets = {
        "csv": output_dir / "h1_covariate_availability.csv",
        "json": output_dir / "h1_covariate_availability_provenance.json",
        "markdown": output_dir / "H1_COVARIATE_AVAILABILITY_AUDIT_001.md",
    }
    if output_dir.exists() or any(target.exists() for target in targets.values()):
        raise FileExistsError(f"Refusing to overwrite H1 audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    table.to_csv(targets["csv"], index=False)
    targets["json"].write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# H1 covariate-availability reporting audit — run 001",
        "",
        "This score-free reporting audit describes metadata availability in the completed H1 feature cohorts. It does not read feature values, scores, models, audio, or ASR; it does not compute an association, bootstrap, candidate, or intervention result.",
        "",
        "## Partial-adjustment implementation",
        "",
        "`partial_spearman` uses an intercept; ranked duration and integrated loudness with median fill; speaker and attack one-hot dummies with an explicit missing category; and removes a tested feature or response if it would otherwise control itself. The full coverage table and byte-bound source manifest are in the adjacent CSV and JSON.",
        "",
        "## Held-out bootstrap context",
        "",
        "The already sealed confirmation intervals use 2,000 label-stratified clustered percentile replicates with seed 2609 and speaker ID when nonempty, otherwise source ID. The selected spoof slice has 54 speaker clusters in InTheWild and 367 in ASVspoof5. No bootstrap was rerun for this report.",
        "",
        "## Compact coverage",
        "",
        _markdown_table(table),
        "",
        "All paths, input hashes, schemas, and prohibited operations are recorded in `h1_covariate_availability_provenance.json`.",
    ]
    targets["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
