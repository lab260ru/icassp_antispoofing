#!/usr/bin/env python3
"""Declare held-out H1 bootstrap candidates from an immutable discovery freeze.

This command is intentionally a provenance-preserving projection, not a
selection utility.  It reads only a previously frozen discovery manifest and
creates exact analyzer-schema rows for named held-out datasets.  In
particular, it never opens an association table, score panel, feature table,
or audio file and never ranks, filters, or otherwise reselects candidates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_associations import CANDIDATE_MANIFEST_COLUMNS, load_frozen_candidates


DISCOVERY_DATASETS = ("ASVspoof2019_LA", "ASVspoof2021_LA", "ASVspoof2021_DF")
CONFIRMATION_DATASETS = ("InTheWild", "ASVspoof5")
IDENTITY_COLUMNS = ("model", "view", "feature", "class_label")
PROVENANCE_COLUMNS = ("selection_status", "selection_split", "selection_basis", "frozen_at_utc")


def sha256_file(path: Path) -> str:
    """Return a content hash for a compact, tracked provenance artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc_timestamp(value: str) -> str:
    """Require an explicit RFC3339 UTC declaration timestamp."""
    if not value.endswith("Z"):
        raise ValueError("--declared-at-utc must be an explicit UTC timestamp ending in 'Z'.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(
            "--declared-at-utc must be ISO-8601 / RFC3339 UTC, e.g. 2026-08-10T12:34:56Z."
        ) from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("--declared-at-utc must resolve to UTC.")
    return value


def validate_confirmation_datasets(targets: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Accept each registered held-out corpus once and preserve caller order."""
    requested = tuple(targets)
    if not requested:
        raise ValueError("At least one --target-dataset is required.")
    unknown = sorted(set(requested) - set(CONFIRMATION_DATASETS))
    if unknown:
        raise ValueError(f"Target datasets must be limited to {list(CONFIRMATION_DATASETS)}; got {unknown}.")
    if len(set(requested)) != len(requested):
        raise ValueError("Each --target-dataset may be supplied at most once.")
    return requested


def load_discovery_freeze(path: Path) -> pd.DataFrame:
    """Load a complete discovery freeze and reject any confirmation-derived input."""
    candidates = load_frozen_candidates(path)
    if candidates.empty:
        raise ValueError("The discovery freeze contains no candidates to declare for confirmation.")
    if not (candidates["selection_split"].str.lower() == "discovery").all():
        raise ValueError("The source manifest must be a discovery freeze, not a preregistration or confirmation manifest.")
    source_datasets = set(candidates["dataset"])
    if source_datasets != set(DISCOVERY_DATASETS):
        raise ValueError(
            "The source manifest must contain exactly the three registered discovery datasets "
            f"{list(DISCOVERY_DATASETS)}, found {sorted(source_datasets)}."
        )

    for identity, rows in candidates.groupby(list(IDENTITY_COLUMNS), sort=False, dropna=False):
        observed = set(rows["dataset"])
        if observed != set(DISCOVERY_DATASETS) or len(rows) != len(DISCOVERY_DATASETS):
            raise ValueError(
                "Every frozen candidate identity must have one provenance row for each discovery dataset; "
                f"identity={identity}, datasets={sorted(observed)}."
            )
        inconsistent = [column for column in PROVENANCE_COLUMNS if rows[column].nunique(dropna=False) != 1]
        if inconsistent:
            raise ValueError(
                "A frozen candidate identity has inconsistent discovery provenance in "
                f"{inconsistent}: {identity}."
            )
    return candidates


def make_confirmation_manifest(
    discovery_candidates: pd.DataFrame,
    targets: tuple[str, ...],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Project frozen identities to held-out names while preserving provenance fields.

    The second return value is a compact source-key mapping for the adjacent
    provenance report; it is deliberately not added to the analyzer manifest,
    whose schema must remain exact.
    """
    output_rows: list[dict[str, object]] = []
    identity_report: list[dict[str, object]] = []
    grouped = discovery_candidates.groupby(list(IDENTITY_COLUMNS), sort=True, dropna=False)
    for identity, source_rows in grouped:
        source_rows = source_rows.sort_values("dataset", kind="stable")
        template = source_rows.iloc[0]
        identity_record = dict(zip(IDENTITY_COLUMNS, identity))
        identity_record["source_discovery_keys"] = source_rows[
            ["dataset", *IDENTITY_COLUMNS]
        ].to_dict("records")
        identity_report.append(identity_record)
        for target in targets:
            output_rows.append(
                {
                    "dataset": target,
                    **{column: template[column] for column in IDENTITY_COLUMNS},
                    **{column: template[column] for column in PROVENANCE_COLUMNS},
                }
            )
    manifest = pd.DataFrame(output_rows, columns=CANDIDATE_MANIFEST_COLUMNS)
    if manifest.duplicated(["dataset", *IDENTITY_COLUMNS]).any():
        raise ValueError("Internal error: declaration generated duplicate exact confirmation candidates.")
    manifest = manifest.sort_values(
        ["dataset", *IDENTITY_COLUMNS], kind="stable"
    ).reset_index(drop=True)
    return manifest, identity_report


def _selection_basis_hashes(candidates: pd.DataFrame) -> dict[str, str | None]:
    """Record pinned local basis hashes when their declared files are available."""
    hashes: dict[str, str | None] = {}
    for basis in sorted(candidates["selection_basis"].unique()):
        path = REPO_ROOT / basis
        hashes[str(basis)] = sha256_file(path) if path.is_file() else None
    return hashes


def write_declaration_outputs(
    manifest: pd.DataFrame,
    identity_report: list[dict[str, object]],
    *,
    source_manifest: Path,
    source_candidates: pd.DataFrame,
    targets: tuple[str, ...],
    declared_at_utc: str,
    output_manifest: Path,
    provenance_report: Path,
) -> None:
    """Write a non-overwriting target manifest and declaration-only report."""
    source_manifest = source_manifest.resolve()
    output_manifest = output_manifest.resolve()
    provenance_report = provenance_report.resolve()
    if output_manifest in {source_manifest, provenance_report} or provenance_report == source_manifest:
        raise ValueError("Source manifest, output manifest, and provenance report must be different paths.")
    existing = [path for path in (output_manifest, provenance_report) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite declaration artifact(s): {[str(path) for path in existing]}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    provenance_report.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    report = {
        "artifact_kind": "h1_confirmation_candidate_declaration",
        "claim_guard": (
            "Declaration-only projection of a frozen discovery identity; it is not a confirmation result, "
            "a new selection, a ranking, a portability finding, or a causal claim."
        ),
        "declared_at_utc": declared_at_utc,
        "target_datasets": list(targets),
        "construction": {
            "reads_confirmation_association_or_bootstrap_output": False,
            "reads_confirmation_features_scores_or_audio": False,
            "ranks_filters_or_reselects_candidates": False,
            "identity_projection": "Each unique (model, view, feature, class_label) from the complete discovery freeze is copied to every explicitly named target dataset.",
        },
        "source_discovery_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
            "row_count": int(len(source_candidates)),
            "schema": CANDIDATE_MANIFEST_COLUMNS,
            "selection_basis_sha256_if_available": _selection_basis_hashes(source_candidates),
        },
        "source_candidate_identities": identity_report,
        "manifest": {
            "path": str(output_manifest),
            "sha256": sha256_file(output_manifest),
            "row_count": int(len(manifest)),
            "schema": CANDIDATE_MANIFEST_COLUMNS,
        },
    }
    provenance_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-manifest", required=True, type=Path)
    parser.add_argument(
        "--target-dataset",
        action="append",
        required=True,
        choices=CONFIRMATION_DATASETS,
        help="Registered held-out dataset to receive an unchanged frozen identity; repeat for both targets.",
    )
    parser.add_argument("--declared-at-utc", required=True, type=parse_utc_timestamp)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--provenance-report", required=True, type=Path)
    args = parser.parse_args()

    targets = validate_confirmation_datasets(args.target_dataset)
    source_candidates = load_discovery_freeze(args.discovery_manifest)
    manifest, identity_report = make_confirmation_manifest(source_candidates, targets)
    write_declaration_outputs(
        manifest,
        identity_report,
        source_manifest=args.discovery_manifest,
        source_candidates=source_candidates,
        targets=targets,
        declared_at_utc=args.declared_at_utc,
        output_manifest=args.output_manifest,
        provenance_report=args.provenance_report,
    )
    print(f"wrote {args.output_manifest} ({len(manifest)} declaration-only confirmation candidate rows)")
    print(f"wrote {args.provenance_report}")


if __name__ == "__main__":
    main()
