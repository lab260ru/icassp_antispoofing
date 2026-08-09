"""Select H2B Q1 quality-feasible transform arms without detector data.

The selector consumes a complete detector-free Q1 table, its compact Q1
summary, and the committed finite arm manifest.  It implements the locked
Wilson-bound/target-delta/WER ordering and is structurally unable to load an
Arena score artifact or a detector response.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.h2_pre_score_pairs import assert_score_independent_columns
from src.h2b_quality_calibration import _load_q1_arm_manifest


H2B_Q1_SELECTION_VERSION = "h2b_q1_quality_selection_v1"
Z_95 = 1.959963984540054


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_lower_bound(successes: int, total: int, *, z: float = Z_95) -> float:
    """Return a two-sided Wilson lower bound, with 0 for an empty group."""
    if total <= 0:
        return 0.0
    if successes < 0 or successes > total or not math.isfinite(z) or z <= 0.0:
        raise ValueError("Invalid Wilson successes, total, or z value")
    p = successes / total
    denominator = 1.0 + (z * z) / total
    centre = p + (z * z) / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    return float((centre - radius) / denominator)


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"H2B Q1 required JSON is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"H2B Q1 JSON is invalid: {source}") from error
    if not isinstance(value, Mapping):
        raise ValueError("H2B Q1 JSON must be an object")
    return value


def _require_complete_summary(summary: Mapping[str, Any], expected_pairs: int) -> None:
    if summary.get("artifact_kind") != "h2b_q1_detector_free_quality_summary":
        raise ValueError("H2B Q1 selection requires the finalized detector-free Q1 summary")
    if summary.get("detector_scoring_allowed") is not False:
        raise ValueError("H2B Q1 selection refuses a score-authorized summary")
    if summary.get("complete") is not True or int(summary.get("completed_pairs", -1)) != expected_pairs:
        raise ValueError("H2B Q1 selection requires a complete quality table")


def select_h2b_q1_families(
    quality_table: str | Path,
    quality_summary: str | Path,
    arm_manifest: str | Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Evaluate every locked Q1 family and select at most one candidate per family."""
    table_path = Path(quality_table)
    if not table_path.is_file():
        raise FileNotFoundError(f"H2B Q1 quality table is missing: {table_path}")
    table = pd.read_parquet(table_path)
    assert_score_independent_columns(table.columns)
    required = {"arm", "retained", "target_feature", "target_feature_before", "target_feature_after", "wer"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"H2B Q1 quality table lacks required fields: {missing}")
    arms, manifest = _load_q1_arm_manifest(arm_manifest)
    known_arms = {arm.arm_id: arm for arm in arms}
    if table.empty or set(table["arm"].astype(str)) != set(known_arms):
        raise ValueError("H2B Q1 quality table arm set does not exactly match its locked arm manifest")
    if table.duplicated(["pair_id"]).any():
        raise ValueError("H2B Q1 quality table has duplicate pair IDs")
    expected_pairs = len(table)
    _require_complete_summary(_read_json(quality_summary), expected_pairs)

    policy = manifest["selection_policy"]
    lower_min = float(policy["minimum_lower_wilson_retention_bound"])
    families = manifest["families"]
    rows: list[dict[str, Any]] = []
    for family in families:
        family_id = str(family["family_id"])
        target_feature = str(family["target_feature"])
        minimum_delta = float(family["minimum_abs_median_target_delta"])
        candidates = [str(value) for value in family["candidate_arm_ids"]]
        for arm_id in candidates:
            arm_table = table.loc[table["arm"].astype(str).eq(arm_id)].copy()
            if not arm_table["target_feature"].astype(str).eq(target_feature).all():
                raise ValueError(f"H2B Q1 target feature mismatch for arm {arm_id}")
            retained = arm_table["retained"].astype(bool)
            total = int(len(arm_table))
            successes = int(retained.sum())
            before = pd.to_numeric(arm_table["target_feature_before"], errors="coerce")
            after = pd.to_numeric(arm_table["target_feature_after"], errors="coerce")
            target_delta = after - before
            finite_delta = target_delta[np.isfinite(target_delta)]
            wer = pd.to_numeric(arm_table["wer"], errors="coerce")
            finite_wer = wer[np.isfinite(wer)]
            lower = wilson_lower_bound(successes, total)
            abs_median_delta = float(np.median(np.abs(finite_delta))) if len(finite_delta) else float("nan")
            median_wer = float(np.median(finite_wer)) if len(finite_wer) else float("nan")
            eligible = bool(
                lower >= lower_min
                and math.isfinite(abs_median_delta)
                and abs_median_delta >= minimum_delta
                and math.isfinite(median_wer)
            )
            rows.append(
                {
                    "family_id": family_id,
                    "arm": arm_id,
                    "target_feature": target_feature,
                    "n_pairs": total,
                    "retained_pairs": successes,
                    "retained_fraction": successes / total,
                    "lower_wilson_95": lower,
                    "minimum_lower_wilson_95": lower_min,
                    "median_abs_target_delta": abs_median_delta,
                    "minimum_abs_median_target_delta": minimum_delta,
                    "median_wer": median_wer,
                    "eligible": eligible,
                    "selected": False,
                    "status": "eligible_not_selected" if eligible else "not_eligible",
                }
            )
    report = pd.DataFrame(rows).sort_values(["family_id", "arm"], kind="stable").reset_index(drop=True)
    for family_id, indices in report.groupby("family_id", sort=True).groups.items():
        candidates = report.loc[list(indices)]
        eligible = candidates.loc[candidates["eligible"]]
        if eligible.empty:
            continue
        winner = eligible.sort_values(
            ["lower_wilson_95", "median_abs_target_delta", "median_wer", "arm"],
            ascending=[False, False, True, True],
            kind="stable",
        ).index[0]
        report.loc[winner, ["selected", "status"]] = [True, "selected_quality_feasible"]
    provenance: dict[str, Any] = {
        "artifact_kind": "h2b_q1_quality_selection",
        "version": H2B_Q1_SELECTION_VERSION,
        "claim_guard": (
            "Detector-free quality-arm selection only; no detector output, H1 candidate, EER, paired score delta, "
            "or causal conclusion is present. A selected family remains ineligible for scoring until Q2/Q3/Q4."
        ),
        "detector_scoring_allowed": False,
        "quality_table": {"path": str(table_path.resolve()), "sha256": sha256_file(table_path), "n_rows": expected_pairs},
        "quality_summary": {"path": str(Path(quality_summary).resolve()), "sha256": sha256_file(quality_summary)},
        "arm_manifest": {"path": str(Path(arm_manifest).resolve()), "sha256": sha256_file(arm_manifest)},
        "selection_policy": policy,
        "selected_arms": report.loc[report["selected"], ["family_id", "arm"]].to_dict(orient="records"),
        "report_sha256": hashlib.sha256(_canonical_json(report.to_dict(orient="records")).encode("utf-8")).hexdigest(),
    }
    return report, provenance


def write_h2b_q1_selection(
    report: pd.DataFrame,
    provenance: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write non-overwriting compact CSV/JSON selection artifacts."""
    root = Path(output_dir).resolve()
    report_path, provenance_path = root / "quality_family_selection.csv", root / "quality_family_selection.json"
    if root.exists() or report_path.exists() or provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite H2B Q1 selection output: {root}")
    root.mkdir(parents=True, exist_ok=False)
    report.to_csv(report_path, index=False)
    provenance_path.write_text(json.dumps(dict(provenance), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, provenance_path
