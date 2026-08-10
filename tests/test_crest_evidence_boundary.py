"""Guardrails for the sealed supplementary crest evidence-boundary renderer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.crest_evidence_boundary import (
    CORPORA,
    H2_ARMS,
    MANIFEST_PATH,
    _entry_path,
    _read_csv,
    _validate_manifest,
    load_locked_evidence,
    render_locked_crest_evidence_boundary,
)


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_locked_inputs_revalidate_and_extract_only_the_fixed_quantities() -> None:
    manifest, validated, evidence = load_locked_evidence()
    assert tuple(manifest["fixed_corpus_order"]) == CORPORA
    assert set(validated) == {
        "h1_association_asvspoof2019_la",
        "h1_association_asvspoof2021_la",
        "h1_association_asvspoof2021_df",
        "h1_association_inthewild",
        "h1_association_asvspoof5",
        "h1_confirmation_inthewild",
        "h1_confirmation_asvspoof5",
        "h2_quality_full_002_summary",
        "h4_label_cue_matrix",
    }
    assert evidence["h1_partial_spearman_rho"].tolist() == pytest.approx(
        [-0.4538847801, -0.0991427559, -0.0649496949, -0.0326059797, 0.0013834716]
    )
    assert evidence["h2_retained_fraction"].tolist() == pytest.approx([0.181, 0.006, 0.646, 0.999])
    assert evidence["h4_signed_label_auroc"].tolist() == pytest.approx(
        [-0.4927612, -0.4593344, -0.2619364, 0.34277296, -0.01947456]
    )
    assert tuple(evidence["h1_held_out_ci"]) == ("InTheWild", "ASVspoof5")
    assert tuple(H2_ARMS) == ("drc_cf3", "drc_cf6", "small_gain_plus_0p1db", "polarity")


def test_manifest_rejects_any_redirect_or_hash_change() -> None:
    redirected = deepcopy(_manifest())
    source = next(item for item in redirected["inputs"] if item["id"] == "h1_association_asvspoof2019_la")
    source["relative_path"] = "experiments/h1_feature_association/results/ASVspoof2021_LA/association_summary.csv"
    with pytest.raises(ValueError, match="changed fixed relative_path"):
        _validate_manifest(redirected)

    altered_hash = deepcopy(_manifest())
    source = next(item for item in altered_hash["inputs"] if item["id"] == "h2_quality_full_002_summary")
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed fixed sha256"):
        _validate_manifest(altered_hash)


def test_rejection_firewall_blocks_forbidden_paths_and_response_columns(tmp_path: Path) -> None:
    forbidden = tmp_path / "scores" / "catalog.csv"
    forbidden.parent.mkdir()
    forbidden.write_text("dataset,value\nASVspoof2019_LA,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden source boundary"):
        _entry_path({"id": "adversarial", "absolute_path": str(forbidden)})

    response_csv = tmp_path / "summary.csv"
    response_csv.write_text("dataset,response_score\nASVspoof2019_LA,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response-like columns"):
        _read_csv(response_csv, label="adversarial response table")


def test_renderer_writes_vector_raster_and_hash_bound_metadata(tmp_path: Path) -> None:
    outputs = render_locked_crest_evidence_boundary(tmp_path / "figure")
    assert outputs["pdf"].is_file() and outputs["pdf"].stat().st_size > 1024
    assert outputs["png"].is_file() and outputs["png"].stat().st_size > 1024
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["figure"]["pdf_vector"] is True
    assert metadata["figure"]["png_dpi"] == 300
    assert metadata["claim_boundary"]["causal_claim"] is False
    assert metadata["claim_boundary"]["detector_scoring_available"] is False
    assert len(metadata["input_hashes"]) == 9
