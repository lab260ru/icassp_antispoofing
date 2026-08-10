"""Synthetic guardrails for the prospective two-panel main-paper figure.

These tests deliberately do not open H1/H2 result artifacts and never render.
They validate the static contract plus isolated, synthetic source schemas.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import src.main_crest_evidence_boundary as figure


def _manifest() -> dict[str, object]:
    return json.loads(figure.MANIFEST_PATH.read_text(encoding="utf-8"))


def _h1_row(*, dataset: str, stage: str = "screen") -> dict[str, str]:
    return {
        "dataset": dataset,
        "model": "Spectra-AASIST",
        "view": "full_waveform",
        "feature": "crest_factor_db",
        "class_label": "1",
        "analysis_stage": stage,
        "partial_spearman_rho": "-0.12",
        "partial_spearman_q": "0.20",
    }


def test_static_manifest_is_exactly_the_eight_source_two_panel_contract() -> None:
    manifest = _manifest()
    validated = figure._validate_manifest(manifest, verify_sources=False)

    assert tuple(manifest["fixed_corpus_order"]) == figure.CORPORA
    assert tuple(manifest["fixed_h2_arm_order"]) == figure.H2_ARMS
    assert manifest["focus"] == figure.FOCUS
    assert [item["id"] for item in manifest["inputs"]] == list(figure.EXPECTED_SOURCES)
    assert set(validated) == set(figure.EXPECTED_SOURCES)
    assert len(manifest["inputs"]) == 8
    assert "h4" not in json.dumps(manifest).casefold()
    assert "label separation" not in manifest["caption"].casefold()


def test_manifest_rejects_substitution_extra_input_and_h4_redirect() -> None:
    substituted = deepcopy(_manifest())
    substituted["inputs"][0]["relative_path"] = (
        "experiments/h1_feature_association/results/ASVspoof2021_LA/association_summary.csv"
    )
    with pytest.raises(ValueError, match="changed fixed relative_path"):
        figure._validate_manifest(substituted, verify_sources=False)

    with_extra = deepcopy(_manifest())
    with_extra["inputs"].append(deepcopy(with_extra["inputs"][0]))
    with pytest.raises(ValueError, match="exactly the fixed eight-source set"):
        figure._validate_manifest(with_extra, verify_sources=False)

    h4_redirect = deepcopy(_manifest())
    h4_redirect["inputs"][0]["relative_path"] = "experiments/h4_label_transportability/result.csv"
    h4_redirect["inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed fixed relative_path"):
        figure._validate_manifest(h4_redirect, verify_sources=False)


@pytest.mark.parametrize(
    "relative_path",
    [
        "experiments/h4_label_transportability/table.csv",
        "artifacts/raw_scores/catalog.csv",
        "models/checkpoint.onnx",
        "audio/clip.wav",
        "asr/transcript.json",
        "experiments/future_directions/h2b/result.csv",
    ],
)
def test_source_firewall_rejects_disallowed_boundaries(relative_path: str) -> None:
    with pytest.raises(ValueError, match="forbidden source boundary"):
        figure._entry_path({"id": "forbidden", "relative_path": relative_path}, require_file=False)


def test_h1_parsers_accept_only_the_exact_fixed_slice_and_frozen_confirmation() -> None:
    assert figure._parse_h1_association([_h1_row(dataset="ASVspoof2019_LA")], dataset="ASVspoof2019_LA", label="synthetic") == pytest.approx(-0.12)

    wrong_slice = _h1_row(dataset="ASVspoof2019_LA")
    wrong_slice["feature"] = "spectral_flatness"
    with pytest.raises(ValueError, match="exactly one fixed H1 slice row"):
        figure._parse_h1_association([wrong_slice], dataset="ASVspoof2019_LA", label="synthetic")

    confirmation = _h1_row(dataset="InTheWild", stage="confirmation_bootstrap")
    confirmation.update(
        {
            "selection_status": "frozen",
            "selection_split": "discovery",
            "partial_spearman_ci_low": "-0.20",
            "partial_spearman_ci_high": "-0.03",
        }
    )
    assert figure._parse_h1_confirmation([confirmation], dataset="InTheWild", label="synthetic") == pytest.approx(
        (-0.20, -0.03)
    )
    confirmation["selection_status"] = "post_hoc"
    with pytest.raises(ValueError, match="discovery-frozen"):
        figure._parse_h1_confirmation([confirmation], dataset="InTheWild", label="synthetic")


def test_csv_reader_rejects_response_like_columns_on_synthetic_input(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    source.write_text("dataset,response_score\nASVspoof2019_LA,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response-like columns"):
        figure._read_csv(source, label="synthetic")


def test_h2_parser_preserves_no_detector_boundary_and_fixed_arm_order() -> None:
    summary = {
        "artifact_kind": "h2_detector_free_quality_summary",
        "detector_scoring_allowed": False,
        "panel_gate_status": "not_frozen",
        "per_arm": {
            "drc_cf3": {"retained_fraction": 0.181},
            "drc_cf6": {"retained_fraction": 0.006},
            "small_gain_plus_0p1db": {"retained_fraction": 0.646},
            "polarity": {"retained_fraction": 0.999},
        },
    }
    assert figure._parse_h2_retention(summary).tolist() == pytest.approx([0.181, 0.006, 0.646, 0.999])

    summary["detector_scoring_allowed"] = True
    with pytest.raises(ValueError, match="no-detector"):
        figure._parse_h2_retention(summary)


def test_renderer_refuses_before_reading_inputs_without_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_input_read() -> None:
        raise AssertionError("sealed input should not be read without authorization")

    monkeypatch.setattr(figure, "load_locked_main_evidence", unexpected_input_read)
    with pytest.raises(PermissionError, match="without explicit maintainer authorization"):
        figure.render_locked_main_figure()


def test_output_contract_is_fixed_to_the_three_main_paper_assets() -> None:
    outputs = figure._fixed_output_paths()
    assert outputs == {
        "pdf": figure.OUTPUT_DIR / "fig_crest_evidence_boundary_main.pdf",
        "png": figure.OUTPUT_DIR / "fig_crest_evidence_boundary_main.png",
        "metadata": figure.OUTPUT_DIR / "fig_crest_evidence_boundary_main.metadata.json",
    }


def test_output_contract_refuses_to_replace_an_existing_paper_asset(tmp_path: Path) -> None:
    outputs = {
        "pdf": tmp_path / "figure.pdf",
        "png": tmp_path / "figure.png",
        "metadata": tmp_path / "figure.metadata.json",
    }
    figure._require_fresh_output_paths(outputs)
    outputs["png"].write_bytes(b"already-inspected")
    with pytest.raises(FileExistsError, match="already exist"):
        figure._require_fresh_output_paths(outputs)
