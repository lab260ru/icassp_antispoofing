"""Static, no-input-read tests for the reviewed layout-only main figure v2."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.main_crest_evidence_boundary_v2 as figure_v2


def test_v2_dependency_pins_match_the_committed_v1_contract() -> None:
    figure_v2._verify_layout_only_dependencies()


def test_v2_refuses_before_reading_inputs_without_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_input_read() -> None:
        raise AssertionError("v2 must not read sealed inputs without authorization")

    monkeypatch.setattr(figure_v2, "load_locked_main_evidence", unexpected_input_read)
    with pytest.raises(PermissionError, match="without explicit maintainer authorization"):
        figure_v2.render_main_figure_v2()


def test_v2_requires_a_nonempty_authorization_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(figure_v2, "_verify_layout_only_dependencies", lambda: None)
    with pytest.raises(PermissionError, match="non-empty"):
        figure_v2.render_main_figure_v2(authorized=True, authorization_note="")


def test_v2_output_contract_is_new_and_non_overwritable(tmp_path: Path) -> None:
    outputs = {"pdf": tmp_path / "v2.pdf", "png": tmp_path / "v2.png", "metadata": tmp_path / "v2.json"}
    figure_v2._require_fresh_output_paths(outputs)
    outputs["pdf"].write_bytes(b"preserved-v2")
    with pytest.raises(FileExistsError, match="already exist"):
        figure_v2._require_fresh_output_paths(outputs)


def test_v2_does_not_depend_on_h4_h5_h6_or_s1_sources() -> None:
    source = figure_v2.V1_MODULE_PATH.read_text(encoding="utf-8")
    protocol = figure_v2.V2_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "H4/H5/H6/S1" not in source
    assert "H4/H5/H6/S1" in protocol
    assert figure_v2.OUTPUT_DIR.name == "figures"
