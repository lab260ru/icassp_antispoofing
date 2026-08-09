"""Unit tests for the static anonymous-paper PDF readiness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.paper_pdf_checks as checks


class _FakePage(dict):
    def __init__(self, text: str, fonts: dict[str, object], *, width: float = 612.0, height: float = 792.0) -> None:
        super().__init__({"/Resources": {"/Font": fonts}})
        self._text = text
        self.mediabox = _FakeBox(width, height)

    def extract_text(self) -> str:
        return self._text


class _FakeBox:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height


class _FakeReader:
    def __init__(self, pages: list[_FakePage], metadata: dict[str, str] | None = None) -> None:
        self.pages = pages
        self.metadata = metadata or {"/Creator": "test"}


def _embedded_font() -> dict[str, object]:
    return {"/BaseFont": "/Test", "/FontDescriptor": {"/FontFile2": object()}}


def _reader_factory(reader: _FakeReader):
    def build(_path: str) -> _FakeReader:
        return reader

    return build


def test_audit_accepts_anonymous_four_page_pdf_with_embedded_font(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-fake")
    pages = [
        _FakePage("Title", {"/F1": _embedded_font()}),
        _FakePage("TABLE I results", {"/F1": _embedded_font()}),
        _FakePage("References", {"/F1": _embedded_font()}),
        _FakePage("continued", {"/F1": _embedded_font()}),
    ]
    monkeypatch.setattr(checks, "PdfReader", _reader_factory(_FakeReader(pages)))

    report = checks.audit_working_draft_pdf(pdf)

    assert report["ok"] is True
    assert report["font_count"] == 1
    assert report["forbidden_text_hits"] == {}
    assert report["us_letter_geometry"] is True
    assert report["page_sizes_points"] == [[612.0, 792.0]] * 4


def test_audit_reports_identity_metadata_and_missing_font_embedding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-fake")
    unembedded_font = {"/BaseFont": "/Test"}
    pages = [
        _FakePage("Kirill", {"/F1": unembedded_font}, width=595.0, height=842.0),
        _FakePage("not a table", {"/F1": unembedded_font}),
        _FakePage("not references", {"/F1": unembedded_font}),
    ]
    monkeypatch.setattr(checks, "PdfReader", _reader_factory(_FakeReader(pages, {"/Author": "A Name"})))

    report = checks.audit_working_draft_pdf(pdf)

    assert report["ok"] is False
    assert "kirill" in report["forbidden_text_hits"]
    assert any("/Author" in str(error) for error in report["errors"])
    assert any("unembedded" in str(error) for error in report["errors"])
    assert any("US Letter" in str(error) for error in report["errors"])


def test_audit_rejects_missing_file_and_invalid_page_expectation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checks.audit_working_draft_pdf(tmp_path / "missing.pdf")
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-fake")
    with pytest.raises(ValueError, match="positive"):
        checks.audit_working_draft_pdf(pdf, expected_pages=0)
