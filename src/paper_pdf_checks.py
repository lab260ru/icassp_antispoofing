"""Static, reproducible checks for the anonymous ICASSP working-draft PDF.

This module is intentionally a *local readiness check*, not a replacement for
an ICASSP-provided template, PDF eXpress, or conference submission checker.
It verifies only properties that can be recovered from a built PDF: expected
page count, essential layout landmarks, embedded fonts, metadata, and a small
set of project-identifying strings that must not appear in an anonymous draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


DEFAULT_FORBIDDEN_TEXT = ("lab260", "kirill", "nikita", "ivan")
US_LETTER_WIDTH_POINTS = 612.0
US_LETTER_HEIGHT_POINTS = 792.0
PAGE_SIZE_TOLERANCE_POINTS = 0.5


def _resolve(value: Any) -> Any:
    """Resolve a pypdf indirect object while retaining plain test doubles."""
    resolver = getattr(value, "get_object", None)
    return resolver() if callable(resolver) else value


def _font_is_embedded(font: Any) -> bool:
    """Return whether a simple or composite PDF font contains an embedded program."""
    resolved_font = _resolve(font)
    descriptor = _resolve(resolved_font.get("/FontDescriptor"))
    if descriptor is None:
        descendants = resolved_font.get("/DescendantFonts") or []
        if descendants:
            descriptor = _resolve(_resolve(descendants[0]).get("/FontDescriptor"))
    return bool(descriptor and any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")))


def _embedded_font_summary(reader: Any) -> list[dict[str, object]]:
    fonts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for page_number, page in enumerate(reader.pages, start=1):
        resources = _resolve(page.get("/Resources")) or {}
        page_fonts = _resolve(resources.get("/Font")) or {}
        for resource_name, font in page_fonts.items():
            resolved_font = _resolve(font)
            base_font = str(resolved_font.get("/BaseFont", "unknown"))
            key = (str(resource_name), base_font)
            if key in seen:
                continue
            seen.add(key)
            fonts.append(
                {
                    "resource": str(resource_name),
                    "base_font": base_font,
                    "embedded": _font_is_embedded(resolved_font),
                    "first_page": page_number,
                }
            )
    return fonts


def _page_size_points(page: Any) -> tuple[float, float]:
    """Return a page's MediaBox width/height in PDF points."""
    media_box = getattr(page, "mediabox", None)
    if media_box is None:
        media_box = _resolve(page.get("/MediaBox"))
    if media_box is None:
        raise ValueError("PDF page has no MediaBox")
    width = getattr(media_box, "width", None)
    height = getattr(media_box, "height", None)
    if width is None or height is None:
        lower_left = media_box.lower_left
        upper_right = media_box.upper_right
        width = upper_right[0] - lower_left[0]
        height = upper_right[1] - lower_left[1]
    return float(width), float(height)


def _is_us_letter(size: tuple[float, float]) -> bool:
    """Return whether a PDF MediaBox matches US Letter within conversion noise."""
    width, height = size
    return (
        abs(width - US_LETTER_WIDTH_POINTS) <= PAGE_SIZE_TOLERANCE_POINTS
        and abs(height - US_LETTER_HEIGHT_POINTS) <= PAGE_SIZE_TOLERANCE_POINTS
    )


def audit_working_draft_pdf(
    pdf_path: Path | str,
    *,
    expected_pages: int = 4,
    forbidden_text: tuple[str, ...] = DEFAULT_FORBIDDEN_TEXT,
) -> dict[str, object]:
    """Inspect a built anonymous working draft and return a serializable report.

    The caller owns any submission-policy decision. A report with ``ok=False``
    is a local failure that should be corrected before treating the PDF as a
    readable handoff artifact.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Paper PDF does not exist: {path}")
    if expected_pages < 1:
        raise ValueError("expected_pages must be positive")

    reader = PdfReader(str(path))
    page_text = [(page.extract_text() or "") for page in reader.pages]
    lower_text = [text.casefold() for text in page_text]
    page_sizes = [_page_size_points(page) for page in reader.pages]
    fonts = _embedded_font_summary(reader)
    metadata = {str(key): str(value) for key, value in dict(reader.metadata or {}).items()}
    errors: list[str] = []
    if len(reader.pages) != expected_pages:
        errors.append(f"expected {expected_pages} pages but found {len(reader.pages)}")
    if len(page_text) < 2 or "table i" not in lower_text[1]:
        errors.append("Table I is not recoverable from extracted text on page 2")
    if len(page_text) < 3 or "references" not in lower_text[2]:
        errors.append("References are not recoverable from extracted text on page 3")
    if not all(_is_us_letter(size) for size in page_sizes):
        observed = ", ".join(f"{width:.1f}x{height:.1f}" for width, height in page_sizes)
        errors.append(f"all PDF pages must have US Letter MediaBox 612.0x792.0 pt; observed {observed}")
    missing_fonts = [str(font["base_font"]) for font in fonts if not bool(font["embedded"])]
    if missing_fonts:
        errors.append(f"unembedded fonts: {', '.join(missing_fonts)}")
    author_metadata = metadata.get("/Author", "").strip()
    if author_metadata:
        errors.append("PDF /Author metadata must be empty for anonymous review")
    text_hits = {
        token: [index + 1 for index, text in enumerate(lower_text) if token.casefold() in text]
        for token in forbidden_text
    }
    text_hits = {token: pages for token, pages in text_hits.items() if pages}
    if text_hits:
        compact = "; ".join(f"{token} on page(s) {pages}" for token, pages in text_hits.items())
        errors.append(f"project-identifying text recovered from PDF: {compact}")

    return {
        "pdf": str(path.resolve()),
        "page_count": len(reader.pages),
        "expected_pages": expected_pages,
        "table_i_on_page_2": len(page_text) >= 2 and "table i" in lower_text[1],
        "references_on_page_3": len(page_text) >= 3 and "references" in lower_text[2],
        "page_sizes_points": [[width, height] for width, height in page_sizes],
        "us_letter_geometry": all(_is_us_letter(size) for size in page_sizes),
        "font_count": len(fonts),
        "fonts": fonts,
        "metadata": metadata,
        "forbidden_text_hits": text_hits,
        "errors": errors,
        "ok": not errors,
        "scope_note": "Local static readiness check only; not an official ICASSP template or PDF eXpress validation.",
    }
