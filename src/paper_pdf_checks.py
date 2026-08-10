"""Static, reproducible checks for the ICASSP working-draft PDF.

This module is intentionally a *local readiness check*, not a replacement for
an ICASSP-provided template, PDF eXpress, or conference submission checker.
It verifies only properties that can be recovered from a built PDF: expected
page count, essential layout landmarks, embedded fonts, and metadata. Identity
checks are enabled only for the repository's anonymous *internal* draft. ICASSP
2027 itself uses single-anonymous review, so a final submission must include
the real author block and should use the submission-stage mode.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from pypdf import PdfReader


DEFAULT_FORBIDDEN_TEXT = ("lab260", "kirill", "nikita", "ivan")
US_LETTER_WIDTH_POINTS = 612.0
US_LETTER_HEIGHT_POINTS = 792.0
PAGE_SIZE_TOLERANCE_POINTS = 0.5
TABLE_LABEL_PATTERN = re.compile(r"\btable\s+(?:1|i)\b", flags=re.IGNORECASE)


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
    expected_pages: int = 5,
    table_page: int = 3,
    references_page: int | None = None,
    forbidden_text: tuple[str, ...] = DEFAULT_FORBIDDEN_TEXT,
    anonymous_working_draft: bool = True,
) -> dict[str, object]:
    """Inspect a built working draft and return a serializable report.

    The caller owns any submission-policy decision. A report with ``ok=False``
    is a local failure that should be corrected before treating the PDF as a
    readable handoff artifact. The default landmarks match the current draft:
    Table 1 on page 3, the atlas on page 4, and a references-only fifth page.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Paper PDF does not exist: {path}")
    if expected_pages < 1 or table_page < 1:
        raise ValueError("expected_pages and table_page must be positive")
    if references_page is None:
        references_page = expected_pages
    if references_page < 1:
        raise ValueError("references_page must be positive")

    reader = PdfReader(str(path))
    page_text = [(page.extract_text() or "") for page in reader.pages]
    lower_text = [text.casefold() for text in page_text]
    page_sizes = [_page_size_points(page) for page in reader.pages]
    fonts = _embedded_font_summary(reader)
    metadata = {str(key): str(value) for key, value in dict(reader.metadata or {}).items()}
    errors: list[str] = []
    if len(reader.pages) != expected_pages:
        errors.append(f"expected {expected_pages} pages but found {len(reader.pages)}")
    table_on_expected_page = (
        len(page_text) >= table_page
        and TABLE_LABEL_PATTERN.search(page_text[table_page - 1]) is not None
    )
    if not table_on_expected_page:
        errors.append(f"Table 1/I is not recoverable from extracted text on page {table_page}")
    references_on_expected_page = (
        len(page_text) >= references_page and "references" in lower_text[references_page - 1]
    )
    if not references_on_expected_page:
        errors.append(f"References are not recoverable from extracted text on page {references_page}")
    if not all(_is_us_letter(size) for size in page_sizes):
        observed = ", ".join(f"{width:.1f}x{height:.1f}" for width, height in page_sizes)
        errors.append(f"all PDF pages must have US Letter MediaBox 612.0x792.0 pt; observed {observed}")
    missing_fonts = [str(font["base_font"]) for font in fonts if not bool(font["embedded"])]
    if missing_fonts:
        errors.append(f"unembedded fonts: {', '.join(missing_fonts)}")
    text_hits = {
        token: [index + 1 for index, text in enumerate(lower_text) if token.casefold() in text]
        for token in forbidden_text
    }
    text_hits = {token: pages for token, pages in text_hits.items() if pages}
    if anonymous_working_draft:
        author_metadata = metadata.get("/Author", "").strip()
        if author_metadata:
            errors.append("PDF /Author metadata must be empty for anonymous working-draft review")
        if text_hits:
            compact = "; ".join(f"{token} on page(s) {pages}" for token, pages in text_hits.items())
            errors.append(f"project-identifying text recovered from anonymous working draft: {compact}")

    return {
        "pdf": str(path.resolve()),
        "page_count": len(reader.pages),
        "expected_pages": expected_pages,
        "table_page": table_page,
        "references_page": references_page,
        "table_1_or_i_on_expected_page": table_on_expected_page,
        "references_on_expected_page": references_on_expected_page,
        "page_sizes_points": [[width, height] for width, height in page_sizes],
        "us_letter_geometry": all(_is_us_letter(size) for size in page_sizes),
        "font_count": len(fonts),
        "fonts": fonts,
        "metadata": metadata,
        "anonymous_working_draft": anonymous_working_draft,
        "identity_checks_applied": anonymous_working_draft,
        "forbidden_text_hits": text_hits,
        "errors": errors,
        "ok": not errors,
        "scope_note": (
            "Local static readiness check only; not an official ICASSP template or PDF eXpress validation. "
            "ICASSP 2027 uses single-anonymous review, so final submissions require real author information."
        ),
    }
