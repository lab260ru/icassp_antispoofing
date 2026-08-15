#!/usr/bin/env python3
"""The submitted PDFs must satisfy the properties a submission portal rejects on.

Everything else in this suite protects the paper's claims. This protects the
file. ICASSP's portal rejects on mechanical grounds that have nothing to do with
whether the work is any good: a font the reader's machine has to guess at, a
page size that is not US Letter, an encrypted file the reviewers' tooling cannot
open. Those failures arrive at upload time, which for us is the last hour before
a deadline, and they are silent until then -- a PDF with a missing font builds
green, renders correctly on the machine that made it, and looks fine to every
other check in here.

These were verified by hand once, at tag v1.9, and hand-verification does not
survive nine more commits. The paper has changed on every one of them.

What is checked, and why each one:

  * Every font embedded. A non-embedded font is substituted by the reader, which
    silently changes metrics -- a paper measured at five pages here becomes six
    somewhere else, and mathematics is the first thing to break.
  * US Letter, 612x792 points, within a point of tolerance. ICASSP specifies it;
    A4 is the usual accident and is 3mm narrower and 18mm taller.
  * Not encrypted. Tectonic does not encrypt, but a post-processing step could,
    and an encrypted file fails at the portal rather than here.

The supplement is held to the same standard. It is a submitted file too.

Usage:  python scripts/check_pdf_submittable.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pypdf

REPO = Path(__file__).resolve().parent.parent
DOCS = ("paper/build/main.pdf", "paper/supplementary/build/supp.pdf")
LETTER = (612.0, 792.0)
TOL = 1.0
EMBED_KEYS = ("/FontFile", "/FontFile2", "/FontFile3")


def font_report(reader: pypdf.PdfReader) -> tuple[int, list[str]]:
    """(fonts seen, names of fonts with no embedded programme)."""
    seen, missing = 0, []
    for pg in reader.pages:
        res = pg.get("/Resources")
        if res is None:
            continue
        fonts = res.get_object().get("/Font")
        if fonts is None:
            continue
        for f in fonts.get_object().values():
            fo = f.get_object()
            seen += 1
            name = str(fo.get("/BaseFont", "?"))
            # Type0 fonts carry the descriptor on their descendant.
            desc = fo.get("/FontDescriptor")
            if desc is None:
                kids = fo.get("/DescendantFonts")
                if kids is not None:
                    kid = kids.get_object()[0].get_object()
                    desc = kid.get("/FontDescriptor")
            if desc is None:
                missing.append(name)
                continue
            d = desc.get_object()
            if not any(k in d for k in EMBED_KEYS):
                missing.append(name)
    return seen, sorted(set(missing))


def main() -> int:
    bad = []
    for rel in DOCS:
        p = REPO / rel
        if not p.exists():
            bad.append(f"{rel}: not built")
            continue
        r = pypdf.PdfReader(str(p))
        if r.is_encrypted:
            bad.append(f"{rel}: encrypted")
        seen, missing = font_report(r)
        if missing:
            bad.append(f"{rel}: {len(missing)} font(s) not embedded: "
                       f"{', '.join(missing[:4])}")
        sizes = set()
        for pg in r.pages:
            b = pg.mediabox
            sizes.add((round(float(b.width), 1), round(float(b.height), 1)))
        offsize = [s for s in sizes
                   if abs(s[0] - LETTER[0]) > TOL or abs(s[1] - LETTER[1]) > TOL]
        if offsize:
            bad.append(f"{rel}: page size {offsize[0]} is not US Letter "
                       f"{LETTER}")
        if not bad or not any(rel in b for b in bad):
            print(f"  OK   {rel.split('/')[-1]}: {len(r.pages)} pp, {seen} font "
                  f"refs all embedded, US Letter, unencrypted")
    if bad:
        print(f"  FAIL {len(bad)} submission-blocking PDF problem(s):")
        for b in bad:
            print(f"       {b}")
        print("       These fail at the portal, not here, and the portal closes "
              "at the deadline.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
