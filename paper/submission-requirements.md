# ICASSP 2027 submission requirements

Last checked: 2026-08-10 15:10 UTC.

This initial draft targets the regular ICASSP 2027 conference-paper track. The
official [publishing and paper-presentation page](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/)
states that a regular paper may contain up to four pages of technical content,
including figures and references, plus an optional fifth page containing
references only. The official [call for papers](https://2027.ieeeicassp.org/call-for-papers/)
lists the full-paper submission deadline as 2026-09-16.

## Review and author information

ICASSP 2027's official editorial policy states that papers use
single-anonymous review: reviewers know author names, but reviewers are
anonymous to authors and other reviewers. The current main.tex contains the
authorized working-draft author name and affiliations. Confirm author order,
affiliations, and any portal-required email field against the submission-system
metadata before upload; see AUTHOR_BLOCK_REQUIRED.md.

## Template status

The current draft uses the user-provided ICASSP2026_Paper_Templates.zip bundle,
pinned as paper/template/ICASSP2026/spconf.sty and IEEEbib.bst. Its archive
identity and extraction scope are recorded in
paper/template/ICASSP2026/TEMPLATE_PROVENANCE.md. This is a reproducible
ICASSP-2026 working format, not a claim that it is the eventual ICASSP-2027
kit. At the last check, the official 2027 publishing, call-for-papers, and
editorial-policy pages state the format requirement but do not expose a
downloadable 2027 paper-kit archive or LaTeX file. Do not substitute an
unverified template: reconcile the current pinned inputs only when the official
2027 archive is accessible, then compile and visually check the final PDF
again.

## Current compliance target

- Keep technical material within four pages; do not rely on a fifth page for
  anything other than references.
- Confirm author metadata before upload.
- Preserve the exact class and bibliography sources used for every paper build.
- Run the final PDF through the then-current ICASSP/IEEE checker.

## Local static readiness check

scripts/check_paper_pdf.py is a deliberately narrower preflight. It verifies
US-letter geometry, exact Table 1/References landmarks, and embedded fonts.
The current named-author build has four technical pages plus a references-only
fifth page and uses its single-anonymous-submission stage; usage and limits are
in SUBMISSION_READINESS.md.
