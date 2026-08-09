# ICASSP 2027 submission requirements

Last checked: 2026-08-09 23:31 UTC.

This initial draft targets the regular ICASSP 2027 conference-paper track.
The official [publishing and paper-presentation page](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/)
states that a regular paper may contain **up to four pages of technical
content, including figures and references**, plus an optional **fifth page
containing references only**. The official [call for papers](https://2027.ieeeicassp.org/call-for-papers/)
lists the full-paper submission deadline as **2026-09-16**, with notification
on 2027-01-13, final-paper submission on 2027-01-27, and author registration
by 2027-02-10.

## Review and author information

ICASSP 2027's official editorial policy states that papers use
**single-anonymous review**: reviewers know the author names, but reviewers
are anonymous to authors and other reviewers. The current `main.tex` author
block is an intentionally anonymous internal placeholder because no authorized
author metadata has been supplied. It must be replaced with the confirmed names,
affiliations, and order before upload; see `AUTHOR_BLOCK_REQUIRED.md`. Do not
upload the present anonymous working PDF as an ICASSP 2027 submission.

## Template status

`paper/template/IEEEtran/` contains the pinned IEEE conference class and
bibliography style used to compile the current anonymous draft. It is a
reproducible working template, not represented as an ICASSP-2027-specific
archive. The official author page describes paper-format templates but, at the
time above, did not expose a downloadable ICASSP-2027 LaTeX bundle through its
public author navigation. Before submission, replace or reconcile this pinned
copy with the official archive if IEEE posts one, then compile and visually
check the final PDF again. This is an external submission-readiness condition,
not evidence that the working IEEEtran copy is the eventual conference kit.

The generic official IEEE conference-template ZIP is linked from IEEE's
template page at
`https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-latex-template.zip`.
On 2026-08-09, a direct authenticated-free download attempt from this runtime
received an HTTP 202 CloudFront WAF challenge (`x-amzn-waf-action: challenge`),
so no ZIP was saved and the pinned files were not silently replaced. Retry from
an approved browser/network or obtain the archive from the conference when its
ICASSP-specific author bundle is published; hash and compare it before use.

## Current compliance target

- Keep technical material within four pages; do not rely on a fifth page for
  anything other than references.
- Replace the internal placeholder with verified author names, affiliations,
  and order before upload; check those fields against the submission-system
  metadata.
- Preserve the exact class/BST sources used for every paper build in this
  repository.
- Run the final PDF through the then-current IEEE/ICASSP compliance checker;
  the conference editorial policy says submissions receive template and
  paper-length checks.

## Local static readiness check

`scripts/check_paper_pdf.py` supplies a reproducible, deliberately narrower
preflight for both the current anonymous working draft and a future
single-anonymous submission. It confirms the four-page US-letter target, Table
I/References landmarks, and embedded fonts; its anonymous-working-draft mode
also checks empty `/Author` metadata and selected project-identifying strings.
The current PDF passes the internal mode; usage and its limits are in
`SUBMISSION_READINESS.md`.
