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
- Keep the review version anonymous and verify author metadata before upload.
- Preserve the exact class/BST sources used for every paper build in this
  repository.
- Run the final PDF through the then-current IEEE/ICASSP compliance checker;
  the conference editorial policy says submissions receive template and
  paper-length checks.

## Local static readiness check

`scripts/check_paper_pdf.py` supplies a reproducible, deliberately narrower
preflight for the current anonymous working draft. It confirms the four-page
US-letter target, Table I/References landmarks, embedded fonts, empty `/Author`
metadata, and no selected project-identifying strings in rendered text. The
current PDF passes this check; usage and its limits are in
`SUBMISSION_READINESS.md`.
