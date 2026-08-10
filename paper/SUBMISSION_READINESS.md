# Static paper-submission readiness check

The repository provides a narrow, reproducible check for the readable working
draft. It verifies US-letter geometry, exact layout landmarks, and embedded
fonts. The current named-author draft has four technical pages followed by a
references-only fifth page, permitted by the stated ICASSP page policy, and
uses the single-anonymous submission stage.

Run it after compiling the manuscript:

~~~bash
cd /home/kirill/icassp_antispoofing
PYTHONPATH=. python3 scripts/check_paper_pdf.py \
  --pdf paper/build/main.pdf \
  --review-stage single-anonymous-submission
~~~

A passing report has five US-letter pages, recovers the exact Table 1/I label
from page 3 and References from page 5, and reports all fonts embedded. The
fifth page is reserved for references by the manuscript source. The command
emits JSON so its result can be stored with a future submission package.

ICASSP 2027 is single-anonymous: reviewers know author names. The authorized
working metadata is recorded in AUTHOR_BLOCK_REQUIRED.md; this stage retains
geometry/layout/font checks without requiring anonymous author metadata.

## Boundaries

This is not a replacement for the official ICASSP 2027 template, the
conference submission portal, or IEEE PDF eXpress. Re-run this check, compile
main.tex, and use the then-current official conference checker after
reconciling the final template and author block.
