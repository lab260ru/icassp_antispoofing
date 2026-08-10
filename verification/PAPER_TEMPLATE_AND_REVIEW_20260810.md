# Named-author template and review verification — 2026-08-10

This verification records the current named-author ICASSP working draft. It
does not claim official ICASSP-2027 template approval, portal acceptance, or
external peer review.

## User-provided template inputs

The user-provided ICASSP2026_Paper_Templates.zip archive is archived on the
HDD and has SHA-256
`a3a2b007568d545caa8a264358831c86efb67a6ef3db2e27f2cde502e1fb29a1`.
The tracked build inputs are paper/template/ICASSP2026/spconf.sty
(`dc5d632639040cb183f2ab62780f314845aa021be73048c8a9ec9c2072d64a86`) and
paper/template/ICASSP2026/IEEEbib.bst
(`7e6ca0c8b72158d504a12bb091f817c07032a021ba41ca783cca4c2dd80d570b`).

After clearing only rebuildable main.aux/main.bbl intermediates, a
keep-intermediates build reported
`\\bibstyle{template/ICASSP2026/IEEEbib}`. Those ignored intermediates were
then removed again.

## Paper build and local preflight

Executed from paper/:

~~~bash
tectonic --outdir build main.tex
~~~

The readable tracked artifact is paper/build/main.pdf with SHA-256
`12775049b4321682a60a913c70d169c5983b5e87f7d91604704676df85289dca`.
The final source uses the authorized author metadata, a literal all-caps title,
and the vendored spconf/IEEEbib pair.

The named-author preflight passed:

~~~bash
PYTHONPATH=. python3 scripts/check_paper_pdf.py \
  --pdf paper/build/main.pdf \
  --review-stage single-anonymous-submission
~~~

It verifies five US-letter pages, the exact Table 1/I label on page 3,
References on page 5, and ten embedded font resources. Pages 1--4 contain
technical material; page 5 is references-only. The preflight is local only,
not IEEE PDF eXpress or official ICASSP certification.

## Tests and review

`PYTHONPATH=. python3 -m pytest -q` passed 87 tests in 6.45 seconds.
The three fresh Codex-only reviews and their resolution record are in
paper/reviews/codex_post_template_20260810/. They are internal reviews, not
conference peer review. All source/layout corrections judged actionable for
this checkpoint were applied before the final build and preflight.
