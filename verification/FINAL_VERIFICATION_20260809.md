# Final local verification — 2026-08-09

Verification time: 2026-08-09T23:31Z. This records the final local state of
the initial research package. It does not claim an external paper review,
template approval, remote push, or Telegram delivery.

## Test suite

Executed from the repository root:

```bash
PYTHONPATH=. python3 -m pytest -q
```

Initial verification passed **79 tests in 6.45s**. After adding the static PDF
preflight, the complete suite passed **82 tests in 6.47s**; after its US-letter
geometry extension, it again passed **82 tests in 6.48s**. After the
single-anonymous submission-stage mode and author-block handoff, it passed
**83 tests in 6.47s**.
`git diff --check` also completed without whitespace errors.

## Paper build

The manuscript was rebuilt from source after the author-block status correction:

```bash
cd paper
tectonic --keep-intermediates --outdir build main.tex
```

Result: success. The refreshed, tracked readable artifact is
`paper/build/main.pdf` with SHA-256
`8e1ca620048f057bf88f1e492004b598dc7db2048e1277c6ea6fb8e678dfb9bc`.
`pypdf` extraction verifies four pages, Table I on page 2, and References
starting on page 3. The generated bibliography has seven `\bibitem` entries;
the build has no unresolved-citation marker.

The reproducible static check below passes on the internal anonymous working
draft. It verifies embedded fonts, an empty `/Author` field, and no selected
project-identifying text in addition to the page/layout landmarks; it is
explicitly not an IEEE PDF eXpress or official ICASSP-template replacement.

```bash
PYTHONPATH=. python3 scripts/check_paper_pdf.py --pdf paper/build/main.pdf
```

The passing report records four US-letter MediaBoxes, 16 embedded font
resources, no forbidden-text hit, and no local readiness error. ICASSP 2027
uses single-anonymous review, so this anonymous state is only an internal
working-draft check; `paper/AUTHOR_BLOCK_REQUIRED.md` governs the required
author insertion and subsequent submission-stage preflight. The same current
placeholder PDF also passes that submission-stage layout/font mode; it is not
considered upload-ready until real author information is inserted.

## Research boundary checks

- H1’s frozen Spectra-AASIST/full-waveform/spoof crest candidate is not
  five-corpus portable; it must not be promoted to causal evidence.
- H2 failed its locked pre-score quality gate and has no paired detector-score
  outcome. H2B Q1 likewise selected no quality-eligible non-control family.
- H4 is a sealed, score-free descriptive label--cue atlas. Its 23 terminal
  associations and supplementary heatmap cannot select an H1/H2/H3 candidate.

## External handoff conditions

The official ICASSP page limit/dates, single-anonymous author requirement, and
generic working-template caveat are recorded in `paper/submission-requirements.md`.
The local paper-review launcher was most recently executed against the
post-author-policy PDF, but the Claude CLI was unauthenticated, so no reviewer
verdict exists. Remote push and Telegram notifications require credentials not
present in the environment; queued user-facing updates are in
`to_human/pending-notifications.md`.
