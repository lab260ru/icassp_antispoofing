# Final local verification — 2026-08-09

Verification time: 2026-08-09T23:31Z. This records the final local state of
the initial research package. It does not claim an external paper review,
template approval, remote push, or Telegram delivery.

## Test suite

Executed from the repository root:

```bash
PYTHONPATH=. python3 -m pytest -q
```

Result: **79 passed in 6.45s**. `git diff --check` also completed without
whitespace errors.

## Paper build

The ignored TeX intermediates under `paper/build/` were removed, then the
unchanged manuscript was rebuilt from source:

```bash
cd paper
tectonic --keep-intermediates --outdir build main.tex
```

Result: success. The refreshed, tracked readable artifact is
`paper/build/main.pdf` with SHA-256
`6de0e0c34a673f483baea0b4f090158da6b770da6bf33ad14f1e2c01bd4ebec8`.
`pypdf` extraction verifies four pages, Table I on page 2, and References
starting on page 3. The generated bibliography has seven `\bibitem` entries;
the build has no unresolved-citation marker.

## Research boundary checks

- H1’s frozen Spectra-AASIST/full-waveform/spoof crest candidate is not
  five-corpus portable; it must not be promoted to causal evidence.
- H2 failed its locked pre-score quality gate and has no paired detector-score
  outcome. H2B Q1 likewise selected no quality-eligible non-control family.
- H4 is a sealed, score-free descriptive label--cue atlas. Its 23 terminal
  associations and supplementary heatmap cannot select an H1/H2/H3 candidate.

## External handoff conditions

The official ICASSP page limit/dates and the generic working-template caveat
are recorded in `paper/submission-requirements.md`. The local paper-review
launcher was executed but the Claude CLI was unauthenticated, so no reviewer
verdict exists. Remote push and Telegram notifications require credentials not
present in the environment; queued user-facing updates are in
`to_human/pending-notifications.md`.
