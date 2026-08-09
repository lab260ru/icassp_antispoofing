# Static paper-submission readiness check

The repository provides a narrow, reproducible check for the readable working
draft. It verifies the four-page US-letter target, key layout landmarks, and
embedded fonts. The current internal placeholder draft also checks empty
`/Author` metadata and selected project-identifying text.

Run it after compiling the manuscript:

```bash
cd /home/kirill/icassp_antispoofing
PYTHONPATH=. python3 scripts/check_paper_pdf.py \
  --pdf paper/build/main.pdf \
  --review-stage anonymous-working-draft
```

For the current anonymous initial draft, a passing report has four US-letter
pages, recovers Table I from page 2 and References from page 3, and reports all
fonts embedded. The command emits JSON so its result can be stored with a
future submission package.

ICASSP 2027 is **single-anonymous**: reviewers know author names. Once an
authorized author replaces the placeholder block, use
`--review-stage single-anonymous-submission`; this retains all geometry/layout/
font checks but does not wrongly require anonymous author metadata. See
`AUTHOR_BLOCK_REQUIRED.md`.

## Boundaries

This is not a replacement for the official ICASSP 2027 template, the conference
submission portal, or IEEE PDF eXpress. The official archive was not publicly
available through the ICASSP author navigation when last checked; see
`submission-requirements.md`. Re-run this check, compile `main.tex`, and use
the then-current official conference checker after reconciling the final
template and author block.
