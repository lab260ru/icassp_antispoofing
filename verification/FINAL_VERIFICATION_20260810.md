# Final verification checkpoint — 2026-08-10

This checkpoint verifies the reviewed readable paper, complete test suite,
sealed H6 display, and offline recovery artifact. It does not reopen a
research gate or claim an official conference submission check.

## Local verification

| Check | Command / result |
| --- | --- |
| Full repository suite | `PYTHONPATH=. python3 -m pytest -q` — **139 passed** in 26.87 s |
| Worktree safety | `git diff --check` passed; the only intentionally untracked item is the user-provided `ICASSP2026_Paper_Templates.zip` archive |
| Paper PDF | `paper/build/main.pdf`, SHA-256 `495bd3bb7d476915b94c592c9879f8d908ccf37892f255c993efb0fee3d8cdf0` |
| Paper preflight | `PYTHONPATH=. python3 scripts/check_paper_pdf.py --pdf paper/build/main.pdf --review-stage single-anonymous-submission` — five US-letter pages, Table 1 page 3, references page 5, 10 embedded fonts, no errors |
| H6 display | exact sealed H6 compact inputs, vector PDF plus 300-DPI PNG and metadata; all recorded in `experiments/h6_score_agreement/results/H6_SUPPLEMENTARY_FIGURE_001.md` |
| Main Figure 1 v2 | exact eight H1/H2 compact inputs, reviewed vector PDF plus 300-DPI PNG and metadata; `paper/reviews/codex_main_figure_20260810/v2_verification.md` passes |
| Offline bundle | `git bundle verify` passed for the complete-history HDD bundle documented in `to_human/GIT_BUNDLE_HANDOFF_20260810.md` |

## Scientific boundary retained

- H1's discovery-frozen crest candidate fails the five-corpus portability rule.
- H2 and H2B stop before detector scoring; H3 is not run.
- H4/H5/H6 are sealed descriptive diagnostics only.
- The paper makes no causal feature-reliance or mitigation-performance claim.

The local preflight is not the official ICASSP-2027 template/checker or IEEE
PDF eXpress. Those remain submission-stage external requirements.
