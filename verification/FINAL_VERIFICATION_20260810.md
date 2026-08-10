# Final verification checkpoint — 2026-08-10

This rolling checkpoint was refreshed at **2026-08-10 19:35 UTC**. It verifies
the reviewed readable paper, complete test suite, sealed supplementary displays,
and offline recovery plan. It does not reopen a research gate or claim an
official conference submission check.

## Local verification

| Check | Command / result |
| --- | --- |
| Full repository suite | `PYTHONPATH=. python3 -m pytest -q` — **149 passed** in 27.46 s |
| Worktree safety | `git diff --check` passed; the only intentionally untracked item is the user-provided `ICASSP2026_Paper_Templates.zip` archive |
| Paper PDF | `paper/build/main.pdf`, SHA-256 `4395f1e131e74e926c475a43c7a965a51ee46dcc73387dbf4e9166120b7fdd4c` |
| Paper preflight | `PYTHONPATH=. python3 scripts/check_paper_pdf.py --pdf paper/build/main.pdf --review-stage single-anonymous-submission` — five US-letter pages, Table 1 page 3, references page 5, 10 embedded fonts, no errors |
| H6 display | exact sealed H6 compact inputs, vector PDF plus 300-DPI PNG and metadata; all recorded in `experiments/h6_score_agreement/results/H6_SUPPLEMENTARY_FIGURE_001.md` |
| H7 display | exact sealed H7 compact inputs, vector PDF plus 300-DPI PNG and metadata; its Codex-only claim/provenance/visual audit accepts it as supplementary-only, recorded in `experiments/h7_feature_transfer/results/h7_analysis_001/H7_SUPPLEMENTARY_FIGURE_001.md` and `paper/reviews/codex_h7_supplement_20260810/review.md` |
| Main Figure 1 v2 | exact eight H1/H2 compact inputs, reviewed vector PDF plus 300-DPI PNG and metadata; `paper/reviews/codex_main_figure_20260810/v2_verification.md` passes |
| H1 adjustment disclosure | exact five projected metadata sources, 10 corpus-by-label coverage rows, source hashes, and the already sealed confirmation-bootstrap context; no score/feature read or bootstrap rerun |
| Offline bundle | `git bundle verify` passed for the complete-history 9,272,188-byte HDD bundle at head `8892b7931344b79351b8b1e22d006be92b574456`, documented in `to_human/GIT_BUNDLE_HANDOFF_20260810.md` |
| Git delivery | Local `8892b7931344b79351b8b1e22d006be92b574456` equaled `origin/research/icassp-signal-audit` at verification time; only the intentionally untracked template ZIP was present |

## Scientific boundary retained

- H1's discovery-frozen crest candidate fails the five-corpus portability rule.
- H2 and H2B stop before detector scoring; H3 is not run.
- H4/H5/H6/H7 are sealed descriptive diagnostics only.
- The paper makes no causal feature-reliance or mitigation-performance claim.

The local preflight is not the official ICASSP-2027 template/checker or IEEE
PDF eXpress. Those remain submission-stage external requirements.
