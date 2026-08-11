# Post-H10 paper-integrity check — 2026-08-11

**Purpose:** record why the H10 terminal follow-up did not modify the current
H9 manuscript, and recheck that the readable paper remains internally
consistent after the separate H10 branch completed.

## Decision boundary

H10 is a predeclared source-frozen CD-ADD extension of the sealed H9
checkpoint panel. Its P ensemble has lower EER and below-zero paired bootstrap
intervals against both controls, but its 8.30% relative EER reduction against
B1 misses H10's locked 10% practical-effect requirement. The H10 decision gate
is false; its independent red-team audit finds no invalidating defect.

Accordingly, `paper/main.tex`, `paper/build/main.pdf`, Figure 1, tables,
abstract, title, references, and H9 claims were intentionally **not** edited
for H10. The manuscript continues to state only the passing H9 SONAR+ArAD
controlled-transfer result. This is not selective threshold relaxation: H10's
complete result, hash ledger, stop rule, and audit are retained in the
repository/HDD at the locations below.

## Checked artifacts

| Item | Result |
| --- | --- |
| H10 protocol/result/audit | `experiments/future_directions/H10_CDADD_EXTERNAL_REPLICATION_PROTOCOL.md`, `results/H10_CDADD_TERMINAL_EVALUATION_001.md`, and `reviews/H10_CDADD_TERMINAL_RED_TEAM_AUDIT_20260811.md` preserve the terminal stop. |
| H10 code contract tests | `PYTHONPATH=. python3 -m pytest -q tests/test_h10_cdadd_adapter.py tests/test_h9_pcr_target_materialize.py tests/test_h9_pcr_evaluation.py` → 13 passed. |
| H9 paper PDF | `paper/build/main.pdf`, SHA-256 `bebc23b48478daa7e1262f1a764fde4e6166e63bac1ee153c37bc73d590726f6`. |
| PDF static preflight | Passed in `single-anonymous-submission` mode: five US-letter pages, Table 1 on page 2, references on page 5, and 13 embedded fonts. |
| H10 heavyweight outputs | Retained only on the designated HDD under `runs/h10_cdadd/`; they are hash-pinned by the H10 result note and never staged. |

## Future boundary

A future architecture replication requires a fresh protocol, a fresh
architecture/source-only ledger, and targets not already observed by H9/H10.
It may not use CD-ADD to choose a method or a target panel after H10's result.
Until that work completes, the correct ICASSP draft is the narrow H9 paper,
not a three-target or architecture-general claim.
