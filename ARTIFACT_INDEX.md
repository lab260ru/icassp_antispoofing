# Artifact index and continuation map

This is the compact entry point for the complete research record. It identifies
the authoritative protocol and interpretation note for each completed loop;
those notes, rather than this index, hold the detailed hash ledgers and
statistics. Large data, models, checkpointed pair tables, and full logs are
kept below `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/` and must never be
committed to this repository.

## Start here

1. Read `AGENTS.md`, `research-state.yaml`, `findings.md`, and
   `research-log.md` in that order.
2. Check `git status --short --branch`, active jobs, and the referenced HDD
   directory before starting a task.
3. Read the governing protocol before its result note. Never reopen a sealed
   loop by changing a threshold, arm grid, sample cap, score direction, or
   candidate after seeing its outcome.

## Experiment record

| Loop | Governing protocol | Authoritative result / artifact | Current conclusion and hard boundary |
|---|---|---|---|
| H1 five-corpus association audit | `experiments/h1_feature_association/protocol.md` | `experiments/h1_feature_association/results/five_corpus_aggregate_20260809T214500Z/AGGREGATION_RUN_20260809.md` | The discovery-frozen `Spectra-AASIST/full_waveform/spoof/crest_factor_db` candidate is not portable. The 19/168 registry entries are descriptive only and cannot choose a new H2/H3 cue. |
| H2 waveform intervention screen | `experiments/h2_causal_interventions/protocol.md` plus `H2_QUALITY_GATE_CORRECTION.md` | `experiments/h2_causal_interventions/results/quality_runs/H2_QUALITY_RUN_002.md` | Three of four detector-free arms fail the locked 90% retained-pair gate. No score-eligible manifest or transformed-detector score exists; no causal claim is permitted. |
| H2B independent quality-first loop | `experiments/future_directions/H2B_QUALITY_FIRST_OUTER_LOOP_PROTOCOL.md` | `experiments/future_directions/results/H2B_Q0_RUN_001.md` and `H2B_Q1_RUN_001.md` | The 256-identity, 9-arm DeepVoice Q1 run selected no non-control family. Q2--Q4 remain closed; do not tune its grid or Wilson threshold. |
| H4 score-free transportability atlas | `experiments/h4_label_transportability/protocol.md` | `experiments/h4_label_transportability/results/H4_INPUT_FREEZE_001.md` and `H4_ANALYSIS_001.md` | All 420 cells and 84 aggregations complete; 23 terminal descriptive units. H4 never reads scores/models/audio and cannot select a downstream candidate. |
| H5 score-free view-invariance atlas | `experiments/h5_view_invariance/protocol.md` | `experiments/h5_view_invariance/results/H5_INPUT_FREEZE_001.md` and `H5_ANALYSIS_001.md` | All 420 cells and 84 aggregations complete; 17 terminal descriptive units. Six unavailable InTheWild silence/clipping cells remain explicit; H5 cannot select a feature or alter H1--H4/H2B/H3. |
| H6 published-score agreement atlas | `experiments/h6_score_agreement/protocol.md` | `experiments/h6_score_agreement/results/H6_INPUT_FREEZE_001.md` and `H6_ANALYSIS_001.md` | All 280 cells and 28 summaries complete with 5,000 exact joins/cell; substantial agreement heterogeneity is descriptive only and cannot select a model or alter other gates. |
| H3 feature-conditioned training | `experiments/h3_feature_conditioned_fix/protocol.md` | No run artifact exists by design. | H3 is not authorized because no causal H2 result exists. Any future training must use BF16 and a new, valid causal prerequisite. |
| Post-handoff roadmap | `experiments/future_directions/POST_20260810_RESEARCH_PLAN.md` | No result artifact exists by design. | New work must use an independent protocol/freeze and cannot repair H1/H2/H2B/H3 after outcomes. |

## Model and data provenance

- Dataset/model revisions and response-artifact metadata: `data/arena-index.yaml`.
- Feature definitions: `docs/feature_registry_v1_28.md`.
- H2 runner/parity and throughput details: `experiments/h2_causal_interventions/PARITY_RUN_001.md`, `RES2TCNGUARD_ONNX_DIAGNOSIS.md`, `H2_THROUGHPUT_RUN_001.md`, and `H2_THROUGHPUT_RUN_002_RES2TCNGUARD.md`.
- A future fourth scorer must follow the exact pin and parity process in `experiments/h2_causal_interventions/W2V2_AASIST_READINESS.md`; it is not currently a registered runner.
- Every completed large-table result above contains its own explicit HDD location,
  byte count, and SHA-256 ledger. Re-hash before using an HDD artifact.

## Paper and review record

| Item | Location | Status |
|---|---|---|
| Named-author source / readable PDF | `paper/main.tex`, `paper/build/main.pdf` | Current initial draft has four technical pages plus a references-only fifth page and contains authorized working metadata for the single-anonymous policy. Its reviewed v2 Figure 1 directly shows the H1 crest boundary and one-corpus H2 quality stop; see `paper/figures/FIGURE_CREST_EVIDENCE_BOUNDARY_MAIN.md`. Confirm portal metadata before upload; see `paper/AUTHOR_BLOCK_REQUIRED.md`. |
| Requirements and template status | `paper/submission-requirements.md`, `paper/template/ICASSP2026/TEMPLATE_PROVENANCE.md` | The user-provided ICASSP-2026 spconf/BST build inputs are pinned with archive hash. Reconcile with the official ICASSP-2027 kit before submission. |
| Static local PDF preflight | `paper/SUBMISSION_READINESS.md`, `scripts/check_paper_pdf.py` | Current named-author PDF passes five-page US-letter, exact Table 1/References landmark, and font checks in single-anonymous-submission mode. This is not IEEE PDF eXpress or an official template check. |
| Pre-deadline objective audit | `verification/PREDEADLINE_GOAL_AUDIT_20260810.md` | Requirement-by-requirement evidence ledger for the active 20:00 UTC handoff; explicitly separates completed work from quality-gated and external submission non-completions. |
| Citation ledger | `paper/citation-verification.md` | Eight cited records are mapped to bounded claims. |
| External paper-review attempts | `paper/reviews/*/REVIEW_STATUS.md` | Four timestamped panels were launched, including against the current author-policy draft, but all stopped before review because the local Claude CLI is unauthenticated. No review verdict exists. |
| Codex-only internal reviews | `paper/reviews/codex_post_template_20260810/` | Three fresh methods/template/presentation reviews and a resolution record for the named-author ICASSP-2026-template draft. This is internal technical review, not conference peer review. |
| Final Codex-only audit | `paper/reviews/codex_final_audit_20260810/` | Three independent review perspectives, meta-review, concern matrix, and evidence-safe resolution for the current compiled working draft. H5/H6/S1 remain supplementary-only. |
| Codex-only Figure 1 review | `paper/reviews/codex_main_figure_20260810/` | Three independent reviews plus a v2 layout verification. The resolved main figure uses only H1/H2 hash-pinned inputs and makes no causal or cross-panel claim. |
| H1 adjustment disclosure | `experiments/h1_feature_association/results/covariate_availability_audit_001/` | Score-free table of exact duration/loudness/speaker/attack/source metadata coverage for the five H1 cohorts, plus sealed held-out bootstrap context. It computes no association or new selection decision. |
| H7 feature-only transfer audit | `experiments/h7_feature_transfer/protocol.md`, `experiments/h7_feature_transfer/results/h7_analysis_001/` | Complete fixed-recipe, score-free five-cell leave-one-corpus-out label-transfer matrix over all 28 features. It is descriptive only and cannot identify a detector cue, causal effect, or mitigation improvement. |
| H1 paper-supplement source | `paper/supplementary/H1_ADJUSTMENT_DISCLOSURE_001.md` | Hash-linked display-only table for a future authorized supplement package; it copies no score-dependent result and makes no claim that a public archive or conference supplement exists. |
| Supplementary crest evidence boundary | `experiments/paper_extension/results/crest_evidence_boundary_s1_001/` | Fixed-input S1 PDF/PNG shows the sealed H1 crest slice, H2 quality stop, and score-free H4 label context. It is descriptive only and does not alter any research gate. |
| Supplementary H5 concordance heatmap | `experiments/h5_view_invariance/H5_SUPPLEMENTARY_CONCORDANCE_FIGURE_PROTOCOL.md` | `experiments/h5_view_invariance/results/H5_SUPPLEMENTARY_CONCORDANCE_FIGURE_001.md` | Hash-validated display of all 84 sealed H5 aggregates. The v2 layout is visually approved; unavailable/stable markers remain descriptive only. |
| Supplementary H6 agreement heatmap | `experiments/h6_score_agreement/H6_SUPPLEMENTARY_FIGURE_PROTOCOL.md` | `experiments/h6_score_agreement/results/H6_SUPPLEMENTARY_FIGURE_001.md` | Hash-validated fixed-order display of all 280 sealed H6 agreement cells. It is descriptive only and cannot rank/select models or revise a research gate. |
| Internal audits | `paper/reviews/five_corpus_h2_quality_20260809/` | Claim-to-artifact and scope/layout audits only; never call them peer review. |

## Reproduce the current local handoff

```bash
cd /home/kirill/icassp_antispoofing
PYTHONPATH=. python3 -m pytest -q
cd paper && tectonic --outdir build main.tex && cd ..
PYTHONPATH=. python3 scripts/check_paper_pdf.py \
  --pdf paper/build/main.pdf \
  --review-stage single-anonymous-submission
```

The most recent full suite is 139 passed. `verification/FINAL_VERIFICATION_20260810.md` records the current PDF hash, static preflight, full test suite, reviewed main-figure revision, H6 supplementary display, and complete-history bundle evidence; rerun the commands after any relevant source change.

## External delivery and resume blockers

- GitHub push: after the user replaced the local .env token, a normal non-force push created the remote `research/icassp-signal-audit` branch on 2026-08-10; see `to_human/GITHUB_PUSH_RECEIPT_20260810.md`.
- Portable backup: `to_human/GIT_BUNDLE_HANDOFF_20260810.md` records the verified complete-history HDD bundle and restoration command. It is not a GitHub delivery.
- Telegram milestone delivery: the latest paper/push-blocker update was delivered on 2026-08-10; its redacted receipt is in `to_human/TELEGRAM_DELIVERY_LOG_20260810.md`. Historical messages remain queued and can be listed/sent one-at-a-time through `to_human/TELEGRAM_DELIVERY.md`.
- Paper review: use the user-requested Codex-only timestamped review bundle and address its findings; do not retry the unauthenticated external reviewer runtime.
- Template reconciliation: obtain an approved ICASSP-2027 or official generic IEEE archive from an accessible route, hash/compare it to the pinned files, rebuild, rerun the static preflight, and use the official conference checker.
- Author information: confirm the authorized working metadata and any portal-required email fields, then use `--review-stage single-anonymous-submission` before upload.

`to_human/FINAL_HANDOFF_20260810.md` supplies the concise user-facing
continuation plan. No external action should weaken or bypass the H1/H2/H2B/H4
negative-result stop rules.
