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
| H3 feature-conditioned training | `experiments/h3_feature_conditioned_fix/protocol.md` | No run artifact exists by design. | H3 is not authorized because no causal H2 result exists. Any future training must use BF16 and a new, valid causal prerequisite. |

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
| Internal placeholder source / readable PDF | `paper/main.tex`, `paper/build/main.pdf` | Current initial draft is four pages, but its anonymous block is not upload-ready: ICASSP 2027 uses single-anonymous review. An authorized author must follow `paper/AUTHOR_BLOCK_REQUIRED.md`, then compile and commit the refreshed PDF. |
| Requirements and template status | `paper/submission-requirements.md` | The official ICASSP page limit is recorded. The official generic IEEE ZIP is known but this runtime receives a CloudFront WAF challenge; no unverified replacement was made. |
| Static local PDF preflight | `paper/SUBMISSION_READINESS.md`, `scripts/check_paper_pdf.py` | Current internal PDF passes four-page US-letter, layout, font, and anonymous-draft checks. Submission mode retains the geometry/layout/font checks after authors are added. This is not IEEE PDF eXpress or an official template check. |
| Citation ledger | `paper/citation-verification.md` | Seven cited records are mapped to bounded claims. |
| External paper-review attempts | `paper/reviews/*/REVIEW_STATUS.md` | Three timestamped panels were launched, but all stopped before review because the local Claude CLI is unauthenticated. No review verdict exists. |
| Internal audits | `paper/reviews/five_corpus_h2_quality_20260809/` | Claim-to-artifact and scope/layout audits only; never call them peer review. |

## Reproduce the current local handoff

```bash
cd /home/kirill/icassp_antispoofing
PYTHONPATH=. python3 -m pytest -q
cd paper && tectonic --outdir build main.tex && cd ..
PYTHONPATH=. python3 scripts/check_paper_pdf.py \
  --pdf paper/build/main.pdf \
  --review-stage anonymous-working-draft
```

The most recent recorded full suite is 83 passed. `verification/FINAL_VERIFICATION_20260809.md` records the PDF hash and the full local build/test evidence; rerun the commands after any relevant source change.

## External delivery and resume blockers

- GitHub push: requires `GH_TOKEN` or `GITHUB_TOKEN` in the environment; do not put a credential in files, history, or commands.
- Telegram milestone delivery: requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; messages are queued in `to_human/pending-notifications.md`.
- Paper review: authenticate the local Claude CLI, then launch a new timestamped three-reviewer bundle and address any findings.
- Template reconciliation: obtain an approved ICASSP-2027 or official generic IEEE archive from an accessible route, hash/compare it to the pinned files, rebuild, rerun the static preflight, and use the official conference checker.
- Author information: provide authorized author names, affiliations, and order; replace the internal placeholder, then use `--review-stage single-anonymous-submission` before upload.

`to_human/FINAL_HANDOFF_20260809.md` supplies the concise user-facing
continuation plan. No external action should weaken or bypass the H1/H2/H2B/H4
negative-result stop rules.
