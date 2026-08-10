# Research Continuity Guide

## Mission and deadline

Conduct a reproducible audit of signal features in cross-dataset speech
anti-spoofing and prepare an ICASSP 2027 initial draft. The original draft
target was **2026-08-10 18:03 UTC**; the user explicitly authorized bounded,
evidence-preserving continuation through **2026-08-10 20:00 UTC**. The
provisional story is now a negative portability/quality-gate audit, not a
causal effect or feature-conditioned fix, unless a new independently frozen
protocol supplies the missing evidence.

## Resume order

1. Read `research-state.yaml`, `findings.md`, and the latest entry in
   `research-log.md`.
2. Inspect `git status --short --branch` and the active experiment's
   `protocol.md` before running anything.
3. Check active jobs and the HDD run directory before starting duplicates.
4. Record each material decision in `implementation-notes.md` and append the
   result to the active experiment's `analysis.md`.

## Current checkpoint — 2026-08-10, authoritative override

This section supersedes older ``Current checkpoint`` details below when they
conflict. All completed loops are hash-bound; do not reopen them by tuning a
threshold, sample, transform, scorer, feature, or model after seeing results.

- **H1 is terminal.** The sole discovery-frozen
  `Spectra-AASIST/full_waveform/spoof/crest_factor_db` association fails the
  fixed five-corpus portability rule. The 19/168 descriptive registry entries
  cannot select a new intervention or training cue. Read
  `experiments/h1_feature_association/results/five_corpus_aggregate_20260809T214500Z/AGGREGATION_RUN_20260809.md`.
- **H2 and H2B are terminal before scoring.** H2's detector-free retained-pair
  rates are DRC-3 18.1%, DRC-6 0.6%, gain 64.6%, and polarity 99.9%; no
  score-eligible manifest exists and no transformed detector score may be
  inferred. H2B Q1 selected no transform family (nearest lower Wilson bound
  0.89624 is below 0.90), so Q2--Q4 remain closed. H3 is consequently not
  authorized.
- **H4/H5/H6 are completed descriptive atlases only.** H4 has 420 label-cue
  cells/84 aggregates and 23 terminal descriptive units. H5 has 420
  score/label-free view-invariance cells/84 aggregates, 17 terminal
  descriptive units, and six explicit InTheWild degeneracies. H6 has 280
  within-class published-score agreement cells/28 fixed pair summaries, each
  with 5,000 exact joins; agreement ranges from -0.629261 to 0.868770. None
  can select a cue/model, rank a detector, establish causality, or alter
  H1--H3/H2B.
- **Paper handoff:** `paper/main.tex` and `paper/build/main.pdf` are the
  current readable named-author ICASSP-2026-template working draft. Commit
  `7d430c4` is the latest audited paper checkpoint; its local preflight passes
  (four technical US-letter pages, references-only fifth page, Table 1 and
  reference landmarks, embedded fonts) and the full test suite passes 115.
  The Codex-only audit bundle is
  `paper/reviews/codex_final_audit_20260810/`. H5, H6, and S1 are
  supplementary-only unless a new submission-package decision is recorded.
  **Every `paper/main.tex` edit must immediately run** `cd paper && tectonic
  --outdir build main.tex` **and commit the refreshed `paper/build/main.pdf`
  in the same paper change.**
- **Delivery and resume:** `origin/research/icassp-signal-audit` has been
  pushed through the H6 supplementary-handoff checkpoint and the last Telegram
  milestone is message ID 140, with a redacted receipt in
  `to_human/TELEGRAM_DELIVERY_LOG_20260810.md`. Use `git log` and
  `git status --branch` to confirm the exact remote/local head before resuming.
  Never stage the
  user-provided `ICASSP2026_Paper_Templates.zip` or any `.env` file. No active
  training, scoring, ASR, or download job is expected; check before assuming
  otherwise.
- **Only safe continuation:** archival documentation or a display-only,
  hash-validated supplementary rendering is allowed before the 20:00 UTC
  handoff. Any new causal, training, or feature-selection study requires a
  fresh independent protocol and freeze; it may not repair the H2/H2B gates.

## Current checkpoint — 2026-08-09

- **H1 runs 005--007 are complete on all three discovery corpora.**
  Their immutable source tables are under
  `experiments/h1_feature_association/results/<dataset>/`; each has 1,344
  within-class screen rows (8 models × 2 classes × 3 views × 28 features) from
  the locked 5,000-per-class subset. The three-input
  `results/discovery_aggregate_3datasets/` report correctly marks portability
  as not evaluable. Do not call any discovery result portable or causal.
- **H1 held-out confirmation and aggregate are complete.** InTheWild and
  ASVspoof5 have their own committed, 1,344-cell result directories and
  one-row manifest-selected bootstrap outputs. The explicit five-input
  aggregate is
  `results/five_corpus_aggregate_20260809T214500Z/`; it is selection-free and
  evaluates all 168 registry units. Nineteen units meet the configured
  descriptive association rule, but the sole discovery-frozen
  Spectra-AASIST/full-waveform/spoof/`crest_factor_db` candidate **does not**:
  it is adjusted-negative in InTheWild but adjusted-null and direction-reversed
  in ASVspoof5. Read `AGGREGATION_RUN_20260809.md`. This excludes a portable
  crest-factor claim. Do not add any of the 19 units to H2 after seeing these
  confirmation data; the existing crest H2 run is exploratory only.
- **Association estimator safety is tested.** Use
  `PYTHONPATH=. python3 scripts/analyze_associations.py ...`; the normal
  `pytest` console entry point may omit the repository from `sys.path`, so run
  `PYTHONPATH=. python3 -m pytest -q tests/test_statistics.py
  tests/test_audio_features.py`. The partial-correlation code deliberately
  removes a tested variable from its own controls. Bootstrap CIs require a
  frozen provenance-bearing manifest; see
  `experiments/h1_feature_association/BOOTSTRAP_CLI.md`.
- **H1 operational freeze:**
  `results/frozen_discovery_to_h2_20260809T201711Z/frozen_candidates.csv` is
  immutable and contains only three discovery provenance rows for
  Spectra-AASIST/`crest_factor_db`; see its adjacent report and
  `DISCOVERY_TO_H2_FREEZE.md`. It is exploratory, excludes confirmation data,
  and is not itself a causal or portable result. Commit it before any bootstrap
  or H2 execution. The held-out feature extractions and scoped H1 artifacts are
  now complete; do not relaunch them. Check processes and HDD paths before any
  new H2 work.
- **H2 has two parity-validated runners but is not ready to claim.** Read
  `experiments/h2_causal_interventions/model_capability_audit.md`,
  `H2_ONNX_PARITY.md`, and `PARITY_RUN_001.md` before any intervention work.
  Run ONNX commands through `scripts/run_h2_cuda.sh`; it exposes the active
  environment's CUDA 13 wheel libraries. AASIST uses raw input and
  Spectra-AASIST uses external pre-emphasis 0.97. Initial frozen-workload
  sweeps select AASIST batch 2 (GPU 1) and Spectra batch 8 (GPU 0); see
  `H2_THROUGHPUT_RUN_001.md`. The fixed ASR/WER runtime is now pinned in
  `H2_ASR_WER_RUNTIME.md` (OpenAI Whisper `small.en`, FP16 on physical GPU 3
  exposed as logical `cuda:0`); it has not transcribed or cleared a waveform
  quality gate. The source PyTorch Res2TCNGuard evaluator is additionally
  parity-validated; its bundled ONNX export is explicitly blocked (read
  `RES2TCNGUARD_ONNX_DIAGNOSIS.md`). H2 still requires execution of the
  pair-quality gate and a validated fourth model or a pre-frozen XLSR-SLS
  fallback. Score artifacts
  cannot score transformed audio. The waveform-only registered arms are in
  `src/h2_waveform_transforms.py`; they emit diagnostics but do not decide
  gates or supply any causal evidence. The first 1,000-row ASVspoof2019
  score-blind pair panel and its four-arm ledger are frozen in
  `results/pre_score/h2_crest_pre_score_001_20260809T205114Z/`; quality rows
  from a complete valid panel do not yet exist. `h2_quality_full_001` was
  deliberately stopped after 133/4,000 detector-free pair checkpoints because
  its original clipping gate rejected pre-existing source clipping; its HDD
  artifacts are preserved for diagnosis only and must not be analyzed as
  results. Commit `eb6298c` corrected the gate to assess transform-induced
  clipping. `h2_quality_full_002` completed all 4,000 detector-free pairs on
  GPU 3, but its formal quality freeze refused at the first arm: DRC-3 retains
  181/1,000 pairs (18.1%), DRC-6 6/1,000 (0.6%), and +0.1 dB gain 646/1,000
  (64.6%); only polarity reaches 99.9%. See
  `results/quality_runs/H2_QUALITY_RUN_002.md`. No score-eligible manifest
  exists; no detector may read a pair and no causal claim may be made. The
  committed
  `scripts/freeze_h2_quality_manifest.py` (read `H2_QUALITY_FREEZE.md`);
  would create a retained-only hash-sealed output only if a future new run
  clears every arm. Only that output may be passed to the committed
  `scripts/run_h2_paired_scoring.py` (read `H2_PAIRED_SCORING.md`). That scorer
  supports only Spectra, AASIST, and source-PyTorch Res2; no fourth scorer is
  implied. Read `H2_QUALITY_GATE_CORRECTION.md` before resuming H2. Res2
  source-PyTorch batch 2 is the recorded initial setting (198.09 clips/s); read
  `H2_THROUGHPUT_RUN_002_RES2TCNGUARD.md`.
- **Next H2 direction:** Read
  `experiments/future_directions/H2B_QUALITY_FIRST_OUTER_LOOP_PROTOCOL.md`
  before starting a new intervention. H2B is a separate score-blinded
  quality-feasibility study, not a repair of the failed H2 panel. Do not access
  scores/detectors during Q0--Q2, do not reuse the current five-corpus atlas
  to choose H2B's cue, and do not run Q4 without a fourth parity-validated
  scorer. Q0 and Q1 are complete: read
  `experiments/future_directions/results/H2B_Q0_RUN_001.md` and
  `experiments/future_directions/results/H2B_Q1_RUN_001.md`. Q1 completed all
  2,304 DeepVoice waveform/ASR-quality pairs but its locked selector chose no
  non-control family: the closest lower Wilson bound is 0.89624, below 0.90.
  **Do not change its thresholds/arms or start Q2--Q4 from this loop.** No
  H2B detector result exists.
- **H4 score-free atlas:** Read
  `experiments/h4_label_transportability/protocol.md`, then its input and run
  notes in `experiments/h4_label_transportability/results/`. The five-corpus,
  response-free `v1_28` analysis is complete: all 420 label-AUROC/bootstrap
  cells are valid and 23 of 84 `view × feature` units meet the **terminal,
  descriptive-only** H4 rule. Full-waveform crest reverses sign in InTheWild
  and is near null in ASVspoof5. Do not use the 23 units to select a detector
  cue, intervention, H1/H2/H2B/H3 candidate, or training experiment; do not
  tune the sealed sample cap, directions, thresholds, or bootstrap count. The
  compact hashes and the supplementary score-free heatmap are recorded in
  `results/H4_ANALYSIS_001.md`; large source/output tables remain on the HDD.
- **Fourth scorer handoff:** `experiments/h2_causal_interventions/W2V2_AASIST_READINESS.md`
  pins the only viable future route: download just
  `w2v2-aasist.onnx` from `SpeechAntiSpoofingBenchmarks/W2V2-AASIST` revision
  `196128e5a5101d5cb6ac7701597891bc7de7e7b5` to HDD, require SHA-256
  `837169def567cd68d94f7b5a6bd7ef55a7b64ea9cec61364944bcb66521e92d3`,
  inspect its graph, then perform CUDA parity before any scorer registration.
  Its advertised PyTorch/TensorRT wrappers are incomplete at that revision;
  never repair or substitute them ad hoc.
- **Paper draft:** `paper/main.tex` compiles with
  `tectonic --outdir build main.tex` from `paper/`. Track the source, citation
  ledger, and `paper/build/main.pdf`; ignore build intermediates. The compiled
  ICASSP-2026-spconf-format initial draft reports the completed five-corpus H1 analysis, its
  descriptive association atlas, and the H2 quality-gate failure, without a
  causal score claim. `paper/reviews/five_corpus_h2_quality_20260809/`
  contains an internal claim-to-artifact audit, not peer review. The
  spoof-class frozen crest candidate fails portability; a separate
  bona-fide/full-waveform crest slice appears in the descriptive atlas and
  must never be substituted into H2 post hoc.
  ICASSP 2027 uses **single-anonymous review**. The current source contains the
  authorized working metadata for Kirill Borodin with lab260 (Moscow, Russia)
  and BitmanagerAI (Dubai, UAE); read `paper/AUTHOR_BLOCK_REQUIRED.md` before
  changing it or uploading. The user-provided ICASSP 2026 spconf/BST inputs are
  pinned in `paper/template/ICASSP2026/`; their archive hash and the ICASSP-2027
  reconciliation requirement are recorded in `TEMPLATE_PROVENANCE.md`.
  **Whenever `paper/main.tex` changes, compile it successfully and update the
  committed `paper/build/main.pdf` in the same change.**
  `scripts/check_paper_pdf.py --pdf paper/build/main.pdf` is the reproducible
  local preflight: it confirms the present four-technical-page plus
  references-only-fifth-page US-letter layout, exact Table 1/References
  landmarks, and embedded fonts. `anonymous-working-draft` mode additionally
  checks blank `/Author` metadata and selected project-identity strings;
  `single-anonymous-submission` mode must be used after author insertion. It
  is deliberately narrower than the official ICASSP template or IEEE PDF
  eXpress; read `paper/SUBMISSION_READINESS.md` before treating it as a
  compliance signal.
  The external three-reviewer launcher was attempted for the initial draft,
  the five-corpus update, the post-scope/layout draft, and the post-author-policy
  draft, but was blocked because the local Claude CLI is unauthenticated;
  see `paper/reviews/initial_draft/REVIEW_STATUS.md` and
  `paper/reviews/five_corpus_h2_quality_20260809/REVIEW_STATUS.md`, and
  `paper/reviews/post_scope_layout_20260809/REVIEW_STATUS.md`, and
  `paper/reviews/post_author_policy_20260809/REVIEW_STATUS.md`. Do not pretend
  a review took place. The user now requires only Codex-agent reviews. The
  fresh named-author bundle is
  `paper/reviews/codex_post_template_20260810/`; preserve its internal-review
  status rather than retrying external reviewer tooling or calling it peer
  review.
  **Required workflow:** after every modification to `paper/main.tex`, run
  `cd paper && tectonic --outdir build main.tex`, verify a PDF was written,
  and commit the refreshed `paper/build/main.pdf` in the same paper commit.
- **Remote and Telegram:** the local ignored `.env` can provide runtime
  credentials. After the user replaced its GitHub token on 2026-08-10, a normal
  non-force push created the remote research/icassp-signal-audit branch; see
  `to_human/GITHUB_PUSH_RECEIPT_20260810.md`. The latest Telegram milestone was
  delivered successfully (receipt: `to_human/TELEGRAM_DELIVERY_LOG_20260810.md`).
  Never print, commit, recover, or reuse credentials from chat. Source only the
  current `.env` to push/send. Append milestones to
  `to_human/pending-notifications.md`; `scripts/send_pending_telegram.py --list` is network-free;
  with both Telegram variables present, `--send-latest` sends exactly one
  update and prints no credential. Read `to_human/TELEGRAM_DELIVERY.md` before
  delivery. The ICASSP page currently exposes no conference-specific
  LaTeX bundle. The official generic IEEE ZIP URL is documented in
  `paper/submission-requirements.md`, but direct download from this runtime gets
  a CloudFront HTTP-202 WAF challenge. The user provided an ICASSP-2026 archive,
  which is now pinned with provenance; do not substitute another unverified
  mirror, and reconcile with the 2027 kit before submission.
  A complete-history Git bundle is available on the HDD for offline recovery;
  its hash and restore command are in `to_human/GIT_BUNDLE_HANDOFF_20260809.md`.
  It does not replace the required eventual normal push.

## Artifact locations

- Repository: `/home/kirill/icassp_antispoofing`
- Large data/models/cache/checkpoints/logs:
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`
- Dataset/model provenance: `data/arena-index.yaml` and `data/README.md`
- Paper source and reviews: `paper/`
- Human updates and queued notifications: `to_human/`
- Cross-loop artifact map: `ARTIFACT_INDEX.md`

Do not commit raw audio, downloaded model weights, checkpoints, secrets, or
large runtime logs. Commit manifests, checksums, compact result tables, plots,
protocols, code, and all interpretation notes.

## Compute policy

There are four NVIDIA RTX 6000 Ada GPUs (48 GB each) with no NVLink. Run
independent training or intervention jobs one per GPU; do not use DDP for the
initial draft. All newly trained variants use BF16. Select the fastest measured
batch/worker setting per backbone and record it in the run metadata.

## Credentials and communication

Never place credential values in tracked files, shell history, logs, command
lines, or Telegram messages. Use only these environment variables:

- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `GH_TOKEN` or `GITHUB_TOKEN`
- `HF_TOKEN` only if a public Hub operation unexpectedly requires authentication

If a required credential is unavailable, append a concise message to
`to_human/pending-notifications.md`, continue all non-blocked work, and send
the queued message once the matching environment variables exist. The remote
must remain credential-free: `https://github.com/lab260ru/icassp_antispoofing.git`.

## Git discipline

Work on `research/icassp-signal-audit`. Commit every protocol before generating
its results, then commit results and outer-loop reflections separately. Push at
bootstrap, after each outer loop, after completed training, after paper review,
and at final handoff. Never force-push.
