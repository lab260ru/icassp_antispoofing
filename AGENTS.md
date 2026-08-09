# Research Continuity Guide

## Mission and deadline

Conduct a reproducible causal audit of signal features in cross-dataset speech
anti-spoofing and prepare an ICASSP 2027 initial draft by **2026-08-10 18:03
UTC**. The provisional story is a causal audit plus a robust feature-conditioned
fix; it may pivot only according to the recorded outer-loop evidence gate.

## Resume order

1. Read `research-state.yaml`, `findings.md`, and the latest entry in
   `research-log.md`.
2. Inspect `git status --short --branch` and the active experiment's
   `protocol.md` before running anything.
3. Check active jobs and the HDD run directory before starting duplicates.
4. Record each material decision in `implementation-notes.md` and append the
   result to the active experiment's `analysis.md`.

## Current checkpoint — 2026-08-09

- **H1 runs 005--006 are complete on ASVspoof2019_LA and ASVspoof2021_LA.**
  Their immutable source tables are under
  `experiments/h1_feature_association/results/<dataset>/`; each has 1,344
  within-class screen rows (8 models × 2 classes × 3 views × 28 features) from
  the locked 5,000-per-class subset. `results/discovery_aggregate/` is an
  explicit two-input descriptive report that correctly marks portability as
  not evaluable. Do not call either result portable or causal.
- **Association estimator safety is tested.** Use
  `PYTHONPATH=. python3 scripts/analyze_associations.py ...`; the normal
  `pytest` console entry point may omit the repository from `sys.path`, so run
  `PYTHONPATH=. python3 -m pytest -q tests/test_statistics.py
  tests/test_audio_features.py`. The partial-correlation code deliberately
  removes a tested variable from its own controls. Bootstrap CIs require a
  frozen provenance-bearing manifest; see
  `experiments/h1_feature_association/BOOTSTRAP_CLI.md`.
- **Active, non-duplicable job:** ASVspoof2021_DF is downloading via the
  resumable public-HF downloader. Check processes and the HDD paths before
  relaunching. After it is ready, ingest its already-downloaded score panel and
  run H1 into its own scope; freeze candidates only after all three discovery
  datasets are analyzed.
- **H2 has two parity-validated runners but is not ready to claim.** Read
  `experiments/h2_causal_interventions/model_capability_audit.md`,
  `H2_ONNX_PARITY.md`, and `PARITY_RUN_001.md` before any intervention work.
  Run ONNX commands through `scripts/run_h2_cuda.sh`; it exposes the active
  environment's CUDA 13 wheel libraries. AASIST uses raw input and
  Spectra-AASIST uses external pre-emphasis 0.97. Initial frozen-workload
  sweeps select AASIST batch 2 (GPU 1) and Spectra batch 8 (GPU 0); see
  `H2_THROUGHPUT_RUN_001.md`. H2 still requires a fixed ASR WER runtime and a
  validated fourth model or a pre-frozen XLSR-SLS fallback. Score artifacts
  cannot score transformed audio.
- **Paper draft:** `paper/main.tex` compiles with
  `tectonic --outdir build main.tex` from `paper/`. Track the source, citation
  ledger, and `paper/build/main.pdf`; ignore build intermediates. The initial
  3-page PDF reports only the first discovery observation and its limitations.
  The required three-reviewer launcher was attempted but blocked because the
  local Claude CLI is unauthenticated; see
  `paper/reviews/initial_draft/REVIEW_STATUS.md`. Do not pretend a review took
  place. Once authenticated, run a new timestamped paper-review bundle.
- **Remote and Telegram:** a credential-free remote is configured, but no
  runtime GitHub or Telegram credential exists. Keep committing locally and
  append milestones to `to_human/pending-notifications.md`; push/send only when
  environment credentials are supplied. Never recover or reuse previously
  exposed credentials.

## Artifact locations

- Repository: `/home/kirill/icassp_antispoofing`
- Large data/models/cache/checkpoints/logs:
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`
- Dataset/model provenance: `data/arena-index.yaml` and `data/README.md`
- Paper source and reviews: `paper/`
- Human updates and queued notifications: `to_human/`

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
