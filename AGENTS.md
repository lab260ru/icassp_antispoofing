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

- **H1 run_005 is complete on ASVspoof2019_LA.** The immutable source tables
  are `experiments/h1_feature_association/results/association_summary.csv`,
  `feature_label_metrics.csv`, and `association_join_report.json`. They contain
  1,344 within-class screen rows (8 models × 2 classes × 3 views × 28
  features) from the locked 5,000-per-class subset. Do not call this a portable
  or causal claim: it is only the first discovery corpus.
- **Association estimator safety is tested.** Use
  `PYTHONPATH=. python3 scripts/analyze_associations.py ...`; the normal
  `pytest` console entry point may omit the repository from `sys.path`, so run
  `PYTHONPATH=. python3 -m pytest -q tests/test_statistics.py
  tests/test_audio_features.py`. The partial-correlation code deliberately
  removes a tested variable from its own controls. Bootstrap CIs require a
  frozen provenance-bearing manifest; see
  `experiments/h1_feature_association/BOOTSTRAP_CLI.md`.
- **Active, non-duplicable jobs:** ASVspoof2021_LA full feature extraction is
  running from the locked local dataset; ASVspoof2021_DF is downloading via the
  resumable public-HF downloader. Check processes and the HDD paths before
  relaunching. After their features are ready, ingest their already-downloaded
  score panels and run H1 jointly or separately; freeze candidates only after
  all three discovery datasets are analyzed.
- **H2 is not ready to claim.** Read
  `experiments/h2_causal_interventions/model_capability_audit.md` before any
  intervention work. It requires a pre-arm 128-clip parity/orientation check,
  a fixed ASR WER runtime, and a validated fourth model or a pre-frozen
  XLSR-SLS fallback. Score artifacts cannot score transformed audio.
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
