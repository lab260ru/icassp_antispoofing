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
