# H10 CD-ADD terminal evaluation 001

**Status:** complete, terminal no-retune follow-up. The predeclared H10
extension gate did **not** pass, so this result does not extend the H9 paper
claim or authorize another target/model/loss/source search.

## Fixed chain

- H10 replays H9's exact 12 fresh Res2TCNGuard checkpoint panel, sealed
  ODSS-only source artifacts, B1/B2/P methods, seeds, input policy, and score
  orientation. No source-side choice was reopened.
- Target: every audio-bearing trial of `SpeechAntiSpoofingBenchmarks/CD-ADD`
  revision `b03c6cf3463d67c1525ba20738c495d120675876`: 20,786 trials
  (3,661 bona-fide; 17,125 spoof). Its public-card audit was completed before
  target materialization; records were copied/fingerprinted from `path,audio`
  before the separate `path,label` artifact was written.
- The terminal evaluator replayed the H9 source ledger, found zero exact
  canonical source--target waveform collisions, wrote all 249,432 raw
  predictions (20,786 × 3 methods × 4 seeds) before its only label join, then
  reported the all-trial four-seed probability ensemble and 2,000 shared-ID,
  label-stratified bootstrap contrasts (seed 2909).

## Result

| Method | EER | AUROC |
| --- | ---: | ---: |
| B1: same-pool BCE | 45.425% | 0.5848 |
| B2: BCE + random-pair ranking | 48.921% | 0.5280 |
| P: BCE + same-item ranking | **41.655%** | **0.6147** |

P is 3.77 percentage points lower than B1 (95% CI for P--B1:
[-4.70, -2.77] pp) and 7.24 points lower than B2 ([-8.39, -6.12] pp). Both
paired intervals are below zero. However, the predeclared relative reduction
versus B1 is **8.30%**, below the required 10%; the B2 reduction is 14.85%.
Consequently the full H10 gate is `false` despite every provenance,
reconstruction, collision, prediction-order, and bootstrap rule passing.

## Interpretation and stop

This is evidence consistent with the direction of H9 on a fresh TTS target,
but it is not a passing H10 replication under the fixed practical-effect rule.
Do not relax the 10% boundary, substitute the statistically significant
absolute contrast, choose a different target, subset CD-ADD, change the
checkpoint panel, or alter H9's method. The H9 manuscript remains restricted
to its sealed SONAR+ArAD claim; it has not been edited for H10.

## HDD artifact ledger

All large inputs and outputs reside below
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h10_cdadd/`.

| Artifact | SHA-256 |
| --- | --- |
| CD-ADD target manifest | `3913178f84a10d8be76d5eb3448f07037ab22154e00398f6e5b8351447047f3d` |
| target materialization provenance | `711660edc0f8099147a86dd2e6ba7c33cafc0f9a7e0d0ff618086c0888bd2a8b` |
| raw predictions | `550d1aa354a16ee657f54332dc5bcec50ac077188c7549eff44fc1d4422c2645` |
| terminal metrics | `d7fac38cbc3137e44fc5d26fcd593b5278f9a7f0e53a10d062df2049cc5b7123` |
| per-seed metrics | `2bf6eedd9097418ee7ee9ea115209dacf507bbc6a2134b0ac37915195d0e7920` |
| bootstrap contrasts | `d1dbead5b90655ded765fc1521980373b33f4552e42c7e7d014b7f7ff8d5730c` |
| terminal decision gate | `fb033f5286b0d2420de0805f005a9ee74693719ce347aa4cb4bb7678c59ee63e` |

Read the H10 protocol and decision record before any future extension:
`H10_CDADD_EXTERNAL_REPLICATION_PROTOCOL.md` and
`H10_DECISION_RECORD_20260811.md`.
