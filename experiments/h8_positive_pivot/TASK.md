# H8 positive-result pivot — task

## Objective

By 2026-08-11 13:00 UTC, identify a strong, reproducible ICASSP-scale positive
result in cross-corpus speech anti-spoofing—or transparently document the best
defensible pivot after a diverse, compute-bounded search.

## Starting point

H1--H7 are sealed. Their association, intervention-quality, and descriptive
results cannot be tuned, reselected, or reframed as H8 training evidence. H8
may use compatible raw datasets/models only under a newly written and committed
protocol.

## Working assumptions

- Four 48-GB RTX 6000 Ada GPUs are locally available.
- Large data/models belong under `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing`.
- Every protocol, run ledger, result, decision, and paper change remains in
  this repository; each `paper/main.tex` edit immediately recompiles the PDF.
- Candidate directions must have a measurable held-out metric, a baseline, and
  a credible mechanism; no cherry-picked single-corpus gain is a winner.

## Run mode

Autonomous, local-GPU, multi-hypothesis search. This task admits many
substantively different hypotheses, so previous-work audit and external
research precede any production-code modification or training run.
