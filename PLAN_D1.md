# 3-day plan — decided 2026-08-12, deadline Aug 15 20:00 UTC+3

## Decisions taken (user, Day 1)
- **Objective: maximise P(accept).** Track C is the frame *whatever the decision
  experiments say*. A positive q folds in as a measured premise; it does not
  restructure the paper.
- **Gate authority: agent decides A or C autonomously.** A full Track-B
  mech-interp pivot (new spine, theorem removed) requires explicit user
  sign-off before execution.
- **Widening menu: all four approved** — Qwen probe past horizon, stats
  hardening (mixed-effects + Bayesian), CosyVoice 2, non-English stimuli.

## Why Track C, on the evidence (r15-r17 synthesis, 9 reviews)
- Verdict is flat "major revision" 9/9, but the *empirical core is uncontested*:
  by r17 all three reviewers call the periodicity dissociation publishable on
  its own ("solid and worth publishing", "genuinely strong").
- The theorem's standing degraded monotonically across rounds: r15 "overclaims"
  -> r16 "must be subordinated" -> r17 "decorative for most of the paper".
- **No reviewer in any round says remove.** Every one says demote / subordinate
  / reframe as conditional. Track C is literally what nine reviews asked for.
- The bulk of recurring Major concerns are fixable by reframing, re-ordering and
  re-tabulating data already on disk — not by new compute.

## The reframe (main paper)
1. **Retitle** away from the AR scope mismatch (F5-TTS, non-AR, shows the
   effect; the title still says "Autoregressive").
2. **Abstract reordered**: dissociation first; by-family (n=3) as the primary
   statistic; theorem stated as an untested conditional, not a companion
   finding.
3. **Shrink theory section**; the freed space pays for the additions below.
4. **Label sections confirmatory vs exploratory** explicitly, in the headers,
   not in parentheticals.
5. **Probe-past-horizon downgraded** to "inconclusive, presented for
   transparency", all checkpoints named and reported symmetrically.
6. Concede in-text that the balance of probe evidence currently favours the
   **output-policy rival**, and say what would decide between them.

## Cheap wins reviewers asked for repeatedly (data already in hand)
- Per-checkpoint exact-match column in Table 1 (the most-quoted number is the
  one Table 1 omits -> reads as cherry-picking).
- Figure 1 axis labels — garbled, flagged by **nine consecutive reviewers**.
- Effective-rank definition, probe architecture/CV, layer indices, CTC decoding
  config, and the c-hat extraction procedure — one sentence each.
- Say the supplement is public in the repo.
- Explain Llasa-3B's absence from the k=128 ladder (XTTS-v2's is explained).
- Reconcile the two Whisper audit numbers (0.27-0.65 vs 0.19).
- Never-cycled-pool number reported alongside the cycled one as primary.
- Counting/expressivity citations + what contraction predicts that an
  inductive-bias account does not.

## Repo consistency (stale, quotable, currently self-contradictory)
- `findings.md` + `README.md` say probe-past-horizon is "1 of 2";
  `research-state.yaml` and `probe_horizon_compare.json` say 1 of 3.
- Lemma B delta: `findings.md` says 0.06 / 1.9k; `research-state.yaml` says
  0.45 nats / delta 1.23 / 3.4k.
- F5 duration-intervention result is in the paper but never in `findings.md`.

## GPU queue (cuda:2, cuda:3 only)
1. RUNNING — Jacobian q estimator (c:2), activation-patching pilot (c:3).
2. Qwen probe past horizon + missing control arms for llasa3b/8b.
3. Third recogniser as independent judge (answers the CTC-blank-collapse
   objection with a number, 3 rounds running).
4. Non-English stimuli (needs a non-English CTC judge — wav2vec2-960h is
   English-only).
5. CosyVoice 2 (env fight; download/setup is CPU, runs when a GPU frees).
6. If time: full-ladder never-cycled controls; decoding sweep on all six.

## Cadence
- /paper-review into a fresh `paper/reviews/rN`, continuing from **r18**.
- `bash scripts/verify_submission.sh` green before every "done" claim.
- `bash scripts/notify.sh` at every milestone and every claim change.
- Commit + push after every landed unit on `research/ar-tts-counting-collapse`.
