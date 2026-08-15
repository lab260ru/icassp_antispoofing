# Pre-commitment: enlarging the judge audit across donor checkpoints

Written 2026-08-15, before the run, with 3.4 hours to the deadline. Nothing below
was written or edited after seeing a result; the run's output goes to a new file
and the existing artifact is not touched.

## Why

Three of four reviewers in round 32, and reviewers in three earlier rounds, made
the same objection: the decision to discard Whisper as the judge and adopt a CTC
recogniser rests on an audit of `n=12` (Whisper) and `n=24` (CTC). The paper
states both numbers. The objection is not that they are hidden, it is that they
are small for a choice this load-bearing -- the judge decides every headline
number in the paper.

The audit's `n` is not a sampling cap we chose. `analysis/ctc_validation.py`
slices `atoms[:6]`, but only six donors exist: the stimulus set contains six
`word_rep` items at `k=1`, and the script draws its atoms from one source model
(`--source-model xtts2`). The design, not the slice, is what bounds `n`.

Every checkpoint in the panel has its own `k=1` renderings. The audit's ground
truth is exact *by construction* -- a `k=1` utterance containing exactly one
occurrence of the target, concatenated `N` times -- and that construction is
indifferent to which system produced the atom. Drawing donors from all six
checkpoints multiplies the donor pool sixfold and, as a bonus, stops the audit
being a statement about one system's audio.

## What will be run

`analysis/ctc_validation.py` unchanged in its scoring, once per source model in
`{llasa1b, llasa3b, llasa8b, qwen06b, qwen17b, xtts2}`, at the English defaults
(25 s chunks, no overlap), on `cuda:2`. Output is pooled into a new artifact,
`data/results/judge_audit_scaled.json`. `data/results/ctc_validation.json` is
left exactly as it is.

## What each outcome means -- decided now

The quantity of interest is each recogniser's counting ratio: transcribed
occurrences over true occurrences, where the truth is known by construction.

1. **The direction replicates** -- Whisper's ratio stays far below 1 and the CTC
   judge's stays at or near 1, pooled across donors. Then the paper reports the
   enlarged `n` in place of 12 and 24, the supplement gains the per-donor
   breakdown, and the objection is answered with evidence rather than a hedge.

2. **The direction replicates but is weaker** -- Whisper's pooled ratio rises
   materially (say above 0.5) or the CTC judge's falls materially below 1. Then
   the paper reports the enlarged numbers *and* says the effect is smaller than
   the six-donor audit suggested. The judge choice stands on direction; the
   claim that Whisper transcribes the repeated word "exactly once in every trial
   that returns anything" is withdrawn or restricted to the donors where it held.

3. **The direction fails** -- Whisper is not materially worse than the CTC judge
   pooled across donors. Then the paper's justification for discarding Whisper
   does not survive its own enlarged test, and this must be said plainly in the
   main text: the instrument choice becomes a choice made on six donors that a
   larger audit did not support. The headline numbers do not change, because they
   were never scored with Whisper, but the argument for the instrument does, and
   S14's independent-recogniser replication carries the weight instead.

4. **The run does not finish in time** -- nothing is reported, the existing
   `n=12`/`n=24` stand, and the attempt is recorded in the chronology as
   attempted-and-unfinished rather than silently dropped.

No outcome below case 3 licenses changing which recogniser scored the paper's
results: that scoring is already done, pre-committed, and independently
replicated across four recognisers in S14.

## What would make this analysis wrong

Donor audio from a checkpoint that renders `k=1` badly would inject a truth that
is not true -- the atom is only exact if it really contains one occurrence. The
original audit verified this per atom. The pooled run keeps that verification and
records how many atoms each source model contributed after it.
