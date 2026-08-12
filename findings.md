# Findings — Counting Collapse in Autoregressive TTS

> **Read this box first.** The single most important thing learned in this
> project is a measurement result, not a model result: **Whisper cannot be used
> to score repetition counting.** Its decoder is autoregressive with an LM prior,
> so it de-duplicates repeated speech. On concatenative audio with exact ground
> truth it counts at ratio 0.19 for k≥4 while a CTC recogniser counts at 1.00 on
> the same files. Every behavioural number in this project was wrong until the
> judge was replaced. If you extend this work, score with CTC.



*Project memory. Read this first on every session/loop tick. Append after every
milestone; never rewrite history, mark superseded claims as ~~struck~~.*

## Current Understanding

Autoregressive TTS decoders generate speech tokens conditioned on a fixed text
prompt. When that text contains a phrase repeated `k` times, two things happen
at once:

1. **The conditioning becomes effectively periodic.** Attention over `k`
   near-identical text spans cannot tell occurrence *j* from occurrence *j'*
   (Lemma B: each gets weight ≈ 1/k, entropy ≈ log k). The decoder therefore
   sees approximately the same context at every repetition boundary.
2. **The state map between boundaries becomes autonomous.** If that map is a
   contraction, boundary states converge to a fixed point — the model's internal
   record of *how many repetitions it has produced* is erased geometrically fast.

The consequence (Theorem A) is a hard, quantitative limit: past a horizon
`N* ≈ log(D/(ε(1−q)))/log(1/q)`, no Lipschitz readout — including the model's
own stop-token head — can distinguish "m repetitions done" from "n repetitions
done". Looping or premature truncation is then not a sampling accident but the
only available behaviour.

## Final results (T+23h)

| claim | measurement | status |
|---|---|---|
| Models undercount repeated text | median rel. count error −8.3% [−12.5,−8.3] at k≥6, n=586, CTC judge | **confirmed**, 5/6 checkpoints with CIs disjoint from their own control |
| Controls are unaffected | median control error **0.0%** at every k up to 32, every model | **confirmed** |
| One model is exempt | Qwen3-TTS-1.7B: 0.0% error, yet still a capacity gap | **second regime**, reported as such |
| Capacity saturates under repetition | gain ratio 0.50; 0.50 controlling output diversity; 0.52 on correct renderings only | **confirmed**, 6/6 disjoint CIs |
| Lemma B (attention dilution) | block entropy = 0.98·log k, δ ≤ 0.06 nats, max share ≤ 1.9/k | **confirmed** |
| Theorem A premise (q<1) | not measurable — the boundary estimator is unsound here | **open**, reported as a negative result |
| Capacity predicts count error across models | Spearman +0.49, n=6 | **underpowered**, not claimed |

Per-model count error at k≥6 (CTC judge): Llasa-1B −16.1%, Llasa-3B −12.5%,
Llasa-8B −25.0%, XTTS-v2 −15.6%, Qwen-0.6B −8.3%, Qwen-1.7B 0.0%. Control: 0.0%
for all six.

## Status (SUPERSEDED — Whisper-judged, kept for the record)

> Everything from here to "Lessons and Constraints" was measured with the
> Whisper judge and is superseded by the "Final results" table above. It is kept
> because the *shape* of the reasoning still holds and because the contrast shows
> how much the instrument mattered.

| Prediction | Status |
|---|---|
| Counting collapses at a finite, model-specific k* | **confirmed** (llasa1b, xtts2) |
| Failure tracks periodicity, not length (matched controls) | **confirmed, strongly** |
| P1 geometric decay of boundary distances | **estimator abandoned** — see below |
| P1' representational capacity saturates under repetition | **confirmed**, disjoint 95% CIs |
| P2 state measure predicts behavioural k* | pending (needs ≥4 models) |
| P3 attention dilution ~1/k, entropy ~log k | measured, not yet written up |
| P4 spectral proxies shift faithful vs hallucinated | measured, table generated |

## Patterns and Insights

### The dissociation is the strongest result

Same carrier, same word count, repetition removed. XTTS-v2 counting accuracy on
repeated words falls 1.00 → 0.00 across k ∈ {1…32} while its length-matched
control stays renderable. Llasa-1B renders *twelve to sixteen distinct adverbs*
in one sentence but fails at *six identical* ones. Whatever is failing is not
capacity for long text.

### The mechanism is capacity saturation, not a measurable contraction rate

Effective rank of the generation trajectory grows with k at roughly **half** the
rate for repeated text as for the matched control:

| model | dN_eff/dlog k repeated | control | ratio |
|---|---|---|---|
| Llasa-1B | 22.0 [12.4, 31.3] | 52.7 [43.4, 61.5] | 0.42 |
| XTTS-v2 | 21.6 [14.1, 28.7] | 36.1 [31.2, 40.9] | 0.60 |

95% template-bootstrap CIs disjoint for both. Llasa-1B's repeated trajectory
plateaus near N_eff ≈ 190 while its control climbs to 249.

Read this as Theorem A(ii) made observable: the decoder stops acquiring internal
states in which "m repetitions done" and "n repetitions done" could differ.

### Negative result: the boundary-difference estimator does not work here

Locating repetition boundaries from text attention requires a monotone read
head. Measured: the deep-layer attention centroid advances on ~51% of generation
steps — a random walk, not a sweep. Boundary estimates then collapse onto
near-duplicate steps and d_m measures localisation error. This is self-defeating
by construction: the flatter the attention over repeated spans (Lemma B), the
worse any attention-based localiser gets. Reported in the paper, kept in
`src/common/boundaries.py`.

### Alternatives tested and ruled out

Four competing explanations were raised in review and answered with measurements,
not argument:

| alternative | test | result |
|---|---|---|
| repeated text is just improbable text | per-token NLL of matched pairs under an LM outside the panel (phi-2) | control is the *less* probable member in 87% of 54 pairs; naturalness runs opposite to the effect |
| the attractor is acoustic, not text-side | word vs sentence repetition, whose units differ several-fold in token cost | k* differs by 1.7 repetitions while token count at collapse differs 1.34×; the horizon is counted in repetitions |
| effective-rank decline is tautological — repeated audio *is* monotonous | add realised output diversity as covariates; and restrict to correct renderings | ratio 0.50 raw, 0.50 adjusted, 0.47 correct-only |
| the count survives and only the output policy fails (as in text LMs, arXiv:2605.09239) | ridge probe, early vs late third of the *same* trajectory | retention 0.97 repeated vs 1.12 control, repeated lower in 7/7; degradation, not erasure |

### The XTTS ablation (revised under the CTC judge)

Disabling XTTS-v2's shipped `repetition_penalty=5.0` degrades *both* conditions
severely (repeated −58.3%, control −66.7% count error), so the ablation is
uninformative about the dissociation rather than confirming it. The earlier
reading — that the penalty was masking the collapse — was based on Whisper-judged
data and is withdrawn. The penalised model remains the conservative panel member,
since the penalty acts against the behaviour under study.

## Lessons and Constraints (added)

- **Never pool an ablation variant into panel statistics.** `xtts2norp` is a
  re-run of `xtts2` under a changed decoding setting; counting it as a seventh
  checkpoint inflated the panel size and every pooled number. Caught in review.
- Report proportions with Wilson intervals and per-cell n. Several cells sit at
  0% or 100% where the normal approximation is degenerate.
- A naturalness referee drawn from the tested panel is not a referee. The
  Llasa-1B and phi-2 scorers agree on the headline (87–89% of pairs) but *not* on
  the trend at high k, so only the direction is claimed.

- **Do not name a module `tokenizers.py` under `src/common/`.** Running a script
  from that directory shadows the real package and breaks `transformers`.
- Duration cannot serve as a repetition count: natural speech-rate variation
  gives it ±1 repetition of resolution. It works as a *consistency band* only,
  which is why correctness is decided conjunctively.
- 21.5% of Llasa-1B generations run to the 2048-token cap — runaway generation is
  common enough that the cap itself is a signal, not an artifact to be tuned away.
- Llasa-8B samples ~5× slower than 1B under eager attention; generation now runs
  under SDPA and only the instrumented pass uses eager.

## Lessons and Constraints

- `xcodec2==0.1.5` hard-pins torch 2.5 → must live in its own conda env; Llasa
  speech tokens are therefore saved as ids and vocoded offline in a second pass.
- `qwen-tts` is its own package (not stock transformers) and pins
  transformers 4.57.3 → separate env.
- Whisper itself deduplicates/loops on repetitive audio, so ASR-based repetition
  counts must be triangulated against duration-derived counts.

## Open Questions

- Is the contraction hypothesis satisfied by all models, or does the panel split
  into contracting and non-contracting regimes (the two-regime story)?
- Does q̂ vary monotonically with scale within the Llasa 1B/3B/8B ladder?
- Do number phrases ("six hundred sixty-six…") collapse at a different horizon
  than bare word repetitions, i.e. does semantic structure buy extra counting?

## 2026-08-12 — The ladder was too short, and three claims were overstated

*Prompted by the round-4 Opus reviewer panel, which was right on all three
counts. Every item here was verified against our own data before being acted on.*

### The headline changed twice in one morning

**First:** the reviewer objected that a constant *relative* deficit is not what a
fixed horizon predicts, and the data agreed. Fitting the panel's median rendered
count above k=6 gave SSE 485 for a constant against 2.6 for `c = 0.95k`. Within
k<=32 the count tracks the request; there is no horizon in that range.
(`analysis/horizon_shape.py`.)

**Then:** so we extended the ladder to k = 48, 64, 96, 128
(`data/stimuli/make_stimuli_ext.py`, `scripts/run_ext_ladder.sh`). Past k=32 the
count stops tracking. Repeated medians fall 27 -> 15 between k=48 and k=128 while
the request grows threefold.

**The controls are the point.** They saturate too, near 60 units — these decoders
have a general utterance-length ceiling, and the repeated plateau *alone* cannot
distinguish counting collapse from "the model will not talk for two minutes". So
the reportable quantity is the ratio, which divides that ceiling out:

    N*_rep  23 [16, 39]      N*_ctl  94 [71, 111]      ratio 4.1
    disjoint bootstrap CIs in both checkpoints (`analysis/horizon_ext.py`)

If you extend this: **do not quote a repeated horizon without its matched
control horizon.** On its own it is uninterpretable.

Known weaknesses, stated so nobody rediscovers them as surprises: two
checkpoints only (Qwen was still generating), few items per cell, and the
repeated median *declines* past the plateau where a pure horizon predicts a flat
line — so something beyond the theorem acts at these lengths. CTC blank-collapse
biases the same way, making 4.1 an upper bound.

### Three claims that were wrong, and are now fixed

1. **"Controls come out without a single miscount"** — false. 22.4% of control
   generations carried a nonzero error. Most of it was one word: our CTC judge
   renders `okay` as "o k" in 106 of 106 items. The honest and sharper statement
   is the exact-rate, **94.3% control against 18.2% repeated**.
2. **"The states the decoder visits stop multiplying"** — false. Capacity gain
   stays positive in 5 of 6 checkpoints. Repetition *slows* state acquisition; it
   does not halt it. A decoder at the theorem's fixed point would gain none, and
   none is at it — which fits the horizon result above.
3. **Lemma B's delta was quoted as 0.06 and implied 0.64.** Those bound the
   *average* and the *extreme* logit spread. Both are now reported; the lemma is
   entitled only to the worst case.

### Two scoring bugs that were changing the numbers

* `count_units` stopped scanning at the first unit it could not find. Correct for
  repeated items (all units identical, so a miss means all later ones miss too),
  wrong for controls, where one mis-transcribed filler voided credit for every
  filler after it. Now skips. Verified bit-identical on repeated items.
* **Items truncated by our own token budget were being scored as model failures.**
  `hit_cap` items carry median relative error -0.44 against -0.08 for the rest.
  11.3% of the headline population. Now excluded.

`src/common/population.py` is now the single definition of the reportable
population — the exclusion lists had already drifted between `count_error.py`
(one ablation) and `capacity.py` (four), so re-scoring the full model list would
have silently folded three repetition-penalty arms into the panel.

### The premise has now failed to be established twice

A finite-difference perturbation probe (`analysis/contraction_probe.py`) measured
local contraction directly. The decisive repeated-vs-control contrast came out
null: 7 of 18 pairs in the predicted direction, p=0.12, with the trend mildly the
*wrong* way, replicated at two injection depths. Its own step-size linearity
control also failed. Together with the earlier boundary-distance attempt, that is
two independent failures to measure `q`. **Assumption 2 remains unestablished**,
and the paper says so. Do not present either q-hat as validated.

### Also worth knowing

* The main ladder's controls are **not** repetition-free above k=8: fillers cycle
  an eight-word pool, so they carry period-8 repetition. The contrast the paper
  tests is period-8 against period-1. The extension ladder uses a 146-word pool
  that is never cycled, and asserts it.
* **XTTS-v2 cannot be run past k=32 at all** — a built-in ~602 mel-token ceiling
  (~26 s) censors every longer item. Not a result about XTTS-v2; a limit of the
  extension.
* Whisper's failure on *real* generated audio is worse than the de-duplication
  the original audit found: at k>=24 it emits 10.7-12.3 words/sec, which is not
  physically speech. It hallucinates fluent text over garbled audio. The decision
  to drop it is better supported than when it was made.

## 2026-08-12 (later) — the non-autoregressive baseline, and two review rounds

### The result that most strengthens the paper

Reviewers kept asking the obvious question the title invites: the paper claims
something about *autoregressive* TTS but had never run a decoder that was not.
`src/models/vits_gen.py` runs VITS (`facebook/mms-tts-eng`) on the same ladder --
text encoder, duration predictor, flow vocoder, one shot, no recurrence.

    exact-rate, repeated vs its own length-matched control, k>=6
    llasa1b 13.3/91.5 (+78.1)   qwen06b  7.8/100.0 (+92.2)
    llasa3b 16.9/94.1 (+77.2)   qwen17b 38.9/ 98.9 (+60.0)
    llasa8b 11.8/93.2 (+81.4)   xtts2   16.7/ 87.8 (+71.1)
    VITS    65.6/58.9 ( -6.7)   <- no dissociation at all

**Two things follow, and the second was a bonus.** The deficit is not a property
of long repeated *text*. And it clears the judge: reviewers twice worried that
CTC blank-collapse manufactures the gap by merging adjacent identical words. If
it did, VITS's repeated items would have been depressed too -- same judge, same
stimuli, same words. They were not. Auditing the recogniser in isolation could
never have shown this, because the audio it gets audited on is not the audio in
question.

If you extend this work: **run the non-AR baseline before trusting any judge
result on repeated speech.** It is cheap (VITS renders the whole ladder in
minutes on one GPU) and it is the only control that tests the judge on the actual
distribution.

### The extension ladder, finished

Four checkpoints, two families, 21-26 items per cell (was 2 checkpoints, 3-6).
The repeated median is FLAT at 25-30 from k=48 to 128 while controls climb 46 to
66. The "declines rather than plateaus" caveat from the morning was a small-n
artifact and is gone. Pooled ratio 3.5 (was 4.1).

**Not unanimous, and the exception is the useful part.** Disjoint intervals for
Llasa-1B and Llasa-8B; Qwen3-TTS-1.7B shows *no* repeated-side saturation within
k<=128 and inverts the ratio -- and it is the same checkpoint with no deficit at
k<=32. A conditional theorem behaving conditionally.

### The title changed, because it was overclaiming

Three of four round-6 reviewers said the same thing: "A Formally Verified
Attractor Theory of Repetition Hallucination" asserts a mechanism the paper
concedes it never established. It is now **"Counting Collapse in Autoregressive
Text-to-Speech: A Lean-Verified Bound and a Measured Counting Horizon"**, and the
abstract states both failures itself. Do not quietly restore the old framing.

### Statistics, done properly and honestly

* `analysis/checkpoint_level.py` -- checkpoint, not generation, as the unit.
  Exact-rate gap +76.7 pts [68.4, 84.4] in 6/6; capacity +24.7 in 6/6;
  median-error 5/6, p=0.062, which misses even the exact n=6 floor. Say so.
* `analysis/family_level.py` -- checkpoints cluster in 3 families (Llasa 3,
  Qwen 2), so the effective replicate count is 3. Exact-rate gap 75.4 pts,
  range [71.1, 78.9], positive in all three. Quote this weaker claim.
* **No six-checkpoint panel can survive multiplicity correction on this test.**
  The exact signed-rank floor at n=6 is 0.031, so the smallest Holm-adjusted p
  across three contrasts is 0.094. The paper rests on effect sizes, not
  thresholds. Enlarging the panel is the only fix.

### Two more things that went against us

* `analysis/dilution_sufficiency.py` -- Lemma 1's dose-response prediction is
  REVERSED. Within a (model,k) cell, flatter attention counts *better*
  (r=+0.59 [0.35,0.75], all three flatness measures, controls at zero). Recorded
  as disconfirming. Dilution is not what selects which generations fail.
* `analysis/aperiodic_controls.py` -- reviewers caught that main-ladder controls
  cycle an 8-word pool, so above k=8 the contrast is period-1 vs period-8, not
  vs aperiodic. Checked the two genuinely aperiodic regions: k<=8 (54.2% vs
  77.9% exact) and the extension ladder's never-cycled 146-word pool (median
  error -0.62 vs -0.12). It survives both.

## 2026-08-12 (evening) — three more rivals excluded, one more claim retracted

### The rivals that are now dead, with the data that killed them

* **The repetition penalty is not the cause** (`analysis/penalty_confound.py`).
  This was the strongest decoding-level alternative and we had the data all
  along without reporting it. XTTS-v2 across a 4x penalty range: repeated
  exact-rate 15.6-21.1%, slope -0.37 pts per unit. The three Llasa checkpoints
  pass **no penalty at all** and show the effect at full size (14.0% vs 92.9%).
  Note the asymmetry: disabling XTTS-v2's penalty collapses its *control* to
  30.9%, so 1.0 is an unusable setting, not an informative one — fit slopes only
  over arms whose control survives.
* **The control is genuinely aperiodic now** (`analysis/aperiodic_controls.py`,
  `scripts/run_aperiodic.sh`). Three review rounds objected that above k=8 the
  fillers cycle an 8-word pool. Re-generated k=12..32 from the never-cycled
  146-word pool, paired against the same repeated items: gap +71.2 pts against
  +84.6 for cycled controls. **The confound is worth 13.4 points, not the bulk
  of the effect.** A reviewer estimated two-thirds by comparing k<=8 with k>8,
  which conflates the confound with the effect's own k-dependence.
* **Architecture** (earlier today): VITS shows no dissociation at all.

### The claim we retracted

`analysis/probe_discrimination.py`. The paper cited a linear probe as evidence
that repetition *erases* the count from the state, against the rival where the
count survives and only the output policy fails. It does not show that:

    repeated-item late R^2: median 0.48 (0.41-0.62) -- still decodable
    retention > 1.0 in 2 of 6 checkpoints (range 0.77-1.92)
    repeated retention < its own control in 6 of 6

A state that had lost the count could not support a late R^2 near a half, and
the rival predicts exactly that persistence. Only the *relative* effect survives,
and it is equally consistent with a policy degrading faster on repeated text.
**Do not cite the probe as mechanistic support.** We also removed "a probe
reading the count as well late as early" from the paper's list of refutation
criteria — a test called non-discriminating cannot also be a refutation test.

### Where the mechanism stands (read this before writing any mechanistic claim)

Nothing directly supports the causal story: two failed attempts at the
contraction premise, a sign-reversed dilution dose-response, and a probe that
discriminates nothing. What stands is the *phenomenon* plus six excluded rivals
(length, periodicity-of-control, repetition penalty, ASR judge, architecture,
improbability/acoustics). The title says so: "a horizon it does not yet explain."

### Infrastructure

**GPUs 2 and 3 only.** `src/common/gpus.py` is the single source; every entry
point calls `check_gpu()` and refuses 0/1. Escape hatch `ICASSP_ALLOW_ANY_GPU=1`.
When wiring that guard I broke all seven entry points with a missing `sys.path`
insert — the failure was loud and immediate, which is the argument for enforcing
constraints in code rather than in a README.

**Page budget is the binding constraint on everything.** `main.tex` ends with
`\vfill\pagebreak`, so a single spilled body line costs a whole page. Every
addition must be paid for by a cut. Check with the pypdf snippet in
`implementation-notes.md` before and after any edit.

## 2026-08-12 (late) — the penalty is dead in two architectures

`analysis/penalty_confound.py`, now with Qwen3-TTS-0.6B swept over
1.0/1.05/1.5/3.0 alongside XTTS-v2's 1.0-8.0:

    XTTS-v2, 4x range           repeated exact 15.6-21.1%
    Qwen3-TTS-0.6B, 3x range    repeated exact  7.8-16.7%
    Qwen3-TTS-0.6B, PENALTY OFF repeated 10.0% vs control 100.0%
    Llasa (ships no penalty)    repeated 14.0% vs control 92.9%

**The zero-penalty Qwen arm is the one to quote.** Disabling XTTS-v2's penalty
also collapses its *control* (30.9%), so that arm is uninformative about
repetition; Qwen's control is unharmed at zero penalty, so the comparison can
actually be made. A model that normally ships a penalty, run without one, shows a
90-point gap.

The two architectures disagree on the sign of the slope (-0.37 vs +3.95 points
per unit of penalty), which is the signature of a lever not acting on the
quantity that governs the effect.

**Rule for future arms:** fit slopes only over arms whose control survives. An
arm that cannot render the control says nothing about repetition, and including
it makes the sweep look responsive when it is not.

### The judge's noise floor (`analysis/noise_floor.py`)

    two CTC recognisers disagree by  0.47 repetitions (0.79 at k>=12)
    mean repetitions missing, k>=6:  0.87  = 1.9x that floor

The per-item effect is thin and the paper says so. What carries it: the control
passes the same judge and is exact 94.3% of the time (indiscriminate half-count
noise cannot produce that), and where the two recognisers disagree **ours reports
the higher count in 11 of 13 cases** — our judge under-states the deficit. A more
accurate recogniser would report a larger effect, not a smaller one.

Do not quote a single item as evidence. The claim lives in the rate across
hundreds of generations and in the control contrast.

### Released for verification

`data/audio_sample/` — 165 clips, 7 MB, 16 kHz mono Opus (what the judge
consumes), one per (model, family, k-band, outcome), all eight outcome classes.
`manifest.csv` pairs each with stimulus text, transcript, counts and label. Built
by `scripts/make_audio_sample.py`.
