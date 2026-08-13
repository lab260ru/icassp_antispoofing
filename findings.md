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

> **This table's "Theorem A premise" row is SUPERSEDED (2026-08-12) and must
> not be requoted as written.** "Not measurable" became "measured and
> refuted" once `analysis/jacobian_q.py` passed its own gates: q = 38.1 on
> Llasa-1B, q<1 in 0 of 29 items, control arm also expansive. See "Assumption
> 2 is not unmeasured. It is false." further down this file, and its r19
> follow-up under "the single-checkpoint correction" for the scope this is
> entitled to (one checkpoint, not panel-wide). The panel-level count-error and
> capacity rows in this table are not known to be stale — they were re-verified
> against later per-checkpoint tables in this file at the time they were
> written — but this table itself dates from very early in the project (T+23h)
> and every number in it should be cross-checked against the dated entries
> below before being requoted, per this file's own convention.

| claim | measurement | status |
|---|---|---|
| Models undercount repeated text | median rel. count error −8.3% [−12.5,−6.2] at k≥6, n=457 repeated against 525 control, CTC judge (`count_error.json`) | **confirmed**, 4/6 checkpoints with CIs disjoint from their own control (5/6 below their control) |
| Controls are unaffected | median control error **0.0%** at every k up to 32, every model | **confirmed** |
| One model is exempt | Qwen3-TTS-1.7B: 0.0% error, yet still a capacity gap | **second regime**, reported as such |
| Capacity saturates under repetition | gain ratio 0.46 (`capacity.json`); 0.50 raw / 0.50 diversity-adjusted / 0.52 correct-only (`capacity_confound.json`) | **confirmed**, 5/6 disjoint CIs |
| Lemma B (attention dilution) | block entropy = 0.97·log k; **average** spread δ ≤ 0.45 nats (entropy gap); **extreme** δ ≤ 1.23, most-attended share ≤ 3.4/k worst case, 1.7/k typical | **confirmed** — the lemma is entitled only to the extreme; see §"Lemma B's two deltas" |
| ~~Theorem A premise (q<1) \| not measurable — the boundary estimator is unsound here \| **open**, reported as a negative result~~ | **SUPERSEDED 2026-08-12: measured and refuted on Llasa-1B, q=38.1. See "Assumption 2 is not unmeasured. It is false." below.** | — |
| Capacity predicts count error across models | Spearman +0.49, n=6 | **underpowered**, not claimed |

Per-model median count error at k≥6 (CTC judge, `data/results/count_error.json`
under the current `src/common/population.py` exclusions): Llasa-1B −12.5%,
Llasa-3B −8.3%, Llasa-8B −16.7%, XTTS-v2 −12.5%, Qwen-0.6B −12.5%,
Qwen-1.7B 0.0%. Control: 0.0% for all six. (~~An earlier row read −16.1 / −12.5 /
−25.0 / −15.6 / −8.3 / 0.0; that predates the `hit_cap` and template-t2
exclusions and disagreed with the per-checkpoint table at the foot of this
file.~~)

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
| the attractor is acoustic, not text-side | word vs sentence repetition, whose units differ several-fold in token cost | k* differs by 1.7 repetitions while token count at collapse differs 1.50× (`unit_invariance.json`); the horizon is counted in repetitions |
| effective-rank decline is tautological — repeated audio *is* monotonous | add realised output diversity as covariates; and restrict to correct renderings | ratio 0.50 raw, 0.50 adjusted, 0.52 correct-only (`capacity_confound.json`) |
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

- ~~Is the contraction hypothesis satisfied by all models, or does the panel
  split into contracting and non-contracting regimes (the two-regime
  story)?~~ **Partially answered, 2026-08-12**: on Llasa-1B it is not
  satisfied at all — repeated *and* control text are both expansive (q=38.1
  and 31.8 respectively), so there is no contraction regime on that checkpoint
  to split by. Still genuinely open whether a *different* checkpoint
  contracts; a second architecture family's jacobian is running as of this
  writing (see `research-state.yaml`'s `active_background_jobs`).
- Does q̂ vary monotonically with scale within the Llasa 1B/3B/8B ladder? Still
  open — the one measured point (Llasa-1B, q=38.1) has no sibling yet.
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

> **SUPERSEDED 2026-08-12.** "Unestablished" stopped being the right word once
> a third, gated estimator (`analysis/jacobian_q.py`) measured q directly and
> rejected the premise outright — see "Assumption 2 is not unmeasured. It is
> false." further down this file. The two attempts below are kept as the
> record of what did *not* work (both were later shown, via the boundary-
> distance decay reproducing q≈1 at R²=0.02, to be non-localisation failures
> rather than measurement noise) — they were superseded by a working
> estimator, not corroborated by one, and neither q-hat below should be
> presented as validated.

A finite-difference perturbation probe (`analysis/contraction_probe.py`) measured
local contraction directly. The decisive repeated-vs-control contrast came out
null: 7 of 18 pairs in the predicted direction, p=0.12, with the trend mildly the
*wrong* way, replicated at two injection depths. Its own step-size linearity
control also failed. Together with the earlier boundary-distance attempt, that is
two independent failures to measure `q`. ~~**Assumption 2 remains
unestablished**, and the paper says so.~~ Do not present either q-hat as
validated.

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

> **SUPERSEDED 2026-08-12 — read "the mechanism stands nowhere" further down
> this file instead.** By the end of the final day all four candidates raised
> in this project had been tested to a conclusion, not left open: contraction
> was measured and refuted (single checkpoint), the dilution dose-response
> stayed sign-reversed, the stop-head readout came back inseparable/
> inconclusive, and a causal intervention (rank-1 patch) produced a readable
> null rather than an uninterpretable one. The paragraph below is kept for the
> record of where things stood mid-project; it undersells the later evidence
> by calling the probe merely "non-discriminating" when the causal follow-up
> shows the decoder actively ignoring a transplanted count.

Nothing directly supports the causal story: two failed attempts at the
contraction premise, a sign-reversed dilution dose-response, and a probe that
discriminates nothing. What stands is the *phenomenon* plus six excluded rivals
(length, periodicity-of-control, repetition penalty, ASR judge, architecture,
improbability/acoustics). The title said so at the time: "a horizon it does not
yet explain" — the title has since changed twice more; see the entries below.

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

## 2026-08-12 (late, cont.) — the architectural claim is dead

A second non-AR baseline contradicts the first. **Do not restore the claim that
this deficit is autoregressive.**

    model              rep exact   ctl exact    gap
    AR panel              18.2%       94.3%   +76.1
    F5-TTS (2024)         25.6%       85.6%   +60.0   <- shows it
    VITS (2021)           65.6%       58.9%    -6.7   <- does not

F5-TTS gap by k band: +8.9, +43.3, +56.7, +80.0 — grows exactly as the panel's
does. Durations scale with k (2.6 s at k=1, 11.6 s at k=32), controls hold above
83%, so it is genuinely rendering and the effect is periodicity-specific there
too. `analysis/nonar_baseline.py` reports the two baselines separately and must
keep doing so: pooling them hides the only thing the experiment established.

**The working hypothesis that replaces it** (post-hoc, untested, labelled as such
everywhere it appears): the relevant property is not recurrence but whether the
count must be held internally. VITS predicts a duration per input token, so "how
many" rides the input sequence and is never represented. F5-TTS estimates one
total duration and denoises in parallel, so it must represent how much speech to
make — the burden an AR decoder carries in its state.

**The experiment that would test it** and which we did not run: a duration
ablation inside one architecture — force a per-token duration model into an
otherwise unchanged parallel decoder. That is the first thing to run next.

**What VITS is still good for:** it remains the judge's control. Blank-collapse
would have depressed its repeated items too — same recogniser, same strings — and
did not. No amount of auditing the recogniser against reference audio could show
that, because that audio is not the audio in question.

## 2026-08-12 (late) — first positive evidence for the theorem

`analysis/probe_past_horizon.py`. Every probe in this project ran at k<=32, at or
below the fitted horizon (~30), which is the range where Theorem 1(iv) *permits*
a Lipschitz readout to succeed. So the probe's success there tested nothing, and
its failure to discriminate the state-vs-policy question was not evidence either.

Regenerated k=48..128 with hidden-state capture and reran the same ridge probe:

                          probe MAE   constant-predictor MAE     R2
    k=2..32   (below N*)     0.63             1.12            +0.62
    k=48..128 (past N*)      0.50             0.50            -0.02

**Past the horizon the probe matches a predictor that ignores the states and
answers the mean.** It has learned nothing. That is the theorem's own conclusion,
tested where it applies, and it passes.

**Always quote the constant-predictor column.** R^2 falls when the target varies
less, and k=48..128 spans a third of the log2 range k=2..32 does — a low R^2 past
the horizon would prove nothing by itself. The trivial-baseline comparison is
what makes the result safe.

**Superseded 2026-08-12: the second checkpoint does not replicate.** Llasa-8B,
run through the identical pipeline, *keeps* the count past the horizon (MAE 0.43
against 0.50 for the constant predictor, R² +0.19) --- and 0.43 is, to two
decimals, exactly the margin the control arm achieves. So on Llasa-8B, periodic
conditioning past the horizon does nothing to the probe.

    checkpoint  arm                    MAE   constant   R²
    Llasa-1B    repeated k=48..128    0.50     0.50    -0.02
    Llasa-8B    repeated k=48..128    0.43     0.50    +0.19
    Llasa-1B    control  k=48..128    0.43     0.50    +0.26

~~The theorem's conclusion therefore holds in **1 of 2** checkpoints, and the
paper says one of two.~~ **Superseded: a third checkpoint (Llasa-3B) was run and
it is 1 of 3 — see "third probe checkpoint" at the foot of this file.** Two
readings we cannot separate: Llasa-8B may have a horizon past k=128 (making this
a range problem, not a failure), or the 1B null may be the accident.

Note the falsification list in the discussion names "a probe that recovers the
count past the horizon" as a refuting observation. Llasa-8B *is* that
observation, and the list says so rather than dropping the condition.

The range control still holds: over the *same* k, the probe recovers the count
from control states, so where the count is lost, narrowness is not why. It cannot
turn a one-of-two result into a panel one.

Limits: two checkpoints, 12 repeated items each plus 12 controls, one seed, three
folds. It does **not** establish the premise (nothing here measures q) and does
not make the account causal.

**Two engineering traps if you repeat it:**
- The instrumented pass uses eager attention to capture weights, which
  materialises a T×T matrix per layer and OOMs at ~4000 tokens. The count probe
  reads hidden states only, so pass `--attn-probes 0`; llasa_gen.py now switches
  to SDPA when no attention probes are requested.
- The original extension sweep ran *without* instrumentation, so the items must
  be regenerated (`data/stimuli/stimuli_ext_instr.jsonl`, item ids suffixed `i`).

## 2026-08-12 (final day) — three robustness experiments, one replication failure

Four things were run in the last cycle, three of them because a reviewer
objection could be answered with data rather than prose. Two came back clean, one
weakened a headline number, and one killed the paper's only positive result as a
standalone finding.

**1. The range confound is dead (analysis/probe_past_horizon.py).** The
probe-past-horizon null rested on an argument: R² falls when the target varies
less, so we compared against a constant predictor and called it neutralised. A
reviewer may answer that range is not the only thing that changed. So we ran
12 *control* items — aperiodic carriers, never-cycled vocabulary — over exactly
k ∈ {48,64,96,128} with the same instrumentation. The probe reads the count off
them (MAE 0.43 vs 0.50 constant) while failing on repeated states over the
identical range. Narrowness is ruled out empirically, not argued away.

**2. But the second checkpoint does not replicate.** See the correction above:
Llasa-8B keeps the count. ~~1 of 2~~ **1 of 3 once Llasa-3B landed**, and
reported as such.

**3. The 3.5-fold horizon ratio is form-dependent (analysis/horizon_forms.py).**
Refit with two one-parameter saturating families we did not choose, both with
unit slope at k→0 and asymptote K̂:

    soft horizon  K(1−e^−k/K)   rep 30.1  ctl 104.8  ratio 3.48  ← reported
    hyperbolic    Kk/(K+k)      rep 42.3  ctl 181.7  ratio 4.30
    tanh          K·tanh(k/K)   rep 27.8  ctl  74.6  ratio 2.68

Direction is form-independent — no family moves any checkpoint across 1, and the
only sub-1 checkpoint under all three is the exception §4.2 already reports.
Magnitude is not: pooled ratio 2.7–4.3. "3.5-fold" was quoted with more precision
than the data carries; the range now sits beside it. **Do not** read three
agreeing curves as evidence that saturation is the right model — every family
here saturates by construction, and the decline past the plateau fits none.

*Trap:* the first run of horizon_forms.py defaulted to one extension CSV instead
of two and returned 4.13 for the fit the paper reports as 3.48. If a new script
disagrees with an old one on the same form, suspect the population before the
maths. The soft-horizon row now reproduces horizon_ext.py exactly, which is the
check that both see the same rows.

**4. Our exclusions do not make the gap (analysis/exclusion_sensitivity.py).**
~24% of panel generations are dropped under three rules. With all of them off:

    panel (as reported)            n=1234   gap 76.1
    + judge-unmeasurable template  n=1424   gap 62.6
    + budget-truncated items       n=1362   gap 76.7
    + degenerate audio (scored 0)  n=1242   gap 76.0
    nothing excluded at all        n=1620   gap 61.7

Not manufactured; flattered by ~14 points. Only the template rule moves anything
and it moves the **control** side (94.3 → 79.8), which is the direction the rule
predicts: a control made of many distinct words loses more to a word the judge
cannot transcribe than a repeated one does. The rule that could have flattered
us — dropping budget-truncated items — does not (76.1 → 76.7).

**5. VITS's immunity is not its stock config (analysis/vits_config.py).**
Perturbing the duration predictor, the component that would have to carry the
count:

    stock                            rep 67.6%  ctl 49.1%  gap −18.5
    noise_scale_duration 0.8 → 1.6   rep 53.7%  ctl 20.4%  gap −33.3
    speaking_rate        1.0 → 1.35  rep 61.1%  ctl 31.5%  gap −29.6

Gap is control minus repeated, so the panel's +76 means the control is counted
right and the repeated item is not. Every VITS arm is negative and the
repeated-side median relative error is exactly 0.000 in all three. *Read the sign,
not the size:* both perturbations hurt the control arm more, which is a judge
effect (distinct words cost more under fast/noisy speech), not a counting one.

**6. Checkpoint revisions are pinned (S19, data/results/model_revisions.json).**
All ten repositories including the CTC judge and the vocoder — the judge decides
every count, so a change to it changes every number. Regenerated from the
download cache, never typed.

## 2026-08-12 (final) — greedy decoding, and the deficit is not in the draw

Round 14 called this the cheapest ablation the paper could have run and the one
most relevant to its own premise, and they were right on both counts. Theorem 1
bounds a readout, so no property of the sampling rule enters the proof — yet
every generation in the panel was sampled.

Assumption 1 is what makes it matter. It posits a single time-invariant map F
between repetition boundaries. Stochastic token choice is exactly the per-step
perturbation that would break that autonomy: what gets sampled at repetition m
changes what conditions repetition m+1. Greedy removes the perturbation, so it is
where Assumption 1 is *most* defensible and the deficit has the fewest excuses.

    arm                    n    exact rep   exact ctl    gap
    sampled, three seeds  216      8.3%       83.3%     75.0
    sampled, seed 0        72      5.6%       83.3%     77.8
    greedy                 72     11.1%       83.3%     72.2

The deficit survives. The control rate is identical across all three arms, and
removing sampling noise moves the repeated side about five points. Whatever
produces the failure is in the conditioning, not in the draw.

**Checked, not assumed:** greedy on repetitive text hits a generation budget more
readily than sampling does, and truncated counts are censored downward — the
direction that would manufacture this result. Cap-hit rate is 0.0% on both Qwen
arms, so nothing was excluded on either side. (Llasa runs 9–18%, which is why the
rule exists at all.)

*Trap:* the first launch ran the whole 208-item stimulus file. Greedy hits the
token cap far more often than sampling, so it was heading for hours on families
the comparison never uses. Restart with `data/stimuli/stimuli_greedy.jsonl`
(word_rep + control_word only); generation resumes from the meta file, so nothing
is lost.

Limits: one checkpoint, one alternative decoding mode. Beam search untested.

## 2026-08-12 (final) — the horizon exception, accounted for

Two review rounds objected that Qwen3-TTS-1.7B is set aside as an "unexplained
exception", and that a theory which excludes the checkpoint contradicting it is
doing something a reader should worry about. The panel numbers answer it.

    checkpoint       exact gap   median-err gap   capacity gap
    Llasa-1B            78.1          12.5           38.5
    Llasa-3B            77.2           8.3           46.5
    Llasa-8B            81.4          16.7           30.5
    Qwen3-TTS-0.6B      92.2          12.5           14.0
    XTTS-v2             71.1          12.5           10.7
    Qwen3-TTS-1.7B      60.0           0.0            8.1

It ranks last on all three and is the only checkpoint that ranks last on any.
The exception is where the effect is smallest on every measure, and the horizon
ratio is the measure on which smallest tips into absent.

**What this does not license.** A weakest checkpoint exists in any panel and sits
nearest any threshold by construction, so this is *not* evidence for the theorem.
What it rules out is narrower: the checkpoint that fails to saturate is the one
with almost nothing to saturate, so the horizon result does not rest on
discarding a counterexample. S23 says exactly this — if you find yourself quoting
the ranking as support, re-read that paragraph.

Still open: *why* this checkpoint is weakest. Either its per-repetition map does
not contract, or its horizon lies past k=128 — which for the smallest deficit is
what you would expect. A longer ladder than its generation budget allows is what
separates them.

## 2026-08-12 (final) — third probe checkpoint: 1 of 3, and a threshold that nearly lied

Llasa-3B, run because two checkpoints is not a test, does not lose the count past
the horizon either (`analysis/probe_past_horizon.py`,
`data/results/probe_horizon_compare.json`):

    checkpoint  arm                    MAE   constant   ratio    R²
    Llasa-1B    repeated k=48..128    0.505    0.500     1.01   −0.02
    Llasa-3B    repeated k=48..128    0.494    0.500     0.99   −0.02
    Llasa-8B    repeated k=48..128    0.432    0.500     0.86   +0.19
    Llasa-1B    control  k=48..128    0.431    0.500     0.86   +0.26

**The threshold nearly told a lie in our favour.** "Beats the constant predictor"
put Llasa-3B on the *keeps the count* side by 0.006 MAE over twelve items. Read
naively that is "2 of 3 checkpoints keep the count"; read the other way it is
"2 of 3 show nothing recoverable", which flatters the theorem — and both readings
come out of the same number. The script now calls anything within `TIE = 0.02` of
the constant predictor **indistinguishable** and prints both counts. Under that
band Llasa-1B and Llasa-3B are indistinguishable from the constant predictor and
only Llasa-8B clearly keeps the count.

**We quote 1 of 3**, the strict threshold and the less favourable reading. The
more favourable one (2 of 3 null by effect size) is recorded in S12 beside it, so
a reader weighs it rather than discovers it. Anywhere this project says "1 of 2",
it predates Llasa-3B and is superseded.

The range control is unchanged and still holds: over the *same* k the probe reads
the count off control states (MAE 0.43 against 0.50 constant), so where the count
is lost, narrowness is not why.

*If you add a fourth checkpoint:* every verdict string and paper macro derives
from the lost/kept/tied lists, so the wording re-derives itself. Check
`PhKeptRatios` still reads sensibly — it lists every kept checkpoint, including
ones that merely tie.

## 2026-08-12 (final) — Lemma B's two deltas, labelled

Two different quantities were being quoted under one symbol, which is why the
draft read as inconsistent. Both are measured on the softmax **renormalised over
the k repeated keys** (raw attention is normalised over a sequence whose length
itself grows with k), on deep layers (`layer_frac > 0.6`) at `k >= 6`, from
`data/results/state.csv` via `analysis/make_numbers.py`:

    block entropy slope                 0.97 · log k   (predicted 1)
    within-block share slope           −1.00 in log-log (predicted −1)
    AVERAGE spread   max(log k − H)     δ ≤ 0.45 nats
    EXTREME spread   max share × k       3.4/k worst case → δ ≤ 1.23
                     median share × k    1.7/k typical    → δ = 0.51

**The lemma is entitled only to the extreme.** The entropy gap `log k − H` bounds
the *average* logit spread over the block; the most-attended occurrence is what
bounds the *worst* one, and it is the worst one the bound has to survive. Quoting
0.45 as though it were the bound understates δ by a factor of ~2.7 and makes the
two numbers look like they contradict each other. Report both, labelled, and lean
on 1.23 / 3.4·k⁻¹.

Note also that `attn_share` (the *mean* within-block share) is `1/k` by
construction and carries no information; the informative columns are
`attn_share_max`, `attn_unif_dev` and `attn_block_entropy`.

~~Earlier statement: "block entropy = 0.98·log k, δ ≤ 0.06 nats, max share ≤
1.9/k".~~ Those were measured before the renormalisation fix and before the
population settled; they do not reproduce from `state.csv`.

## 2026-08-12 (final) — the duration intervention: it fixes low k and nothing else

Every other result here is observational. This one intervenes on the
post-hoc hypothesis that replaced the withdrawn architectural claim — that what
matters is whether the model must represent "how many" internally. F5-TTS exposes
`fix_duration`, so the hypothesis can be tested instead of asserted
(`src/models/f5_fixdur.py`, `analysis/duration_intervention.py`,
`data/results/duration_intervention.json`, `behavioural_f5fix.csv`).

The supplied duration comes from F5-TTS's **own control renderings at the same
k**, which it counts correctly. Reading it off the repeated item's own output
would be circular.

    k      free exact   duration given   n/arm
    6         46.7%          80.0%        15
    8         40.0%          73.3%        15
    12        33.3%          40.0%        15
    16        20.0%          26.7%        15
    24         6.7%           0.0%        15
    32         6.7%           0.0%        15

    k < 12    43.3% -> 76.7%   (+33.3 pts, n=30 per arm)
    k >= 12   16.7% -> 16.7%   ( +0.0 pts, n=60 per arm)
    overall   25.6% -> 36.7%   (n=90 per arm, k>=6)

**The split is the finding.** Where the deficit is mild, supplying the total
length removes most of it; where the deficit is severe it does nothing at all —
at k=24 and k=32 the intervention arm is at zero. A model handed the correct
total length still cannot place thirty-two repetitions inside it, so at high k
the failure is not reducible to mis-estimating how much speech to make. Do not
paraphrase the low-k result as "duration explains it": +33.3 points is most of
the low-k deficit, not all of it, and the high-k gain is exactly zero.

*Unit trap, and it cost a whole run:* `fix_duration` in F5-TTS is the length of
the reference clip **plus** the generated speech, not of the generated speech
alone. Passing the target directly asks for a total shorter than the reference,
and the model duly emits near-silence — which is what the first run produced.
`f5_fixdur.py` measures the reference duration and adds it.

Limits: one non-AR checkpoint, one seed, 15 items per (k, arm) cell, and the
hypothesis under test is still post-hoc. A null would have retired the
hypothesis; a split does not confirm it.

## 2026-08-12 — Assumption 2 is not unmeasured. It is false.

Third attempt at the contraction premise, and the first with an estimator that
passes its own gates. `analysis/jacobian_q.py`: top singular value of the
boundary-to-boundary Jacobian by power iteration with **exact** directional
derivatives, teacher-forcing the generated tokens so the map is well defined.

**Self-tests, all passed before the model was touched.** A synthetic Jacobian of
known spectrum recovered to 3.8e-4. The same estimator run backwards in time
returns **exactly** 0.000000 against 43.89 forward — a causal transformer cannot
move information back in time, so this proves the hooks do not leak. Exact JVPs
match finite differences to 2e-5 over the step range where the map is linear.

**Result (Llasa-1B, 29 repeated + 29 length-matched control items, k in
{16,24,32}, seeds 0-2, 2724 usable measurements):**

| cell | repeated | control | paired p |
|---|---|---|---|
| tau, whole stack | **38.09 [32.88, 51.39]** | 31.84 [27.28, 39.84] | 0.017 |
| tau, depth>=12 | 5.46 [5.26, 6.52] | 4.63 [4.40, 4.96] | 0.0012 |
| 2 tau, whole stack | 62.25 [48.55, 78.10] | 34.81 [30.57, 46.61] | 0.0003 |

**q < 1 in 0 of 29 repeated items, in every cell, lag and sub-stack.** The
smallest number anywhere in the table is 4.63.

**And the sign is wrong too.** Repetition makes the map MORE expansive than its
length-matched control, not less: median paired difference +3.25 [+0.52,
+11.62], repeated lower in only 8 of 29 pairs. The premise does not merely fail
to hold; the data run against it.

Two declared biases both point the same way. Teacher-forcing deletes the token
channel, so the measured q is a **lower bound** on the true per-repetition
Lipschitz constant — which makes q >> 1 decisive rather than marginal. Nothing
plausible closes a factor of 38.

**N\* is not computable** (Eq. 1 is undefined for q >= 1). For the theorem to
place N\* at the observed saturation of 23 [16, 39] it would need q ~ 0.884
[0.850, 0.921]. Nothing measured is near that band.

**Independent corroboration of the old null.** The boundary-distance decay, with
boundaries derived from a uniform partition rather than from attention, gives
q-hat 0.981 [0.970, 0.989] repeated at **R^2 = 0.02** — i.e. q ~ 1 with no
geometric decay, the same answer the abandoned attention-based estimator gave.
The earlier null was not a localisation artifact.

**Structural finding, worth more than the number.** In a causal transformer a
same-depth single-vector state map is **exactly zero**, not small: information
reaches a later position only by rising through depth via attention. So the
theorem's `s_m` has no single-layer realisation at all, and only a whole-stack
(per-layer KV cache) formulation is well posed. Any future `h_l(t_m) ->
h_l(t_{m+1})` estimator measures nothing.

**Traps.** (1) The self-test's finite-difference gate fails at h=1e-3 for pure
float32 reasons — a unit vector spread over 2.5M coordinates moves each by 1e-6,
i.e. roundoff divided by h. This is very likely the wall attempt 2's linearity
control hit. Over h in [0.3, 10] the map is linear to 0.01%. (2) tol=1e-8 is
unreachable in float32; the estimate was right to 1e-4 while flagged
unconverged. (3) fp32 + math-SDPA + double backward peaks at 45 GB by 2100
tokens on a 49 GB card. (4) The rendered count is not k on the repeated arm, so
tau = T/k is a construction, not a measurement — hence the lag sweep, which
changes nothing.

**Status: the Track-A gate is closed.** The paper now reports the premise as
measured and refuted rather than untested, which is a stronger claim and a
worse one for the theorem.

## 2026-08-12 (final) — the probe panel doubles: 1 of 5, and the second family inverts

`analysis/probe_past_horizon.py`, now over five checkpoints and with a matched
control arm on **every** one of them, not just Llasa-1B.

    checkpoint       repeated k=48..128        control k=48..128
                     MAE   const  ratio  R²    MAE   const  ratio  R²
    Llasa-1B        0.505  0.500  1.01  −0.02  0.431  0.500  0.86  +0.26
    Llasa-3B        0.494  0.500  0.99  −0.02  0.413  0.500  0.83  +0.33
    Llasa-8B        0.432  0.500  0.86  +0.19  0.293  0.500  0.59  +0.56
    Qwen3-TTS-0.6B  0.370  0.500  0.74  +0.34  0.474  0.500  0.95  +0.10
    Qwen3-TTS-1.7B  0.448  0.500  0.90  +0.15  0.444  0.500  0.89  +0.20

Llasa-1B and Llasa-3B remain inside the `TIE = 0.02` band. The prediction now
holds in **1 of 5**, down from 1 of 3; three checkpoints clearly keep the count
where one did before. Widening the panel made the result worse, again.

**"Scale buys a persistent counter" does not survive the second family.** Within
Llasa the ratio falls with size (1.01 → 0.99 → 0.86), which is what the slogan
described. Within Qwen it *rises* with size (0.6B 0.74 → 1.7B 0.90): the smaller
checkpoint is the one whose states the probe reads best. Two families, two
directions, three points each. Do not write the scale story.

**The range rebuttal now generalises.** All five control arms beat the constant
predictor (0.86, 0.83, 0.59, 0.95, 0.89), so on every checkpoint that lost or
tied the count, narrowness is excluded on that checkpoint's own data rather than
by analogy with Llasa-1B's. That is the one thing here that came back stronger.

**But do not read the two Qwen ratios as "the count survives in Qwen states."**
A predictor given *only* the log length of the generated trajectory — no hidden
states at all — beats the state probe on both:

    length-only MAE vs state-probe MAE (repeated arm)
    Llasa-1B  0.61 / 0.51    Llasa-3B  0.56 / 0.49    Llasa-8B  0.45 / 0.43
    Qwen-0.6B 0.30 / 0.37    Qwen-1.7B 0.36 / 0.45

The reason is visible in the generations: 6 of Qwen-0.6B's 12 repeated items run
to the 8192-token budget, and they are the high-k ones (0/1/3/2 at k=48/64/96/128),
so "how long did it babble" carries k by itself. The constant predictor is the
wrong yardstick for those two checkpoints; against a length-only baseline they
have learned nothing either. The Llasa checkpoints are not exposed this way —
their probe beats length on all three.

**A budget bug underneath it (`src/models/qwen_gen.py`).** `hit_cap` is computed
as `n_steps >= max_new_tokens` where `n_steps = len(hidden_states) - 1`, which is
one short, so it is **False on all 1282 Qwen rows ever generated** while 9
qwen06b and 5 qwen17b rows sit exactly at the budget (`n_speech_tokens` 8191 or
2047). Population rule 4 has therefore never excluded a single Qwen item. Six of
those rows are original extension-ladder items and feed `behavioural_ext.csv`,
so the qwen06b (15.0) and qwen17b (0.27) horizon ratios are computed over items
that rule 4 was supposed to drop. Not fixed here — recomputing `hit_cap` for
existing rows changes published numbers and is a decision, not a patch.

---

> **The entries below backfill several commits that landed changes without a
> matching findings.md entry.** They are added 2026-08-12 (late) while
> reconciling this file, `research-state.yaml` and `implementation-notes.md`
> against ~15 commits of paper and analysis work. Presented in the order the
> underlying commits actually landed; every number below was re-checked
> against its `data/results/*.json` at the time of this reconciliation, not
> copied from a commit message.

## 2026-08-12 — the judge replication: four recognisers, and the decisive number is the arm spread

The CTC-judge objection had survived five review rounds (eight of nine
reviewers in r15–r17, two more in r18): the judge was validated against
concatenative audio with known counts and never against the real generated
failure audio it scores, and CTC blank-collapse — which merges adjacent
identical words — is exactly the confound under study. `analysis/
independent_judge.py` re-scores the same 982 generations with three more
recognisers (`data/results/independent_judge.json`):

    wav2vec2-large-960h-lv60-self  (primary)      +76.7
    hubert-large-ls960-ft          (independent)  +72.3
    whisper-large-v3 (no blank-collapse mechanism at all)  +75.5
    wav2vec2-large-robust-ft-libri-960h (weakest)  +65.1

Positive in 6 of 6 checkpoints and 3 of 3 families under every one of the four.

**The decisive number is not the gap.** Across the four judges the repeated
arm's exact rate spans **1.0 point** (17.1–18.2%) while the control arm's spans
**12.2** (82.1–94.3%). A judge-side collapse of repeated material predicts the
opposite pattern — the arm that would move is the one the models get *wrong*,
not the one they get right. It is not there: swapping the scorer moves the arm
the models already get right (ordinary word-error-rate noise) and leaves the
arm they get wrong almost exactly where it was.

Disagreement between judges is real and is reported, not buried: judges
disagree 3.3x more on repeated items than on controls, growing with k (exact
agreement on repeated items falls 94–97% at k≤8 to 64.4% at k=32). But its
**sign is wrong for the confound**: the primary judge reports the *higher*
count in 61.2% of repeated-item disagreements and 72.9% of control ones (65.9%
pooled across all k) — it under-states the deficit, not manufactures it. Under
the strongest form of the objection (either judge counting as exact) the gap
*widens* to +76.2, not narrows.

**Correction this forces.** `noise_floor.py`'s "our judge reports the higher
count in 11 of 13 disagreements" was computed at n=60. At full n (173
disagreements out of 1559 paired rows) it is 61–73% depending on arm, not the
~85% that 11/13 implies. Still the right side of 50, but do not requote 11/13;
it is superseded and S14 marks it so in place.

Limits: I could not trace the commit message's separately-quoted "+59.4"
figure (restoring the excluded template) to a single field in
`independent_judge.json` — the closest candidates are the primary judge's
template-inclusive gap (62.6) and the independent judge's (58.5). Left out
rather than guessed at; if you need that number, recompute it directly rather
than trusting this paragraph.

## 2026-08-12 — the stop-head instrument is clean, and the result is negative

Asked what the stop decision actually reads: the decoded repetition count, or
elapsed duration. `analysis/stop_head.py`, `data/results/stop_head.json`. The
instrument is sound, which makes the negative harder to dismiss: EOS logits
are recovered exactly (0.075 nats against the instrumented pass's own values),
all four planted positive controls are recovered with zero spurious
attribution on both checkpoints, and the commonality decomposition is additive
to 1e-16.

The result fails on its own pre-committed criteria. **Llasa-1B**: predictors
**inseparable** — canonical correlation between duration and count is 0.970 at
low k, above the 0.95 threshold fixed before any number was read. **Llasa-8B**:
**inconclusive** (0.635). The two checkpoints disagree on which direction the
collinearity even moves — 1B falls 0.97→0.73 low-to-high k, 8B *rises*
0.47→0.64 — so "duration and count decouple at high k" is true on one
checkpoint and false on the other.

One structural finding survives the null: log P(EOS) is a flat floor with a
single spike at the stop step, so at the level of the stop *event* (not the
per-step regression) elapsed duration is not a rival hypothesis, it is the
outcome variable. There, only Llasa-8B shows the predicted dissociation
(repeated under-proportional at high k); Llasa-1B shows the opposite pattern.

An alignment bug was found and fixed while building the instrument:
`hidden`/`attn_*` were sliced `[plen:]` while `entropy`/`top1` came from
`logits[plen-1:-1]` — off by one generation step. It failed quietly (0.075
nats shifted vs 3.55 unshifted), which is why it is recorded here rather than
assumed impossible.

**Design limitation this makes unavoidable, now stated in the paper**: inside
the repeated arm, the requested count and the amount of text are one variable,
so a probe decoding k from decoder states cannot be said to decode a count
rather than a length. Only the matched control separates those, and only
behaviourally. This constrains every representational claim in this project.

## 2026-08-12 — the qwen `hit_cap` fix landed: horizon ratio moves in our favour

Follow-up to the bug recorded immediately above (in the previous entry): the
decision was made to repair the ledgers rather than leave the bug live. Fixed
from `n_speech_tokens`, which was always recorded correctly, with the budget
inferred per item from whether it is an extension-ladder id.

    pooled soft-horizon K_rep    30.1  ->  24.8
    pooled ratio                 3.48  ->  4.22
    hyperbolic form ratio        4.30  ->  5.40
    tanh form ratio              2.68  ->  3.14
    form range                  2.7-4.3 -> 3.1-5.4
    per-checkpoint: llasa1b 5.97, llasa8b 4.12 (unchanged, not Qwen);
                    qwen06b 15.0 -> 13.1 (still unbounded at the fit ceiling);
                    qwen17b 0.27 -> 0.29 (still inverted)

Control-side K_ctl does not move, which is the check that the repair did what
it claimed — the truncated items were on the repeated arm only. **The
correction moves the headline in the paper's own favour**: truncated repeated
items were inflating the repeated arm's fitted saturation scale, so the
deficit had been understated. Verified directly against
`data/results/horizon_ext.json` and `data/results/horizon_forms.json` while
reconciling this file: pooled soft-horizon `N*_rep 24.80 [22.51,42.41]`,
`N*_ctl 104.77 [93.28,114.78]`, ratio `4.224`; per-form pooled ratios
`3.142 (tanh) / 4.224 (soft-horizon) / 5.403 (hyperbolic)`.

Every "N\*_rep 30.1" / "ratio 3.5" / "qwen06b 15.0" / "qwen17b 0.27" / "form
range 2.7-4.3" statement anywhere above this line, and in `research-state.yaml`
before this reconciliation, is superseded by the numbers in this entry. Do not
requote the old ones.

## 2026-08-12 — the causal intervention: disruptive, then readable, and a dead lead

The discussion had said the state-vs-policy manipulation was one "we did not
run." Two arms on Llasa-1B, `analysis/causal_count.py`,
`data/results/causal_count.json`.

**Whole-state splicing (Arm A).** Patches hidden states from a donor at
matched positions during teacher-forced generation, then resumes. Sanity gates
pass (no-op patch bit-identical, n=18). The verdict is neither hit nor null:
states spliced from an **unrelated** item move the rendered count 1.34x as much
as a donor that actually differs in k (cap-hit 15.8% reference vs 37.4%
patched; stop-rate 84.2% vs 62.6%). A protocol in which a random donor is more
disruptive than an informative one cannot distinguish "the count is not
causally used" from "we broke generation." Verdict: **"too disruptive to
interpret."** Arm B (difference-in-means steering, pilot) failed all three of
its own pre-committed gates.

**The fix: a rank-1 patch.** Replacing whole-state splicing with a rank-1
projection that transfers only the probe's count coordinate
(`data/results/causal_count_followup.json`) removes the damage: 0% degenerate
in every cell (against non-trivial degeneracy in the whole-state/steering
pilot — e.g. one steering cell ran 88.9% degenerate before this fix), no-op
self-patch bit-identical 18/18 (max delta exactly zero, which needed a bf16
numerical trap fixed first — see `implementation-notes.md`), and per-cell
stop-rate now within 3–17 points of the reference arm (aggregate 73.5% patched
against 83.3% reference) where the whole-state pilot's aggregate stop-rate sat
21.6 points below reference (62.6% against 84.2%, with individual cells as far
as 33.3 points below). **And the
count still does not move**: cross-k at layer 7, `+0.00` log-count
`[-0.09,+0.09]` over 36 pairs, no opposite-sign up/down pattern in any cell.
This is now reported as a **readable null** — the count coordinate transplants
cleanly and the decoder ignores it — rather than as an uninterpretable
intervention, and it is direct causal support for the output-policy account
that the probe evidence could only gesture at.

**The ridge-steering lead died.** A full alpha sweep — 12 repeated items and
matched controls, 9 alphas, 274 generations — returns Spearman ρ = +0.044,
p = 0.65 (partialling out duration: +0.050). The repeated curve is not
monotone but *peaked*: ~15–16 rendered across alpha in [-0.5, +0.25],
collapsing to 3–6 at both extremes. Controls stay immovable — 24 rendered
units at every alpha. The earlier +0.96 pilot contrast reproduces in
magnitude but reads the two ends of a curve whose middle contradicts it: the
finding that survives is that repeated-text generation destabilises under
perturbation in *either* direction while aperiodic control text does not — a
fragility, not a knob.

**Reported against interest.** The pre-committed P1 disruption gate on the
rank-1 patch still literally returns "too disruptive" at 1.74 — stated as
such, but flagged post-hoc as a degenerate ratio: both arms' extreme-alpha
effect medians are exactly 0.0, so the gate divides by ~nothing rather than
detecting violence.

**This closes the last of four mechanistic candidates.** Contraction: measured
and refuted (see below). Dilution dose-response: disconfirmed, sign-reversed.
Stop-head readout: inseparable / inconclusive (previous entry). Causal
intervention: a readable, causal null. There is no mechanistic spine available
on this evidence, and the paper does not pretend otherwise.

## 2026-08-12 — the mechanism stands nowhere, plainly

Summary that supersedes every earlier "unestablished" / "nothing directly
supports the causal story" framing in this file (both marked superseded in
place above, not deleted): as of this entry, all four mechanistic candidates
this project ever raised have been *tested to a conclusion*.

1. **Contraction** — measured and refuted on Llasa-1B: q = 38.1, not < 1;
   control arm expansive too (31.8), so the premise never described this
   decoder for any input. See "Assumption 2 is not unmeasured. It is false."
2. **Dilution dose-response** — disconfirmed, sign-reversed (r=+0.59).
3. **Stop-head readout** — negative: inseparable on Llasa-1B, inconclusive on
   Llasa-8B.
4. **Causal intervention** — a readable null: the count transplants cleanly
   (rank-1 patch) and the decoder ignores it.

What stands is the *phenomenon* (the dissociation, replicated across four
judges and, as of this reconciliation, a fourth architecture — see below) plus
the excluded rivals, and the causal half of the state/policy dissociation: the
decoder represents the count where the horizon permits a readout, and its
output policy demonstrably does not use that representation. That is a
narrower, better-supported claim than "a horizon it does not yet explain," and
it is what the retitled paper (*"Periodicity, Not Length: Locating the
Repetition-Counting Failure in Neural Text-to-Speech"*) now argues.

## 2026-08-12 — CosyVoice 2, a fourth architecture, and the hierarchical statistics that predicted it

**Hierarchical statistics** (`analysis/hierarchical.py`,
`data/results/hierarchical.json`), landing the honest n=6 inference seven of
nine reviewers asked for instead of the "too small for significance testing"
apology:

- Mixed-effects linear-probability model (architecture as a random effect):
  arm gap **76.4 points**; six interval constructions (Wald, cluster-family,
  cluster-checkpoint, GEE, family-t, two cluster bootstraps) all exclude zero,
  **least favourable lower bound 63.2 points** (cluster-checkpoint). The
  Wald/REML interval is narrower and marked reference-only: with only 3
  families, REML over-shrinks the family-varying arm-slope SD (1.4 points
  against 3.9 observed between families).
- Hierarchical Bayes (NUTS, r-hat ≤ 1.0005, ESS ≥ 3825; cross-checked against
  an independent hand-rolled Metropolis sampler, r-hat ≤ 1.026): panel
  posterior **74.9 [41.5, 89.4]**; every per-checkpoint and per-family
  posterior has P(gap>0)=1.0; **posterior predictive for an architecture
  outside the panel: 65.3 [5.9, 93.3], P(gap>0)=0.996.**
- A 420-specification curve (240 unique populations after collapsing
  duplicates) over exclusions × kmin × control-type × seeds × aggregation ×
  outcome: exact-rate gap ranges **34.4 to 86.4 points**, sign never reverses.
- **The one reading anywhere in this project that touches zero**: under the
  single widest group-level-scale prior tried (HalfNormal(3.0)) for the
  out-of-panel posterior, the lower bound is −1.63 and P(gap>0) drops to
  0.956. Report this when the question is "does *any* credible reading touch
  zero" — the answer is yes, once.

**CosyVoice 2**, alignment-supervised and autoregressive, lands as a fourth
architecture (`src/models/cosyvoice_gen.py`,
`data/results/behavioural_cosyvoice.csv`) — scored and reported separately
from the panel, excluded by name in `population.py`'s non-panel set so a
routine rerun of `run_pipeline.sh` cannot silently fold it in. It is a genuine
held-out test of the fit above: the out-of-panel posterior predictive (65.3
points) was committed in 6f34945 at 17:28, **before CosyVoice 2's first
waveform existed** (~17:50, per the commit record; `logs/cosyvoice2_*.log`
starts at 17:59 and `behavioural_cosyvoice.csv` itself is not scored until
18:58). Observed gap:
**64.4 points** (28.9% repeated exact vs 93.3% control, n=90/arm, k≥6 — I
recomputed this directly from `behavioural_cosyvoice.csv` through
`population.panel()` while reconciling this file and it reproduces to one
decimal).

It is the architecture design most likely to be immune (alignment supervision,
a shipped repetition-aware sampling rule) and is not immune — but it **fails
differently**: it *over-produces* rather than truncating (median repeated
relative error **+12.5%**, i.e. too much speech, not too little) at a **0%**
cap-hit rate. Its generation ceiling scales with the requested text rather
than being a fixed budget, so it makes roughly the right amount of speech and
still loses the count. Report this as a **held-out prediction**, not a
pre-registration: generation was already queued when the fit was committed, so
the architecture was chosen and only its result was unknown. The predictive
interval is ~90 points wide, so landing within 0.9 points of the point
estimate is luckier than the model is entitled to claim credit for.

## 2026-08-12 — the r19 correction: single-checkpoint scope, and a sharper reading of the contraction result

Four reviewers on the substantially-changed paper independently caught the
same thing: the jacobian and rank-1-patch results above are single-checkpoint
(n=29 and n=36, both Llasa-1B) and had been written as panel-wide conclusions
("false for these decoders", plural; "never described this decoder" used to
withdraw the contraction account across all six checkpoints). **Corrected
scope, and the correction applies retroactively to every entry above in this
file that says "these decoders" or implies a panel-wide result**: false for
the *one* decoder measured, refuting the mechanism where it was tested, not
panel-wide. A second architecture family's jacobian is running as of this
writing (see `research-state.yaml`).

**A sharper, less flattering reading of the same number**, also from review:
the *control* arm's Jacobian is expansive too (31.8, not < 1). That means the
contraction premise was never a live description of this decoder for *any*
input — it is not "periodicity breaks contraction," it is "there was no
contraction here to break." This is now what the abstract and Section 4 say,
in place of the earlier "repetition makes the map more expansive" framing.

Two smaller corrections from the same round, recorded because a claim that
moved should show that it moved: (1) the CosyVoice-2 held-out prediction's
provenance sentence pointed readers at the wrong supplement location, which is
now fixed with exact timestamps (hierarchical.json written 17:15, committed
17:28, first CosyVoice-2 waveform 17:50, scored 18:58) — the underlying claim
was already true, only the citation was wrong. (2) The judge asymmetry (+0.27)
had been described as "a residual in the confound's direction, just too
small," which is backwards: +0.27 means our judge reports the *higher* count
on repeated items, the opposite sign from what a blank-collapse confound would
produce. It now says "opposite sign to the confound," not "under-states."

The two "confirmatory" subsection headers were renamed "pre-specified" — the
project fixed the stimulus ladder, controls and exclusion rules before
generation (S16 dates this), but never pre-registered, so "confirmatory" was
claiming more than that supports.

---

> **The entries below backfill twelve commits (`1790187`..`c24a2b1`) that
> landed changes without a matching findings.md entry.** Added 2026-08-13
> while reconciling this file, `research-state.yaml` and
> `implementation-notes.md`. Every number below was re-checked directly
> against its `data/results/*.json` during this reconciliation, not copied
> from a commit message — where a commit message and the JSON disagreed, that
> is called out explicitly rather than silently resolved.

## 2026-08-12 (later) — the never-cycled control now covers the whole panel, and the headline moves against us

The control-periodicity correction (see the 2026-08-12 "three rivals excluded"
entry above) had only run on 4 of 6 checkpoints; `scripts/run_aperiodic_llasa.sh`
fills in Llasa-3B and Llasa-8B (144 new generations) and
`analysis/aperiodic_controls.py` recomputes the headline with the never-cycled
arm substituted in for all six.

    exact-rate gap, k>=6
    by checkpoint    69.6 pts (never-cycled)   vs 76.7 (cycled)
    by family        67.6 pts (never-cycled)   vs 75.4 (cycled)

Positive in 6/6 and 3/3 either way, smallest checkpoint still +60. **The paper
now leads with the smaller, de-confounded number** in both the abstract and
Section 3 (verified against `paper/numbers.tex`: `\ApHeadCk`=69.6,
`\ApHeadFam`=67.6, cycled kept alongside as `\ApHeadCkCyc`=76.7 /
`\ApHeadFamCyc`=75.4, and the abstract text leads with `\ApHeadFam` and gives
the cycled figure second).

**But most of the ~7-point move is not periodicity, and saying so matters as
much as the correction.** A never-cycled control at k=32 must return
thirty-two *distinct* adverbs, and the 146-word pool's tail transcribes worse
than the eight words the ladder cycles (0.79–0.85 against 0.98–1.00
per-occurrence). Scored to within one unit, the substitution costs 1.7 points,
not 7.1 — the median-error contrast (which Figure 1(a) plots) does not move at
all. So 69.6/67.6 are a *floor*, the truth is bracketed between them and
76.7/75.4, and both numbers are reported rather than one replacing the other.
The old 4-checkpoint k=12..32 subset figure ("13.4 points") becomes 11.4 points
over all six.

**The hierarchical/mixed-effects fits still rest on the cycled arm, on
purpose** — `hierarchical.py` asserts against `checkpoint_level.json`, which is
cycled, and would abort on a substituted frame. Stated rather than papered
over: the substituted headline sits beside the modelled one, not in its place.

Traps recorded in `implementation-notes.md`: `run_aperiodic.sh` scored only the
four models it generates into a file it overwrites, so a naive re-run would
have deleted the Llasa-3B/8B rows; `xcodec2_decode.py` / `asr_ctc.py` needed
`--glob` because both Llasa checkpoints carry 24 undecoded instrumentation-only
token files that a bare decode would have vocoded into an unscored population.

## 2026-08-12 (later) — the contraction refutation generalises: five checkpoints, two families, no exceptions

All four r19 reviewers raised the same objection: q was measured on Llasa-1B
alone and the conclusion was stated panel-wide. It now covers five checkpoints
across both architecture families (`data/results/jacobian_q_panel.json`,
verified directly): 346 items, 14,796 measurements.

    llasa1b   q= 38.1 [32.9, 51.4]   control 31.8
    llasa3b   q=118.8 [101.9,156.5]  control 86.4
    llasa8b   q=347.7 [258.1,393.5]  control 185.4
    qwen06b   q= 21.0 [19.7, 24.2]   control 21.9
    qwen17b   q= 24.7 [23.1, 25.7]   control 23.6

`q<1` in **zero** items, in every checkpoint, every lag, every sub-stack, on
both arms. The most favourable cell in the entire study still has a bootstrap
floor of 3.3, and teacher-forcing makes every one of these numbers a *lower*
bound on the true per-repetition Lipschitz constant.

**Two facts sharpen this past mere replication.** (1) `q` **grows with scale
inside Llasa** — 38 → 119 → 348 at 1B → 3B → 8B — which excludes, in the
strongest available direction, the reviewer's "maybe only the larger
checkpoints contract." (2) **Qwen3-TTS-1.7B counts correctly** (the panel's
only zero-median-count-error checkpoint) **and has q=24.7**, as expansive as
everything else. Whatever separates counting from miscounting on this panel,
it is not q.

Per-arm: both arms are expansive in all 5×16=80 lag×sub-stack cells; the
repeated-vs-control contrast is significant in the *wrong* direction on
Llasa-1B only (p=0.017 — see the note below on why this p-value was later cut
from the prose) and null on the other four (p=0.269, 0.256, 0.931, 0.177 for
llasa3b/llasa8b/qwen06b/qwen17b respectively). So the deflationary reading
generalises too: **contraction was never a description of any of these five
decoders, for any input.**

Getting Qwen into this estimator needed a fourth self-test gate: it has no
token sequence to teacher-force (each step's input sums 16 RVQ codec
embeddings plus a text term, and codec ids are never saved), so a pre-hook
captures the actual fused input and is required to reproduce incremental
generation to cosine 0.999 before the model is trusted. (The commit message's
"88/88 and 90/90" ledger-agreement figure for this check could not be traced to
a field in any committed JSON during this reconciliation — used with its
provenance flagged, per the same convention `implementation-notes.md` already
applies to its "~51% of steps" figure.)

Reported against interest: Llasa-8B skipped 22% of its measurements at a
700-token read window on one card. **That 22% is a mix, not one story** — a
follow-up correction (`b86137c`) found only 236 of 365 skips are exact
read-window truncation (fine on a causal model); **129 are plain OOM**, which
the read-window argument does not cover. The supplement now reports both
counts separately rather than the single explanation an earlier pass gave.
Also: Llasa-3B ran before a `MIN_TAU` sample-rate patch that matters at Qwen's
12.5 Hz (an 8-frame floor is 0.16 s at Llasa's 50 Hz but 0.6 s at Qwen's
12.5 Hz, which would have silently discarded nearly every Qwen item) — the
patch's own bias runs toward the premise, not against it.

**Scope note: this supersedes the single-checkpoint caveat this file and
`research-state.yaml` carried until this reconciliation.** Any earlier
sentence here reading "false for the one decoder measured (Llasa-1B), not
panel-wide" is retired by this entry.

## 2026-08-12 (later) — the chain fails at both ends, and delta was measured all along

A reviewer made a sharp point the paper's own page-budget cuts had made
possible: Section 4 measures `q`, the *last* link of the derivation chain
(bounded-delta logits → near-uniform attention → autonomous map `F` →
contraction), and concludes the premise "was never a description of these
decoders" — but testing only the final link cannot distinguish "the map exists
and does not contract" from "the map the derivation constructs never applied
at all."

**The fix was not a new experiment.** Delta was measured long ago and the
`\DeltaMax` macro had been sitting in `numbers.tex`, defined and unused, since
an earlier page-budget squeeze removed the sentence that quoted it. Verified:
`\DeltaMax{}` = 1.23 nats, `\DeltaRatio{}` = 3.4-fold, now quoted directly in
`paper/main.tex` ("$\delta\le\DeltaMax{}$ nats — a $\DeltaRatio{}$-fold weight
ratio across the repeated spans"). This is the same 1.23-nat "extreme" spread
this file already recorded under "Lemma B's two deltas, labelled" — what
changed is that it is now *in the paper's prose*, not just in a macro nobody
quoted.

**So the honest statement is stronger than the one the paper was making: the
chain fails at both ends.** The flattening Lemma B needs is not there (delta
is 1.23 nats / 3.4-fold, not the near-uniformity the lemma assumes), and the
map that flattening would license does not contract either (q ≫ 1
panel-wide). Do not write "the premise fails" as though the contraction
measurement alone carried that claim.

Paid for from the same section: the dilution dose-response paragraph is now a
pointer rather than three lines — with delta reported, the lemma's premise is
already shown to fail, and the dose-response is (per the same reviewer) a
correlational patch on a different question.

## 2026-08-12 (later) — r22: two internal contradictions, both self-inflicted

Three reviewers on the ladder-era draft caught two contradictions the paper
had made with itself.

**Sections titled "Pre-specified" while the text admitted the threshold was
not.** The two results subsections were relabelled "Pre-specified" in response
to an earlier round, then, in response to a *later* round, the text was
honestly amended to disclose that the k≥6 threshold was chosen after seeing
where the arms diverge. Both were right individually; together they
contradicted. Headers are now plain descriptions (verified: `paper/results_body.tex`
now reads "The limit tracks periodicity, not length" and "Not the decoding
rule, the judge, or our exclusions" — no "Pre-specified" or "Confirmatory"
anywhere in the built paper), with the provenance staying in the prose.

**The abstract led with a number the body says it does not stand behind.** The
body reports the never-cycled gap and says "we quote the smaller"; the
abstract opened with the cycled 75.4 instead. Now the abstract leads with 67.6
(never-cycled, by family) and gives 75.4 second — verified directly against
`paper/main.tex`'s abstract text, which reads `\ApHeadFam{}` before
`\ApHeadFamCyc{}`.

**A stray p-value contradicted the paper's own stated standard.** The paper
says its claims rest on effect sizes, not p-values, then quoted p=0.017 for
the Jacobian sign contrast. Dropped from the prose (the claim there is q ≫ 1,
two orders of magnitude past the threshold, which does not need a p-value to
carry it) — the `\JacQPairedP` macro still exists in `numbers.tex` (still
0.017, matching `jacobian_q_panel.json`'s llasa1b `paired_wilcoxon_p`) but is
no longer referenced anywhere in the built paper.

## 2026-08-12/13 — the period ladder: the title stands, and it stands on evidence

The sharpest objection the paper had received, raised independently by all
four r21 reviewers: the design varies periodicity and verbatim token identity
together (period-1 verbatim against aperiodic non-verbatim, nothing in
between), so the title claims the first and the evidence supported only the
pair. A ladder holding carrier, word count and requested count fixed while
varying only the period `p ∈ {1,2,4,8}` at `k ∈ {16,24,32}`, three checkpoints
(llasa1b, qwen06b, xtts2), settles it (`data/results/period_ladder.json`,
verified: `kept`=1134, matching this file's `n=1134`).

    p = 1     7.2% exact
    p = 2    49.2%
    p = 4    77.4%
    p = 8    91.6%

(These are the unweighted means across the three checkpoints of
`exact_by_period_scan`, recomputed directly from the JSON during this
reconciliation and reproducing to 0.1 point.)

**Monotone in every one of the three checkpoints individually**, n=1134, all
five — actually four, see below — scoring rules agree on the ordering. The
decisive cell is p=2, where no token is ever adjacent to itself: 42.4 points
below p=1's rate under three of the scoring rules, and roughly half the
deficit still present. **So the title stands, and stands on evidence rather
than assumption.**

**Caution, stated at the same weight as the result.** The headline scoring
rule (ordered scan) clears its own pre-committed "half survives" bar
(`D(2)/D(1) ≤ 0.5`) by only **0.002** (`mean_ratio`=0.5019 in the JSON), and
does so *only because one checkpoint carries it*: per-checkpoint ratios are
llasa1b 0.426, xtts2 0.430 — **both below half** — with qwen06b's 0.6295 alone
pulling the mean over the line. The two recount rules (unbounded recount,
strict conjunction) are far more comfortable: per-checkpoint ratios span
0.6895–0.8083 across both rules. **The paper's periodicity claim rests on the
recount rules, not on the ordered scan clearing its bar by a hair.** The gap
between scan and recount exists because `score_counts.py`'s ordered scan
cannot detect an *overcount* once p>1 — a looping model reads as an overcount
at p=1 and as spuriously exact at p=2, worth 22 points at p=2 on its own
(51.7% scanned vs 23.9% recounted). Every rate is now reported under three
rules for exactly this reason.

**"Five scoring rules" in an earlier commit message was actually four**: the
by-family rule reproduces the ordered scan exactly on this ladder, because
every architecture family here has exactly one representative checkpoint.

**The replication check is not uniformly exact.** qwen06b and xtts2 reproduce
their published p=1/p=8/p=k rungs to 0.000; llasa1b — the smallest-n checkpoint
— differs by −2.9, +2.8 and +1.7 points. Inside sampling noise, reported as
such rather than as a blanket "reproduces exactly."

**Bootstrap caution, confirmed against source.** `bootstrap_ratio` is called
on the strict-conjunction column, so its interval (ratio [0.633, 0.824], D2
[0.483, 0.729]) belongs to a rule whose own point estimates are 0.738 / 0.612
— *not* to the 0.502/0.424 ordered-scan headline. Do not pair the two.

**Rotations were not optional.** A p=2 arm needs two fillers, and which two is
a lexical confound: the observed spread across the four rotations of one pool
is 24.4% to 53.3% exact on qwen06b alone (verified in `p2_by_rotation`).
Generating all 8/p rotations balances vocabulary and character count
(187.8/188.6/188.8 chars) across p=2,4,8 — not planned, and worth keeping. A
single-rotation design would have carried a ~30-point lexical confound into
the paper's decisive cell.

### And the individuation account is dead — our own proposed mechanism, tested and killed

We had argued — from VITS's immunity, F5-TTS's duration split and the causal
null — that the decoder cannot individuate identical neighbours, and that a
minimal local disambiguator (comma, full stop, or "and" between repetitions)
should therefore recover the count without changing it.
`data/results/disambiguation.json`: verdict **"INDIVIDUATION FALSIFIED"**.

**Number discrepancy, resolved in the JSON's favour (house rule: trust the
JSON over the commit body).** The commit that landed the final version of this
file says "the recovered fraction is 0.9%, not the 0.2% I quoted [earlier] —
the JSON moved as data landed and the macro tracked it." **This does not
match what is on disk.** The current `data/results/disambiguation.json`
(n=465, all three checkpoints present, matches `HEAD` exactly, re-verified
during this reconciliation) gives `R_matched = -0.00021352`, i.e. **−0.02%**,
which is exactly what is baked into the built paper: `paper/numbers.tex:122`
defines `\DisR{}` = **"-0.0"**, used at `paper/results_body.tex:82` ("...
recovers `\DisR{}`%."). Neither the earlier 0.2% (itself correct for its own
moment, computed on an n=179 two-checkpoint precursor) nor the claimed-new
0.9% is what is on disk or built into the paper today. **The correct number is
−0.0% — no meaningful recovery, direction if anything slightly negative.**
Do not requote 0.9% anywhere this project's writing continues.

A second account died alongside individuation: per-token repetition count
`m = k/p` does not explain the curve by itself — repeating each of four words
eight times each is much easier than repeating one word eight times, despite
twice the length at the same `m`. Qualitatively supported by
`period_ladder.json`'s `collapse_onto_m.rank_corr_E_m = −0.739` (exact rate
falls as `m` rises); the specific "66.7% vs 0.0%" figures quoted for this
comparison in commit prose could **not** be traced to a specific JSON field
during this reconciliation — the ladder's `k ∈ {16,24,32}` design has no
`p=1,k=8` cell to pair against `p=4,k=32`'s `m=8`, so the comparison likely
draws on the separate main-panel `k=8` word-repetition rows rather than the
period-ladder JSON itself. Recorded as unverified rather than silently
requoted.

**Traps, kept because they will bite again.** `score_counts.py` cannot see an
overcount above p=1 (see above). `qwen_gen.py`'s `eos_trim_length` indexed
`hidden_states[j+1]` for `j` up to `n-1`, out of range whenever generation
stops without materialising an EOS frame — on transformers 4.57.3 that is
*every* item, and the arm crashed on its first generation; now guarded. The
`coqui` env can no longer import coqui-tts at all under transformers 5.15.0,
so XTTS ran in the `coqui_es` clone, the same pin the published run used.
Both fixes are documented in `implementation-notes.md`'s "Round 21" section.

**Still open, and worth restating plainly for whoever resumes next:** the
ladder covers only three of the six panel checkpoints. A period-ladder
extension to llasa3b was mentioned in project chatter but **no
`data/results/*llasa3b*period*` file exists on disk as of this reconciliation**
— do not assume it landed. If it does, S28 in the supplement needs
re-checking.

## 2026-08-12 (late) — the listening-study kit: built, not run

Five review rounds have asked for a human-vs-CTC agreement number on real
generated audio. The paper has always said, honestly, that no listening study
was run — this commit makes that a weaker position than it needs to be by
reducing the remaining work to an hour of listening, not by producing a
result.

`scripts/make_listening_sheet.py` draws 60 clips from the released 165 (30
repeated against 30 control, exactly 10 per k-band per arm — balanced on
purpose, since an audit weighted toward controls cannot detect an
arm-asymmetric judge error, and that is the entire objection). Blinded: the
listener's sheet (`data/listening/listening_sheet.csv`, 60 rows, confirmed on
disk) carries only a row id, the clip and two empty columns; model, arm, CTC
count and transcript live in a separate `data/listening/listening_key.csv`
that only the scorer reads. `scripts/score_listening.py` reports exact
agreement, mean absolute difference and Cohen's kappa with bootstrap CIs,
split by arm and k-band, plus the *signed* difference by arm — the number that
actually separates "our judge understates the deficit" from "our judge
manufactures it" — with its interpretation pre-committed in the docstring
before any real data exists. It refuses to run on a half-finished sheet.

**As of this reconciliation: no result file exists anywhere under
`data/results/` with "listen" in the name** — confirmed by directory search.
Nobody has done the hour of listening yet. **Do not report a human-agreement
number; none exists.** This is available work for a future session, not a
result to cite.

## 2026-08-13 — reconciliation note: a live causal re-run in flight, do not quote it

While reconciling this file, `research-state.yaml` was found to still describe
the causal-intervention second-checkpoint extension as an unstarted item; it
has since progressed and needs a caution rather than a result.

`data/results/causal_count_second_checkpoint.json` (committed at `HEAD`)
extends the rank-1-patch causal null to Qwen-0.6B, Qwen-1.7B and a re-run of
Llasa-1B — but its first pass used protocol `"decode-time, paired against the
free baseline"`, **not** the resume-reference pairing the original,
*published* Llasa-1B result used (same prefix, no hook, same seed — see
`implementation-notes.md`'s "trap that would have manufactured a causal
result" for why this pairing exists at all: a patched run and a free baseline
consume the sampler's random stream at different offsets and are not
comparable even at the same seed).

The file itself contains the comparison, in a `llasa1b_three_way` block:

    published (resume-reference, original):    median  0.00  [-0.09, +0.09]   0.0% degenerate
    this run, decode-time protocol, same cell:  median +0.065 [-0.258,+0.251]  3.1% degenerate
    this run's own verdict field:               "too disruptive to interpret
                                                   (disruption index 1.02 >= 0.5, ...)"
    this run's resume-reference field:           null -- not yet landed

(Verified directly against the JSON: `patched_degenerate_pct` 3.0864,
central-cell `crossk|L7|P128` median 0.0645, CI [−0.2581, +0.2513].) The
**resume-reference re-run is running now** — as of this reconciliation,
`data/results/causal_rs_llasa1b_*`, `causal_rs_llasa8b_*` and `causal_rq_*`
files exist uncommitted, some with timestamps within minutes of this being
written, alongside modified `analysis/causal_count_resume_second.py` and
`analysis/causal_count_second_score.py`.

**No causal number from that file, or from `causal_rq_*` / `causal_rs_*` /
`causal_pq_*`, may enter the paper until the comparator is clean — including
the two Qwen "readable null" results in the same file**, which used the
identical flawed decode-time protocol (confirmed by inspecting their
`rank1_patch.protocol` fields directly). Do not touch the GPUs this is running
on.

## 2026-08-13 — reconciliation note: the Spanish cross-lingual arm concluded while this file was silent about it

`research-state.yaml` had carried the Spanish arm (`xtts2es`,
`analysis/crosslingual_es.py`) as "in progress" since the evening it was
launched; this file never got an entry for it at all. It has since concluded,
and `implementation-notes.md`'s "Cross-lingual arm" section already documents
the mechanism in full — this is the missing pointer plus the verified verdict.

`data/results/crosslingual_es.json` (tracked, matches `HEAD`): Gate 1
(judge audit) and Gate 2 (vocabulary audit) both **pass**. Gate 3
(informative — control-arm exact-rate must clear 0.5 for the exact-rate
statistic to mean anything) **fails**: `control_exact` = 0.204. The Spanish
XLSR-53 judge's own WER (~8.8%) is too high for an exact-match statistic to
survive even on control items, since exact-match needs all `k`
per-occurrence hits and `(0.80)^k` collapses fast. **Verdict: the judge is
sound, the exact-rate statistic is not transportable to this judge, and the
counting arm is not reportable** — this is `paper/supplementary/supp.tex`'s
own Gate-3 section title. Do not quote a Spanish exact-rate gap; none is
licensed by this data.
