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
| Models undercount repeated text | median rel. count error −8.3% [−12.5,−6.2] at k≥6, n=457 repeated against 525 control, CTC judge (`count_error.json`) | **confirmed**, 4/6 checkpoints with CIs disjoint from their own control (5/6 below their control) |
| Controls are unaffected | median control error **0.0%** at every k up to 32, every model | **confirmed** |
| One model is exempt | Qwen3-TTS-1.7B: 0.0% error, yet still a capacity gap | **second regime**, reported as such |
| Capacity saturates under repetition | gain ratio 0.46 (`capacity.json`); 0.50 raw / 0.50 diversity-adjusted / 0.52 correct-only (`capacity_confound.json`) | **confirmed**, 5/6 disjoint CIs |
| Lemma B (attention dilution) | block entropy = 0.97·log k; **average** spread δ ≤ 0.45 nats (entropy gap); **extreme** δ ≤ 1.23, most-attended share ≤ 3.4/k worst case, 1.7/k typical | **confirmed** — the lemma is entitled only to the extreme; see §"Lemma B's two deltas" |
| Theorem A premise (q<1) | not measurable — the boundary estimator is unsound here | **open**, reported as a negative result |
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
