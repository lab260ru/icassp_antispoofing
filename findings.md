# Findings — Counting Collapse in Autoregressive TTS

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

## Status

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

### The XTTS ablation

Disabling XTTS-v2's shipped `repetition_penalty=5.0` makes the model *worse*
(k* 4 → 2) while the repeated-vs-control gap persists (0.02 vs 0.15). The penalty
was masking the collapse, not causing it, so the penalised model is the
conservative member of the panel. Its capacity contrast loses significance
because disabling the penalty compresses the dynamic range of both conditions —
reported, not glossed.

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
