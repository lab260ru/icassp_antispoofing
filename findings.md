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

## Lessons and Constraints (added)

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
