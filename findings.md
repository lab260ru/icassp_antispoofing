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
| P1 geometric decay of boundary distances | not yet measured |
| P2 N*(q̂) predicts behavioural collapse k* | not yet measured |
| P3 attention dilution ~1/k, entropy ~log k | not yet measured |
| P4 spectral proxies shift faithful vs hallucinated | not yet measured |

## Patterns and Insights

*(populated by the outer loop)*

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
