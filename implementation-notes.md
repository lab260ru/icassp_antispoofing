# Implementation notes

Decision log for the counting-collapse study. Records *why* things are the way
they are, especially the choices that a reader would otherwise be tempted to
"fix". Newest sections at the bottom.

Read alongside `findings.md` (what we know) and `research-state.yaml` (where we
are). `README.md` has the run instructions.

---

## 1. Why the theorem is about a *map*, not about transformers

The Lean development proves an implication: periodic conditioning plus
contraction implies a finite counting horizon. It says nothing about whether real
decoders contract. This split was deliberate and it is the reason the artifact is
worth having — formalising forced us to name the premise (`Assumption 2`) instead
of smuggling it in, and reviewers specifically credited the paper for keeping the
distinction visible.

Consequence for writing: never say "we prove transformers cannot count". Say
"we prove that a contracting per-repetition map cannot be counted past `N*`, and
we measure the consequence in real decoders".

The horizon is stated with explicit constants (`log(mu/2LC)/log q`) rather than
asymptotically because the formal proof yields them for free, and because the
constants are what make the mitigation argument in the discussion possible
(logarithmic in margin, inversely logarithmic in `q` — so the levers are not
equally worth pulling).

## 2. Stimulus design: the control is the experiment

The repetition ladder alone proves nothing — long text is also hard. Every
word- and sentence-repetition item at `k>=2` therefore has a length-matched
control with the same carrier, the same word count, and `k` *distinct* fillers.
The whole argument rests on that pair, so anything that would score the two
differently is a bug (and one such bug shipped; see §6).

`boundary_units` was added after the first analysis pass: without it, controls
had no occurrence of the target word, fell back to uniform boundary placement,
and were being compared against attention-placed boundaries for repeated items.
Comparing two estimators is not comparing two conditions.

Number-phrase counts (`nm_*`) are derived from the text, never hand-written. The
first version hand-counted them and got it wrong — "sixty" is not a whole-word
occurrence of "six" — which silently made every numbers item score 0.00.

## 3. Behavioural scoring is conjunctive on purpose

Whisper is itself an autoregressive model and de-duplicates repeated speech, so a
transcript count alone understates loops. Duration cannot substitute: natural
speech-rate variation gives it about ±1 repetition of resolution, so it is a
*consistency band*, never a count.

An item is therefore correct only when the transcript shows exactly `k` units
**and** the duration is consistent with `k`. The two signals fail in opposite
directions (ASR de-duplication shortens the transcript while lengthening the
audio; babbling does the reverse), so requiring both is what makes the label
robust. Nothing is excluded — every item gets a label. The estimator-agreement
rate is reported for `k<=4`, where a correct rendering exists to be measured; at
high `k` the two diverge because the output no longer corresponds to any clean
count, which is the phenomenon, not a measurement failure.

## 4. The primary state measurement changed, and the first one is kept

**What we tried first.** Locate each repetition boundary from text attention, take
`d_m = ||h_{m+1} - h_m||`, fit `log d_m = a + m log q`. This is the direct reading
of Theorem A and it is what the method section originally described.

**Why it failed.** Boundary localisation needs a monotone text read-head. Measured:
the deep-layer attention centroid advances on ~51% of generation steps — a random
walk, not a sweep. Boundary estimates collapse onto near-duplicate steps, so `d_m`
measures localisation error. The pooled fit returns `q ≈ 1.00` with `R² ≈ 0`, i.e.
no signal.

The failure is self-defeating by construction: the flatter the attention over
repeated spans — exactly what Lemma B predicts — the worse any attention-based
localiser gets. That makes it a real finding rather than an engineering
disappointment, and it is reported in the paper (§"Two things we could not
establish"). `src/common/boundaries.py` is kept for that reason and because the
estimator remains the natural one for a model that *does* have a monotone read
head.

**What replaced it.** Representational capacity: the effective rank of the
generation trajectory as a function of `k`, fitted separately on repeated items
and controls (`analysis/capacity.py`). Boundary-free, well-powered, and closer to
Theorem A(ii) than the boundary version was — the theorem's conclusion is that
states stop being *mutually distinguishable*, which is a statement about the
spread of the visited set and needs no boundary labels at all.

An intermediate attempt (`src/common/dispersion.py`, window-wise dispersion decay
converted to a per-repetition `q̂`) is also kept. It shows the effect but is noisy
at low `k`, where the `n_windows/k` exponent amplifies fit error.

## 5. Lemma B had to be measured on the right object

First measurement gave attention mass per occurrence falling as `k^-0.21` against
a predicted `k^-1` — a 5× discrepancy a reviewer correctly flagged. The error was
ours: Lemma B bounds the softmax **restricted to the `k` repeated keys**, but we
measured raw attention weight, which is normalised over the whole sequence, whose
length itself grows with `k`. That confounds dilution with sequence growth.

Renormalised over the occurrence columns, the lemma is confirmed tightly: block
entropy `= 0.98 log k`, implied spread `δ <= 0.06` nats over the whole ladder,
and the most-attended occurrence never exceeds `1.9/k` of the block's mass.

Note `attn_share` (the *mean* within-block share) is `1/k` by construction and
carries no information; the informative quantities are `attn_share_max`,
`attn_unif_dev` and `attn_block_entropy`.

## 6. Bugs that mattered

- **Control scoring.** Controls had `expected_count=0`, so their expected duration
  was computed as a `k=1` item's; every control then looked like a runaway loop
  and scored 0.00 renderable at high `k`. Fixed by scoring both item types with
  one rule (`count_units`) and computing expected duration from `k`.
- **Shared results file.** `run_pipeline.sh` scored only the models passed to it
  and overwrote `behavioural.csv`, silently dropping every other model's rows.
  It now scores every model that has transcripts.
- **Module shadowing.** `src/common/tokenizers.py` shadowed the real `tokenizers`
  package for any script run from that directory, breaking `transformers` with a
  confusing `ImportError`. Renamed to `offset_tok.py`.
- **Number expander.** A looping model had Whisper emit a 700-digit numeral, which
  crashed place-value expansion. Numerals longer than 12 digits now expand
  digit-wise, which is also the semantically correct reading for a loop.
- **Stale abstract.** The abstract kept claiming the boundary-distance result
  after the estimator had been retracted in the results section. Caught by all
  four reviewers; the most damaging single error in the draft.

## 7. Environment pins, and the failure each one prevents

| pin | prevents |
|---|---|
| `xcodec2` in its own env | it hard-pins torch 2.5; nothing else here can live there |
| `transformers==4.46.3` in that env | `Could not import module 'PreTrainedModel'` under 5.x |
| `torchao==0.6.1` in that env | `module 'torch' has no attribute 'int1'` |
| `transformers` 4.57.x in `coqui` | `isin_mps_friendly` was removed in 5.x |
| Whisper via processor+model, not `pipeline` | the pipeline routes audio through torchcodec, which does not load against the installed FFmpeg |
| SDPA for Llasa sampling, eager only for instrumentation | eager sampling is ~5× slower and only the instrumented pass needs attention weights |

## 8. Performance choices

- Activations are saved uncompressed. zlib on a 150 MB fp16 array costs more CPU
  and peak RSS than the disk it saves, and disk is not scarce here. Llasa-8B was
  dying silently under the compressed path.
- Spectral proxies run on an evenly spaced subset of probe layers; each costs an
  SVD, and P4 is a cross-check against the companion ASR study, not the
  load-bearing measurement.
- SVD inputs are subsampled to 512 steps. The proxies are properties of the
  spectrum's shape, which subsampling preserves.
- The count probe is solved in the **dual**: ~50 items against 1–4k features, so
  the primal normal equations are a `d×d` solve (minutes per layer at `d=4096`)
  while the identical dual is `n×n`. Same estimator, same predictions.

## 9. Confounds addressed with data rather than prose

- **Naturalness.** Repeated text is improbable text, so perhaps models fail on it
  for that reason. Measured: scored under a causal LM, the *control* is the less
  probable member in 89% of the 54 matched pairs, and the gap widens with `k`.
  The harder-to-predict text is the one models render correctly, so naturalness
  runs opposite to the effect.
- **XTTS repetition penalty.** XTTS-v2 ships `repetition_penalty=5.0` on acoustic
  tokens, which acts directly against the behaviour under study. Rerun as the
  `xtts2norp` registry entry with the penalty disabled, under a separate key so it
  never mixes with the main run's outputs.
- **Competing hypothesis.** arXiv:2605.09239 reports that in text LMs the repeated-token
  count stays linearly decodable from the residual stream, locating the failure in
  the output policy rather than the representation. Tested directly with
  `analysis/probe_count.py`, using an early-vs-late window contrast within the same
  trajectory so a difference cannot be explained by probe capacity or item count.
  A linear probe is exactly the readout class Theorem A(iii) constrains, so this is
  the theorem's own claim tested rather than an analogy.
