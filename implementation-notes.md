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

Renormalised over the occurrence columns, the lemma is confirmed: block entropy
`= 0.97 log k` and the within-block share falls with a log-log slope of `-1.00`,
both as predicted.

**Two deltas, and only one of them is the lemma's.** The measurement emits two
quantities that were once quoted under a single symbol, which is what made the
draft read as inconsistent:

| quantity | what it bounds | value (deep layers, `k>=6`) |
|---|---|---|
| entropy gap `max(log k - H)` | the **average** logit spread over the block | `<= 0.45` nats |
| most-attended share `max(share x k)` | the **extreme** — the single worst occurrence | `3.4/k` worst case, `delta <= 1.23` |
| median share `x k` | the same extreme, typical rather than worst | `1.7/k`, `delta = 0.51` |

Lemma B has to survive the worst occurrence, not the mean one, so it is entitled
only to the extreme: quote `delta <= 1.23` / `3.4/k`. Quoting 0.45 as the bound
understates it ~2.7x. `analysis/make_numbers.py` emits them as separate macros
(`DeltaEntropy`, `DeltaMax`, `ShareMaxK`, `ShareMedK`, `DeltaMedian`) for exactly
this reason; do not collapse them again. Superseded values from before the
renormalisation fix: `0.98 log k`, `delta <= 0.06`, `1.9/k`.

Note `attn_share` (the *mean* within-block share) is `1/k` by construction and
carries no information; the informative quantities are `attn_share_max`,
`attn_unif_dev` and `attn_block_entropy`.

Caveat worth knowing before you requote these: this block in `make_numbers.py`
takes `state.csv` as-is and does **not** drop `ABLATIONS`, so `xtts2norp` rows
are inside it. Recomputed without them the numbers are unchanged to the reported
precision (`ShareMedK` moves 1.668 -> 1.678), so nothing rests on it — but the
filter is missing and every other panel statistic applies it.

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

## Population and exclusions (added 2026-08-12) — read before touching any analysis

`src/common/population.py` is the **single** definition of which rows the paper
may report. Call `panel(df)`; do not re-implement the filters. They had already
drifted: `count_error.py` listed one ablation where `capacity.py` listed four, so
re-scoring the full model list would have folded three repetition-penalty arms
into the panel with nothing failing.

Four rules, each with a reason you should not undo without one of your own:

1. Ablation arms (`xtts2norp`, `xtts2rp*`) are one checkpoint under altered
   decoding. Pooling counts XTTS-v2 five times.
2. Degenerate/empty audio has no count; reported as its own rate.
3. Templates whose vocabulary the judge cannot transcribe. The criterion is
   *near-total absence* (`analysis/judge_vocab_audit.py`), which no decoder
   behaviour can produce — six checkpoints do not fail on one filler in 106 of
   106 items while rendering its neighbours. Only t2 fails (`okay`, `hmm`).
   Applied to **both** families so it cannot favour the control side.
4. `hit_cap` items — generation stopped on *our* token budget, not the model's
   stop decision. Median relative error -0.44 against -0.08 for the rest. Scoring
   them as model failures was inflating the deficit.

### Generation budgets

The main sweep runs `--max-new-tokens 2048`. Llasa hits that on 12–21% of items
**including at k=1**, so a cap hit means runaway generation, not long text. The
extension ladder raises it to 8192 because a k=128 item is genuinely ~55 s of
speech at X-codec2's 50 Hz; scoring a budget truncation as a counting failure is
exactly rule 4's artifact.

**XTTS-v2 cannot run the extension at all.** `model.gpt.max_gen_mel_tokens` is
~602 (~26 s) and `xtts_gen.py` only ever *shrinks* it (`min(...)`). Raising it
would run the model outside its training range. Excluded with that reason stated;
it is not a result about XTTS-v2.

### Scoring

`count_units` skips a unit it cannot find rather than ending the item. For
repeated items this is provably a no-op — all units are the same word, so a miss
from position `i` means every later one misses too — and it is verified
bit-identical on them. For controls it stops one mis-transcribed filler from
voiding credit for every filler after it.

### Judge

CTC only. Beyond the known Whisper de-duplication bias, `analysis/ctc_field_validation.py`
found Whisper emitting 10.7–12.3 words/sec on real generated audio at k>=24 —
not physically speech. CTC's own limit is blank-collapse merging adjacent
identical words: it depresses repeated counts and not control counts, so it
inflates our reported gap rather than creating it. Treat magnitudes at k>=16 as
approximate; no claim in the paper rests on one.

### Paper page budget

ICASSP allows 4 content pages + 1 of references. The body **must** end on page 4.
`paper/build.sh` reports the count; to find the overflow:

```python
import pypdf
r = pypdf.PdfReader('paper/build/main.pdf')
print(r.pages[4].extract_text().find('REFERENCES'))  # chars of body on page 5
```

Anything above 0 means the body spills. Reference text totals ~3.9 k chars and a
full page holds ~5.1 k, so references fit on page 5 once the body clears it.

## GPU allocation (added 2026-08-12)

**This project uses GPUs 2 and 3 only.** Cards 0 and 1 on this host belong to
someone else and must not be touched.

This is enforced, not merely documented, because a default buried in a dozen
argparse calls is the kind of constraint nobody notices until it is violated:

* `src/common/gpus.py` holds `ALLOWED = (2, 3)` and `DEFAULT_GPU`. Change the
  allocation there and nowhere else.
* Every generation and transcription entry point calls `check_gpu(args.gpu)`
  immediately after parsing and exits with an explanatory message on 0 or 1.
* All `--gpu` defaults point at an allowed card, so a forgotten flag is safe.
* `scripts/*.sh` pass 2 and 3 explicitly.

The escape hatch is deliberate and loud: `ICASSP_ALLOW_ANY_GPU=1`. Use it only
if the allocation has actually changed, and update `ALLOWED` at the same time.

Note that jobs already running when the restriction arrived were left to finish
rather than killed mid-generation; nothing has been launched on 0 or 1 since.

## Traps in the final day's scripts

Six analyses were added on the last day, each answering one reviewer objection
with data. Five have a trap worth knowing before you rerun them.

**`horizon_forms.py` — populations, not maths.** Its first run returned 4.13 for
the same soft-horizon fit `horizon_ext.py` reports as 3.48. Neither was wrong
arithmetically: the new script's `--ext` default listed one extension CSV where
the old one used two, so they were fitting different rows. *If a new script
disagrees with an old one on the same estimator, suspect the population first.*
The soft-horizon row now reproduces `horizon_ext.py` exactly, and that agreement
is the check.

**`probe_past_horizon.py` — the threshold decides borderline cases.**
"Beats the constant predictor" is a hard boundary, and Llasa-3B sits 0.006 MAE on
the favourable side of it over twelve items. Both readings of that number were
available and one flattered us. The script now calls anything within `TIE = 0.02`
of the constant predictor indistinguishable and prints both counts. If you widen
the panel, check `PhKeptRatios` still reads sensibly — it lists every kept
checkpoint, including ones that merely tie.

**`greedy_decoding.py` — cap hits are the confound.** Greedy on repetitive text
reaches the token budget far more often than sampling, and truncated counts are
censored downward: exactly the direction that manufactures a greedy deficit. The
script applies the panel's cap-hit rule and *reports the rate per arm* (0.0% on
both Qwen arms) rather than asserting the check was done. Also: run it over
`data/stimuli/stimuli_greedy.jsonl`, not the full file — the first launch spent
twenty minutes on families the comparison never uses.

**`vits_config.py` — read the sign, not the size.** Both duration-predictor
perturbations hurt the *control* arm more than the repeated one, because a
control made of distinct words loses more to fast or noisy speech. That inflates
the magnitude of VITS's negative gap and says nothing about repetition.

**`sample_sizes.py` — asserts its own arithmetic.** `457 + 525 = 982` is checked,
not left to the reader, so a change to the exclusion rules that breaks the
reconciliation fails loudly instead of producing a table that no longer adds up.

**`scripts/check_supp_tables.py`** closes the gap `check_numbers.py` leaves. The
main paper is all macros and cannot hold a stale number; the supplement's tables
are hand-typed and could. It fails on any result-JSON number missing from
`supp.tex` — including numbers the supplement legitimately does not quote, which
is a decision worth making on purpose rather than by omission.

## The judge replication, and one inconsistency inside it

`analysis/independent_judge.py` answers the objection five review rounds kept
returning to. Three things to know before rerunning it.

* **`verdict.A` is the raw asymmetry; the printed `confound_signature` flag uses
  the trimmed one.** They agree for HuBERT (both clear the -1.0 threshold by a
  wide margin) so nothing downstream is wrong, but the two are not the same
  quantity and a future judge could land between them. The paper quotes the raw
  HuBERT value and S14 reports both. Fix the script before adding a judge.
* **Whisper's raw asymmetry of -53.7 is not a confound signature.** Its median
  signed difference is exactly zero; 152 runaway-loop items -- one judge reading
  441 repetitions where the other reads 5 -- set the mean. Both judges score
  those wrong, so no exact-rate gap depends on them. Trimmed it is -0.27. This
  was a post-hoc addition and the script's docstring says so.
* **`panel()` drops "degenerate" partly on an empty transcript, which is
  judge-dependent.** Comparing two judges on their own surviving rows therefore
  confounds scorer with population. Every paired statistic runs on the
  intersection; own-panel and paired-panel differ by 0.1 point and both are
  reported. Do not compare judges on unequal populations.

Superseded by this run: `noise_floor.py`'s "our judge reports the higher count
in 11 of 13 disagreements" was n=60. At full n it is 65.9% over all k and 72.2%
at k>=6 (61.2% and 67.1% on the repeated arm alone). Still net in our disfavour,
i.e. our judge still under-states the deficit, but the 85% figure was optimistic
and must not be quoted.

## The stop-head analysis, and the traps it found

`analysis/stop_head.py` asked what the EOS logit reads -- decoded count, or
elapsed duration. It came back negative and is documented rather than promoted.
The traps it surfaced matter beyond that one analysis.

* **`hidden` and `entropy`/`top1` in the instrumented npz are offset by one
  step.** `hidden`/`attn_*` are sliced `[plen:]` (the state left *after* token
  t); `entropy`/`top1` come from `logits[plen-1:-1]` (the distribution that
  *produced* token t). Anything joining the two must shift one. This fails
  quietly: reconstructing the EOS logit gives 0.075 nats of error shifted and
  3.55 unshifted, which reads as a bad probe rather than a bad index. The
  docstring in `src/models/llasa_gen.py` asserted the wrong alignment and is
  fixed.
* **The last probe layer is already post-final-RMSNorm**, so exact logits are
  one matmul with `lm_head.weight` -- no forward pass, no model load. Llasa-1B
  ties embeddings; Llasa-8B does not.
* **Oracle contamination is easy here.** `t/T_item` and `k*t/T_item` score
  R^2 0.27-0.55 and mean nothing: they equal 1.0 at the stop step by
  construction, i.e. they encode the event being explained.
* **Never regress the stop time on log k for word_rep.** k=2->8 adds six words
  to an eleven-word carrier, so the slope is mechanically attenuated at low k
  for reasons unrelated to counting. Regress on expected words.
* **True count-so-far is not measurable in this decoder.** Boundary
  localisation needs the monotone text read-head that killed the first q
  estimator. Only the probe's decoded count is available.
* **The design confounds requested count with text length inside the repeated
  arm** -- they are the same variable there. So no representational analysis on
  repeated items alone can distinguish "encodes the count" from "encodes how
  much text there is"; only the matched control separates them, and it does so
  behaviourally. This is now stated in the paper.
* I/O, not GPU, is the cost: ~10 GB per family off a shared spinning disk,
  2-8 MB/s under sibling load, ~75 min for the first build. A cache lives at
  `/home/kirill/mnt/hdd_6tb_1/icassp_tts/stop_head_cache/` and reruns take ~1 min.

## CosyVoice 2: the alignment-supervised arm, and its four traps (added 2026-08-12)

`FunAudioLLM/CosyVoice2-0.5B` was added because reviewers were right that the
panel contained no AR system whose text/speech alignment is *supervised*. It is
run through the standard chain (`src/models/cosyvoice_gen.py` -> CTC judge ->
`score_counts.py`), 180 items x 3 seeds, results in
`data/results/behavioural_cosyvoice.csv`. Env `cosyvoice` (py3.10, torch
2.13.0+cu130); the CosyVoice source tree is a checkout at
`/home/kirill/mnt/hdd_6tb_1/icassp_tts/third_party/CosyVoice`, not a pip package.

**The sentence splitter is the experiment-killer.** `inference_zero_shot()` --
the entry point in every example -- runs `text_normalize(text, split=True)`,
which sends English through `split_paragraph(..., token_max_n=80)`. A
`sentence_rep` item at k=16 would have been chopped into several utterances and
synthesised independently, i.e. the harness would have removed the periodic
conditioning before the decoder ever saw it, and the model would have "counted"
perfectly for reasons that have nothing to do with counting. The generator
therefore calls `frontend_zero_shot` -> `llm.inference` -> `token2wav` directly
with the whole stimulus, exactly as `xtts_gen.py` avoids XTTS's own splitter.
Any future model that ships a chunker needs the same treatment: check for one
*before* trusting a good result.

**The generation ceiling is proportional, not absolute.** Unlike XTTS-v2's fixed
~602 mel tokens, `Qwen2LM.inference` sets `max_len = 20 * n_text_tokens` and
`min_len = 2 * n_text_tokens` (EOS masked below the floor). Because the cap grows
with the text, it never binds on this ladder: 0.0% cap hits on both arms, worst
item at 98% of its own ceiling. Both bounds are recorded per item (`max_len`,
`min_len`, `hit_cap`, `hit_floor`) so this stays a measured fact rather than an
assumption.

**It hardcodes cuda:0 and takes no device argument.** `CosyVoice2Model.__init__`
and `CosyVoiceFrontEnd.__init__` both do
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. `--gpu` is
therefore implemented as `CUDA_VISIBLE_DEVICES`, set before torch is imported;
`check_gpu` still runs first. Nothing else would have kept it off card 0.

**`torchaudio.load` no longer works.** 2.11+ dispatches to TorchCodec, which does
not load against this host's FFmpeg -- the same trap that forced the Whisper
judge off `pipeline` (§7). `cosyvoice_gen.py` patches `load_wav` in both
`cosyvoice.utils.file_utils` and `cosyvoice.cli.frontend` to a soundfile
implementation with identical semantics.

Two smaller notes. The decode loop is copied verbatim from
`Qwen2LM.inference_wrapper` rather than called, because that loop already asks
for `output_hidden_states=True` and throws all but the last layer away -- the
full trajectory is free, and no teacher-forced second pass is needed. And
CosyVoice 2 samples with `ras_sampling` (a repetition-aware rule that bans a
token repeated within the last 10 steps and resamples), which is a decoding-time
intervention in the same family as XTTS-v2's `repetition_penalty=5.0`; it acts on
25 Hz acoustic tokens rather than words, and it did not prevent the deficit, but
a penalty-style ablation on this model is the obvious next arm.

**Do not let it into `behavioural.csv`.** `scripts/run_pipeline.sh` scores every
model that has transcripts into the shared table, and `population.py` does not
list `cosyvoice2` as an ablation, so a plain rerun of the pipeline would fold this
checkpoint into the paper's panel and move every macro. Its results live in their
own CSV on purpose; decide deliberately whether it joins the panel.

## Widening the probe panel (added 2026-08-12) — two code changes and why

Filling in the missing arms (Qwen 0.6B/1.7B both arms, Llasa-3B/8B controls)
needed two changes beyond running the existing scripts.

**`probe_past_horizon.py` takes a list of control arms, not one.** It was written
when exactly one checkpoint had control activations, so `--past-control` was a
single `model=path` string and `range_confound_ruled_out` was one boolean read
off it. That made the paper's range rebuttal a statement about Llasa-1B wearing
a panel's clothes. It is now per checkpoint: a control arm excuses the range on
*its own* checkpoint and nowhere else, and the script reports which lost
checkpoints have one (`range_confound_checked_on`) and which do not
(`range_confound_unchecked`) rather than collapsing that into a yes. Re-running
it on the old three-checkpoint inputs reproduces the previous output exactly,
which is the check that the generalisation did not move anything.

**`llasa_gen.py` computes the instrumentation entropy in chunks.** The old line
did `softmax` over the whole `[T, V]` logit block in float32 — two ~2 GB tensors
alive at once at T=2700 over a 193k vocabulary — for two scalars per position
that the count probe never reads. That peak, not the model, is what OOMed
Llasa-3B's control run when a co-tenant job grew on the same card. Softmax is
row-wise, so chunking is numerically identical. *If you copy this pattern:* the
first version left `pr` scoped inside the loop while the cleanup line below still
said `del ... pr`, which crashed the item **after** its `.npz` was written and
before its meta row was — so the retry regenerated a file that already existed
and the meta was self-consistent by luck, not design.

**Do not run two generation drivers over the same stimulus file.** A driver whose
wrapper PID was killed but whose real bash survived resumed its loop and started
a second `qwen17b` process on the other card, generating the same item ids
concurrently. Nothing was corrupted (checked: no duplicate meta rows, every
`.npz` reloads), but two processes writing one `.npz` is a real way to lose data.
`kill $!` on a `nohup bash script.sh &` may be killing a wrapper; verify with
`pgrep -af` afterwards rather than trusting `ps -p`.

**Cost note.** The HF cache is on the spinning disk, and three model loads at
once saturate it at ~75 MB/s: Llasa-8B took ~10 minutes to reach the GPU and
19 minutes to do the actual 12 items. Stagger loads rather than launching a fleet.
