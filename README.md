# Counting Collapse in Autoregressive TTS

ICASSP 2026 submission. Autoregressive TTS models loop, truncate, and lose count
on text that repeats a phrase many times. We give a theorem saying they *must*,
machine-check it in Lean 4, and measure its premise and its consequence in six
real checkpoints.

**Start here if you are a new agent:** read `findings.md` (what we know),
then `research-state.yaml` (where we are), then this file (how to run things).

---

## The claim in one paragraph

Under text that repeats a phrase `k` times, softmax attention cannot tell one
copy from another (each gets weight `Θ(1/k)`; Lemma B), so the decoder receives
approximately the same conditioning at every repetition boundary and the map
carrying it from one boundary to the next is approximately autonomous. If that
map contracts, boundary states converge to a fixed point geometrically, and past
a horizon `N* ≤ log(margin/2LC)/log q` **no Lipschitz readout — including the
model's own stop head — can tell how many repetitions have been produced**
(Theorem A). Looping or truncation is then the only behaviour available.

The prediction that distinguishes this from "long text is hard" is that the
failure tracks *periodicity*, not length. It does.

---

## Repository map

```
lean/SpectralTTS/          the formal artifact (mathlib; zero sorry)
  SpectralTTS/CountingCollapse.lean    Theorem A (i)-(iv)
  SpectralTTS/AttentionDilution.lean   Lemma B
data/stimuli/make_stimuli.py           the repetition-ladder benchmark
src/common/                            measurement library (see below)
src/models/{llasa,xtts,qwen}_gen.py    per-family generation + instrumentation
src/models/xcodec2_decode.py           offline vocoding for Llasa token ids
analysis/                              the analyses that produce the paper
paper/                                 main.tex + generated numbers/tables/figs
scripts/                               setup, pipeline driver, verification gates
literature/                            survey.md, gaps.md (40 verified refs)
```

Heavy artifacts live off-repo at `/home/kirill/mnt/hdd_6tb_1/icassp_tts/`
(`audio/`, `tokens/`, `activations/`, `asr/`, `hf_cache/`), symlinked as
`data/generated` and gitignored.

---

## Reproducing

### 0. Environments (once)

```bash
bash env/setup_envs.sh all      # base, qwen, coqui, xcodec2
bash scripts/setup_lean.sh      # elan + mathlib cache (CPU, ~15 min)
bash scripts/download_models.sh # 8 checkpoints, ~74 GB
```

Four conda envs are required because the dependency constraints are mutually
unsatisfiable; `env/setup_envs.sh` documents each pin and why it exists.

### 1. Verify the formal artifact

```bash
bash scripts/check_lean.sh
```

Builds the project, scans for `sorry`, and prints `#print axioms` for all eleven
exported theorems. Passing means every theorem depends only on `propext`,
`Classical.choice`, `Quot.sound`.

### 2. Generate

One model per GPU. All generators are resumable — they skip `(item_id, seed)`
pairs already in their metadata file, so re-running after a crash is safe.

```bash
python src/models/llasa_gen.py --model llasa1b --gpu 0 --seeds 0 1 2
# xtts2 needs the `coqui` env and COQUI_TOS_AGREED=1
# qwen06b/qwen17b need the `qwen` env
```

### 3. Vocode, transcribe, score

```bash
bash scripts/run_pipeline.sh <asr_gpu> llasa1b xtts2 ...
```

Llasa emits token ids (vocoded separately in the `xcodec2` env); every other
family writes wavs directly. Each stage is idempotent, so this can be run
repeatedly while generation is still filling in.

### 4. Analyse and build the paper

```bash
python analysis/state_dynamics.py --models <models> --out data/results/state.csv
python analysis/capacity.py                       # the central measurement
python analysis/horizon_fit.py                    # behavioural collapse points
python analysis/figures.py
bash paper/build.sh                               # regenerates numbers, compiles
```

`paper/build.sh` regenerates `numbers.tex` from the CSVs before every compile, so
a number in the PDF cannot drift from the data it came from. **Never hand-edit a
number in the `.tex` files** — add a macro in `analysis/make_numbers.py`.

---

## The measurement library (`src/common/`)

| module | what it is for |
|---|---|
| `registry.py` | the model panel; single source of truth for keys, paths, probe layers |
| `offset_tok.py` | one tokenizer interface with character offsets across all families (note: **not** named `tokenizers.py` — that shadows the real package) |
| `dispersion.py` | **the primary state measurement**; boundary-free |
| `boundaries.py` | superseded boundary-based estimator, kept for the negative result |
| `asr_transcribe.py` | Whisper large-v3 with word timestamps + audio degeneracy flags |
| `score_counts.py` | repetition counting and outcome classification |

---

## Things that will bite you

- **`src/common/tokenizers.py` must not exist.** Running a script from inside
  `src/common/` puts that directory on `sys.path[0]`, so such a file shadows the
  real `tokenizers` package and breaks `transformers` with a confusing
  `ImportError`. The module is called `offset_tok.py` for this reason.
- **`xcodec2==0.1.5` pins torch 2.5** and leaves `transformers`/`torchao`
  unpinned, both of which have since moved past it. The working combination is
  `transformers==4.46.3`, `torchao==0.6.1`; anything newer fails at import.
- **The HF `automatic-speech-recognition` pipeline routes audio through
  torchcodec**, which does not load against the installed FFmpeg. Drive
  `WhisperProcessor` + `WhisperForConditionalGeneration` directly instead.
- **`coqui-tts` breaks on `transformers` 5.x** (`isin_mps_friendly` was removed);
  the `coqui` env stays on 4.57.x.
- **XTTS replaces spaces with `[SPACE]` tokens** and lowercases before
  tokenizing, so text-column indices must be computed on the transformed string
  (`offset_tok._xtts_prepare`), not the raw one.
- **Llasa with eager attention is ~5× slower to sample.** `llasa_gen.py` switches
  to SDPA for generation and back to eager only for the instrumented pass.
- **Whisper de-duplicates repeated speech**, so an ASR transcript count alone
  understates loops. Correctness is decided conjunctively (exact transcript count
  *and* consistent duration); see `score_counts.classify`.

---

## Discipline

Protocol commits precede result commits — the git history is the
pre-registration. Negative results are reported, not dropped: the paper contains
a subsection on the boundary estimator that did not work and why.
