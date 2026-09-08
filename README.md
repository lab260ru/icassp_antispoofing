# Repetition, Not Length: Isolating the Counting Failure in Neural TTS

Code and data for the paper *"Repetition, Not Length: Isolating the Counting
Failure in Neural Text-to-Speech"* (Borodin, Kudryavtsev, Mkrtchian;
submitted to ICASSP 2027). This branch is the released artifact the paper
cites: the generation code, the test set, the scored results and the analysis
code. The paper sources live on the `arxiv` branch.

## The claim in one paragraph

Ask a TTS model to repeat a word a dozen times and it stops early, runs on,
or locks into a loop. Such reports confound two variables: repeating a phrase
`k` times makes the text both longer and repetitive. Our test set breaks the
confound. Every repeated item is paired with a length-matched control of the
same carrier and word count in which the `k` copies become non-adjacent
distinct fillers. If long text were the problem, the two curves would
coincide. They do not: six checkpoints from three architectures render the
controls at 94.3% exactly right at `k>=6` while the repeated twins fall to
18.2%, a 76.7-point gap per checkpoint (95% interval [68.4, 84.4]).

The gap survives greedy decoding, repetition-penalty sweeps and
repetition-aware sampling, four independent speech recognisers, and 420
analysis specifications without once reversing sign (range 34.4 to 86.4
points). A held-out fourth architecture, CosyVoice 2, lands within a point of
its pre-registered prediction (64.4 observed, 65.3 predicted). One of two
non-autoregressive baselines fails the same way: F5-TTS, which predicts one
total duration, shows a 60-point gap; VITS, which predicts a duration per
token, shows none. The deficit is graded in the period of the text: half of
it survives when no word is adjacent to itself.

What breaks is not storage of the count. The requested count stays decodable
from hidden states while the rendering fails, and repeated text grows
distinguishable states at 0.46 of the control's rate: a slowing, not a halt.
The contraction account of decoder hallucination fails its own measurement
here (the contraction factor `q` never drops below 1). Why the readout fails
is the open question the paper ends on.

## Repository map

```
data/stimuli/            the test set: deterministic generators + items (jsonl)
data/results/            scored behavioural results (CSV/JSON), one file per run
data/audio_sample/       165 clips + manifest, to check the judge by ear
src/common/              measurement library (judge, scoring, populations)
src/models/              per-family generation: llasa, qwen, xtts, vits, f5,
                         cosyvoice; xcodec2_decode.py vocodes Llasa token ids
analysis/                every analysis in the paper, one script per claim
scripts/                 pipeline drivers (run_*.sh, score_*.sh), model download
env/setup_envs.sh        the four conda environments, with every pin explained
```

Heavy artifacts (audio, tokens, activations, ASR caches) are not in the repo;
generators write them under `data/generated`, which is gitignored.

## Reproducing

Environments and checkpoints, once:

```bash
bash env/setup_envs.sh all       # base, qwen, coqui, xcodec2
bash scripts/download_models.sh  # 8 checkpoints, ~74 GB
```

Four conda environments are required because the dependency constraints are
mutually unsatisfiable; `env/setup_envs.sh` documents each pin and why.
GPU allocation is enforced in `src/common/gpus.py`; edit it for your machine.

Generate. One model per GPU; generators are resumable and skip
`(item_id, seed)` pairs already in their metadata:

```bash
python src/models/llasa_gen.py --model llasa1b --gpu 0 --seeds 0 1 2
# xtts2 needs the coqui env and COQUI_TOS_AGREED=1
# qwen06b / qwen17b need the qwen env
```

Vocode, transcribe and score (idempotent, safe to re-run while generation
fills in):

```bash
bash scripts/run_pipeline.sh <asr_gpu> llasa1b xtts2 ...
```

The period ladder, aperiodic controls, extension ladder, penalty sweeps and
repetition-aware sampling each have their own `scripts/run_*.sh` and
`scripts/score_*.sh` driver pair.

Analyse. `analysis/` holds one script per claim in the paper; the ones behind
the headline numbers:

```bash
python analysis/count_error.py            # the behavioural measurement
python analysis/checkpoint_level.py       # headline estimates, Table 2
python analysis/exclusion_sensitivity.py  # do our own exclusions make the gap?
python analysis/period_ladder.py          # the deficit graded in the period
python analysis/greedy_decoding.py        # survives argmax?
python analysis/independent_judge.py      # survives other recognisers?
python analysis/state_dynamics.py         # state growth, 0.46 of control
python analysis/jacobian_q.py             # the contraction factor; q >= 1
python analysis/make_numbers.py           # regenerates every number macro
```

`make_numbers.py` writes `paper/numbers.tex` for the paper build on the
`arxiv` branch, so a number in the PDF cannot drift from the data it came
from. Never hand-edit a number in a `.tex` file; add a macro instead.

## The judge

The transcript count comes from a CTC model (`wav2vec2-large-960h-lv60-self`,
greedy best-path, no language model), in `src/common/asr_ctc.py`. The natural
choice, Whisper, is unsuitable for exactly the audio under study: its decoder
is autoregressive with a language-model prior, and on concatenated utterances
whose true count is known it recovers a median 0.25 of the count where the
CTC judge recovers 1.00. Whisper large-v3 (`src/common/asr_transcribe.py`)
remains as one of the four independent cross-checking recognisers.
Correctness is decided conjunctively (exact transcript count and consistent
duration); see `score_counts.classify`. Exclusion rules are blind to the
count; `scripts/check_exclusions_blind.py` verifies that.

## The measurement library (src/common/)

| module | what it is for |
|---|---|
| `registry.py` | the model panel; single source of truth for keys, paths, probe layers |
| `asr_ctc.py` | the CTC judge, the paper's primary transcriber |
| `asr_transcribe.py` | Whisper large-v3 with word timestamps + degeneracy flags |
| `score_counts.py` | repetition counting and outcome classification |
| `population.py` | THE definition of which rows may be reported |
| `offset_tok.py` | one tokenizer interface with character offsets across families |
| `dispersion.py` | state measurement behind Sec. 3.5 |
| `boundaries.py` | superseded estimator, kept for the negative result |
| `ras.py` | repetition-aware sampling for the decoding ablations |

## Things that will bite you

- **`src/common/tokenizers.py` must not exist.** Running a script from inside
  `src/common/` puts that directory on `sys.path[0]`, so such a file shadows
  the real `tokenizers` package and breaks `transformers` with a confusing
  `ImportError`. The module is called `offset_tok.py` for this reason.
- **`xcodec2==0.1.5` pins torch 2.5** and leaves `transformers` and `torchao`
  unpinned. The working combination is `transformers==4.46.3`,
  `torchao==0.6.1`; anything newer fails at import.
- **The HF `automatic-speech-recognition` pipeline routes audio through
  torchcodec**, which does not load against the installed FFmpeg. Drive
  `WhisperProcessor` + `WhisperForConditionalGeneration` directly instead.
- **`coqui-tts` breaks on `transformers` 5.x** (`isin_mps_friendly` was
  removed); the `coqui` env stays on 4.57.x.
- **XTTS replaces spaces with `[SPACE]` tokens** and lowercases before
  tokenizing, so text-column indices must be computed on the transformed
  string (`offset_tok._xtts_prepare`), not the raw one.
- **Llasa with eager attention is ~5x slower to sample.** `llasa_gen.py`
  switches to SDPA for generation and back to eager only for the
  instrumented pass.
- **ASR de-duplicates repeated speech**, so a transcript count alone can
  understate loops; that is why correctness also requires consistent
  duration, and why the judge is validated on audio with a known count.

## Discipline

Protocol commits precede result commits; the git history is the
pre-registration. Negative results are reported, not dropped: the paper's
discussion opens with the mechanism we tried and rejected, and the analyses
that weakened our own earlier claims are in the history of this repository.
