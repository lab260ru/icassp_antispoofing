# Pending notifications

## 2026-08-09 — bootstrap

Research bootstrap is complete: the workspace, reproducible protocols, HDD
registry, and four-GPU plan are ready. Public Hugging Face access is being
tested next. Telegram delivery is queued because `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` are not currently present in the runtime environment.

## 2026-08-09 — first validated artifact

The Arena catalogue is pinned and public downloads work. ASVspoof 2019 LA is
downloaded at its Arena revision; all 71,237 published Spectra-AASIST scores
joined exactly to labels. The ongoing work is the audio-feature pilot and the
remaining model/dataset downloads. GitHub pushing remains queued until a valid
runtime token is available.

## 2026-08-09 — complete first score panel

All 72 pinned Arena score/result artifacts for the five core datasets are now
downloaded with byte-size validation. The first full analysis panel covers
eight detectors and 71,237 ASVspoof2019 LA trials per detector (569,896 score
rows); the CPU feature extraction is the remaining prerequisite for the first
association result. Delivery remains queued because the Telegram credentials
are not present in the runtime environment.

## 2026-08-09 — H1 discovery data ready

The locked ASVspoof2019 LA discovery subset is fully extracted: 10,000 balanced
utterances and three reproducible views (30,000 rows). Before creating any
association table, the analysis caught and stopped on a self-control edge case
for the loudness feature. The correction is under test; no provisional result
was kept from that failed invocation.

## 2026-08-09 — first H1 discovery result

The corrected ASVspoof2019 LA screen completed 1,344 within-class association
tests across eight published detectors, 28 signal descriptors, and three
waveform views. Crest factor is strongly negatively associated with spoof
evidence in Spectra-AASIST (and some other architectures), but it is not
directionally uniform across the panel. This is a promising heterogeneous-
sensitivity result, not yet a cross-dataset or causal shortcut finding. The
two remaining discovery corpora are now the priority before any candidate is
frozen.

## 2026-08-09 — initial paper checkpoint

The initial ICASSP draft is now compiled as a four-page IEEE-format PDF with
verified citations, the full first-discovery heatmap, and explicit limits on
what the current result can support. The H2 runner/quality-gate audit is also
preserved. The mandated three-reviewer launch was attempted but the local Claude
CLI is not authenticated, so all three subprocesses failed before reviewing;
this is recorded transparently and can be rerun after `claude /login`.

The worktree has committed checkpoints, but remote push and this Telegram
notification are still queued until valid runtime credentials are supplied.

## 2026-08-09 — second H1 discovery corpus

The unchanged H1 screen is complete on ASVspoof2021 LA. Spectra-AASIST's
spoof-class crest association remains negative but is much weaker than in
ASVspoof2019 LA (-0.099 vs. -0.454), and the multi-model pattern is
heterogeneous. I also corrected an output-path bug before it could obscure
provenance: each corpus now has immutable scoped tables and the aggregate tool
refuses to call the five-dataset portability criterion early.

## 2026-08-09 — H2 scorer parity gate

Two executable GPU scorers are now validated against their published Arena
ordering on a frozen 128-clip calibration set: AASIST uses raw waveform input
(rho 0.999994) and Spectra-AASIST uses external pre-emphasis 0.97 (rho
0.999971). This is only a baseline-parity gate, not an intervention result. H2
still waits for the H1 candidate freeze, ASR WER quality gate, and a complete
four-model scorer panel.

## 2026-08-09 — validated H2 inference settings

On the frozen parity set, the CUDA batch sweep selected initial model-specific
settings: Spectra-AASIST batch 8 (236.32 clips/s) and AASIST batch 2 (333.44
clips/s). These are stored as H2 inference engineering measurements; they do
not replace the separate BF16 batch/worker sweep required if H3 training starts.

## 2026-08-09 — literature and paper context checkpoint

The paper now incorporates source-verified ASVspoof 2021 and cross-domain
generalization context, with a compiled PDF and a paper-local citation ledger.
The framing remains intentionally conservative: cross-corpus differences
motivate this audit but do not validate any cue as causal. The ASVspoof2021-DF
audio download is continuing; remote push and Telegram delivery remain queued
because no runtime credentials are available.

## 2026-08-09 — third discovery corpus and H2 candidate freeze

ASVspoof2021-DF is complete despite a recoverable Hub decoder error on its
final protocol file. The locked 10,000-sample H1 screen adds 1,344 tests and
shows Spectra-AASIST's crest-factor association remains negative but attenuates
to -0.065, whereas AASIST reverses direction. The explicit five-dataset
portable rule remains unevaluable without held-out data. A disclosure-marked
exploratory freeze admits only the Spectra-AASIST crest-factor family for H2;
it is not a causal or portability conclusion. Telegram and GitHub push remain
queued pending runtime credentials.

## 2026-08-09 — H2 quality-gate runtime

The H2 ASR/WER component is now reproducibly pinned: OpenAI Whisper `small.en`
with a verified checkpoint hash, deterministic normalized WER, and FP16 GPU
mapping documented. This is only infrastructure; no audio has been transcribed
and no quality gate or causal conclusion has passed. Held-out audio processing
continues, while the multi-model transformed-audio panel is still incomplete.
