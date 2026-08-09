# H2 execution-readiness audit

**Audit date:** 2026-08-09  
**Scope:** read-only preflight of the model/dataset artifacts pinned in
`data/arena-index.yaml`. No model was scored and no weights, data, or runtime
files were downloaded for this audit. This document is not an H2 result.

## Decision

H2 has a practical **development** path through **Spectra-AASIST, AASIST, and
Res2TCNGuard** once its H1 candidates are frozen and its quality/score-parity
gates have passed. The requested fourth model, **W2V2-AASIST**, has a pinned
ONNX artifact but it is not on disk; the corresponding pinned Hub revision also
lacks the Python wrapper named in its model card.

Three validated scorers support a pilot or an engineering check, but they do
*not* by themselves satisfy the preregistered confirmatory criterion of a
consistent direction in at least 3 of 4 models. A confirmatory H2 claim needs
a validated W2V2 ONNX scorer or the already-declared `XLSR-SLS` fallback
(`configs/study.yaml`) frozen *before* H2 responses are inspected. A missing
fourth model cannot count as evidence of consistency or as a zero effect.

## Important distinction: baseline scores are not inference artifacts

The pinned `scores.txt` files are trusted **baseline measurement artifacts**.
They contain one scalar per utterance and can establish score orientation, but
they cannot score an H2-transformed waveform. H2 needs an executable model for
the latter.

For every H2 scorer, save all two raw logits, the selected raw scalar, and the
canonical `score_spoof` scalar. Infer the final direction from the matching
pinned `scores.txt` and labels using `src.arena_io.load_model_scores`; do not
infer it from model names or class labels alone. The H2 effect is always
`score_spoof(transformed) - score_spoof(original)`.

Before any intervention arm is scored, run a frozen, stratified calibration
set of 128 unmodified utterances (fixed IDs, no detector-response selection)
through the new runner using the exact model windowing. Join to the corresponding
Arena score file and record rank correlation, orientation, runtime/provider,
and preprocessing. A scorer is eligible only if it has the predeclared correct
orientation and near-identical rank ordering (`Spearman >= 0.999` is the
recommended acceptance gate). Retain calibration and parity failures as H2
artifacts; never reverse a direction after looking at treatment deltas.

## Dataset contract

All three discovery datasets are pinned in `data/arena-index.yaml`:

| Dataset | Revision | Local representation | H2-relevant notes metadata |
|---|---|---|---|
| ASVspoof2019_LA | `9492c4a85ad91508b6da03c92c98c58aeaa02424` | `data/test-*.parquet`; schema `path`, `audio{bytes,path}`, `label`, `notes` | `utterance_id`, `speaker_id`, `subset` |
| ASVspoof2021_LA | `dc119733697c946fcd17fe7c1541d7f26b4bbe07` | same Parquet schema | `utterance_id`, `speaker_id`, `codec`, `transmission`, `attack_id`, `trim`, `phase` |
| ASVspoof2021_DF | `16d4f7d6c68694ac9b0bd43b83df322d1bc5102e` | same declared Arena schema | treat notes fields as provenance; do not rely on an undocumented field |

The downloaded 2019 and 2021 LA shards were directly inspected. Their embedded
audio is FLAC bytes and their label coding is `0=bonafide`, `1=spoof`; both
are also enforced by `src/arena_io.py`. Decode with `soundfile`, use the
`path` stem as the sample ID, and preserve the JSON `notes` in the H2
manifest. The Arena cards declare 16 kHz mono audio. Never infer labels from
file names.

The preselection must be made **before** H2 model responses: stable seed 2609,
500 examples per label/dataset as specified by `configs/study.yaml`, with any
quality-screen exclusions and their reason written to the manifest. Pairing is
by `sample_id`, not row position. The H2 cluster bootstrap unit should use
`speaker_id` when present and `source_id`/`utterance_id` otherwise, matching
the study configuration.

## Model capability matrix

| Model | Pinned revision / audited source | Exact or viable runner | Input and scalar convention | Readiness |
|---|---|---|---|---|
| Spectra-AASIST | [`lab260/Spectra-AASIST`](https://huggingface.co/lab260/Spectra-AASIST/tree/eb65c2662d9e646d72557b3f4bdd08b000068c7f) `eb65c2662d9e646d72557b3f4bdd08b000068c7f` | Local `spectra-aasist.onnx` via ONNX Runtime CUDA; PyTorch source is present but its SSL dependency is absent | ONNX input `wav: float32[B,64600]`; output `logits: float32[B,2]` was inspected locally. Card: 16 kHz mono; index 0 spoof, 1 bona fide. | **Ready only after parity gate.** ONNX preprocessing must be verified, not assumed. |
| AASIST | [`SpeechAntiSpoofingBenchmarks/AASIST`](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST/tree/16774d458d86d2a021ae31646c1bf66a5331b53e) `16774d458d86d2a021ae31646c1bf66a5331b53e` | Local `aasist.onnx` via ONNX Runtime CUDA | ONNX input `wav: float32[B,64600]`; output `logits: float32[B,2]` inspected locally. Card specifies 16 kHz mono, deterministic first 64,600 samples/tile short clips, and logit 1 as higher bona-fide evidence. | **Ready only after parity gate.** |
| Res2TCNGuard | [`SpeechAntiSpoofingBenchmarks/Res2TCNGuard`](https://huggingface.co/SpeechAntiSpoofingBenchmarks/Res2TCNGuard/tree/4624265fa5e88c0abe425e37c278f3a9288aa914) `4624265fa5e88c0abe425e37c278f3a9288aa914` | Upstream-pinned standalone `evaluate.py` + `_net.py` + `best_1.495.pth`; generic ONNX is a second candidate | Exact standalone evaluator: float32 mono 16 kHz; first 64,600 samples/tile short clips; model returns `logits[:,1]`, higher bona-fide. | **Acquire ~0.9 MB code+checkpoint, then parity gate.** It is intentionally not yet downloaded. |
| W2V2-AASIST | [`SpeechAntiSpoofingBenchmarks/W2V2-AASIST`](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST/tree/196128e5a5101d5cb6ac7701597891bc7de7e7b5) `196128e5a5101d5cb6ac7701597891bc7de7e7b5` | `w2v2-aasist.onnx` after acquisition; do **not** rely on the advertised Python wrapper as shipped | Card states 16 kHz mono, first 64,600/tile short, `logits[:,1]` is bona fide. The local ONNX signature remains uninspected because the 1.26 GB artifact is absent. | **Blocked pending ONNX acquisition + signature/parity validation.** |

### Per-model evidence and dependencies

**Spectra-AASIST.** The local ONNX file is 1,265,496,989 bytes. ONNX Runtime
exposes `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and
`CPUExecutionProvider`; use CUDA for the reference scorer, not TensorRT/FP16,
until rank parity is recorded. The model card's PyTorch quickstart applies
pre-emphasis before its window, whereas the ONNX graph exposes only `wav`.
Consequently, raw waveform versus externally pre-emphasized input is a
*preprocessing question to test in the parity gate*, not an assumption. The
repository's PyTorch `model.py` imports `transformers.Wav2Vec2Model` and the
card says it may fetch XLS-R 300M; `transformers` is not installed in the
current environment. Avoid that route for initial H2.

**AASIST.** The local ONNX file is 1,615,195 bytes and has the same fixed
input/output signature. The Hub revision has `AASIST.pth` and `trt_aasist.py`,
but it does *not* contain the `_net.py` and `aasist.py` files claimed by its
README. Therefore its documented `AASIST().load()` invocation is not currently
reconstructible from that pinned revision. The generic ONNX route is the viable
initial runner. Its output-to-Arena parity remains mandatory.

**Res2TCNGuard.** The pinned revision contains the complete small standalone
path: `_net.py`, `evaluate.py`, and `best_1.495.pth` (837,865 bytes). The
evaluator has only `torch` and `numpy` requirements for in-memory scoring;
its file demonstration additionally uses `soundfile` and `scipy`. All four
are currently installed. The Arena-specific `res2tcnguard.py` also requires
`speech_spoof_bench`, which is not installed, so use `evaluate.py` instead.
The pinned ONNX (834,633 bytes) is a potential future performance path but its
input/output signature has not been inspected because it is not local.

**W2V2-AASIST.** The pinned revision provides `LA_model.pth` (1,271,633,441
bytes), `w2v2-aasist.onnx` (1,264,789,106 bytes), and a TensorRT helper. It
does not provide `_net.py` or `w2v2_aasist.py`, despite the model card naming
them; the helper imports that absent module. The PyTorch route would additionally
need the external XLS-R base model indicated by the card. Do not spend time
repairing that path for the initial draft. Download the declared ONNX artifact
to the HDD only when the three-model run and quality path are working, inspect
its graph, then calibrate it against the pinned Arena scores.

## Fixed waveform/scoring contract

1. Decode original FLAC bytes to float32 mono at 16 kHz. Record whether a
   resample or channel average was needed; the pinned discovery data should
   require neither.
2. Apply the H2 transformation at 16 kHz. Do not write lossy audio: persist
   intermediate derived clips as lossless FLAC under the HDD run directory, or
   regenerate exactly from a recipe hash and seed.
3. Apply each detector's fixed `first 64,600 samples; tile-repeat if shorter`
   window **after** the transformation. Spectra is the sole exception until its
   parity test decides whether external pre-emphasis is part of its ONNX path.
4. Score both original and transformed audio with the same model process,
   provider, batch size, and preprocessing. Save two logits, raw scalar,
   `score_spoof`, and input waveform SHA-256.
5. Treat a change in leading silence as a potentially positional intervention:
   score a no-position-shift endpoint-silencing variant separately from an
   add/trim-duration variant. Do not call the latter a pure silence effect.

Because all fixed-window models privilege the initial 4.04 seconds, avoid
random detector crops entirely. For clips longer than the window, log the
unwindowed and detector-windowed duration, and report the restricted
`duration <= 64,600` sensitivity analysis before making a cue-specific claim.

## Safe candidate intervention arms

The following are candidates; select arms only for H1-frozen cues. Each must
log the achieved feature change using the frozen v1_28 registry, not merely its
requested DSP parameter.

| Arm | Content-preserving implementation | Target/manipulation check | Main confound to report |
|---|---|---|---|
| `drc_cf3`, `drc_cf6` | Deterministic wideband compressor (fixed attack/release/ratio), followed by integrated-loudness restoration and limiter-free peak check | median reduction in `crest_factor_db` near 3 or 6 dB; record RMS-CV, spectral, F0 and phase deltas | Compression changes dynamics beyond crest factor; it is a waveform-family intervention, not a pure crest causal claim. |
| `endpoint_silence` | Energy/VAD endpoint detection, then replace only detected leading/trailing non-speech samples with zeros while preserving length and speech sample positions | change `silence_fraction`; verify the interior speech-core hash is unchanged | It changes endpoint noise floor, not silence duration. |
| `fixed_duration_silence` | VAD-extract speech core and deterministically pad to a specified endpoint duration; preserve all core samples and record length/position change | requested endpoint silence duration and achieved `silence_fraction` | Detector-window position can change; present as a separate positional arm. |
| `tilt_plus3`, `tilt_minus3` | Fixed-phase STFT or stable shelf filtering, then loudness restore | direction and magnitude of `spectral_slope_db_per_khz` plus low/high-energy ratio | Frequency weighting can affect loudness and crest factor; retain all measured deltas. |
| `allpass_a`, `allpass_b` | Causal stable first-order all-pass cascade, e.g. `H(z)=(a+z^-1)/(1+a z^-1)` with `|a|<1`; float32, no resampling | phase features (`group_delay_var`, `inst_freq_dispersion`) change while spectral-magnitude deltas remain within predeclared tolerance | STOI can be insensitive to phase artifacts; use the ASR and spectral-invariance checks as well. |
| `polarity` | Multiply by -1; preserve samples/length otherwise | exact amplitude-feature invariance expected | A detector response is a diagnostic invariance failure, not cue evidence. |
| `small_gain` | Global gain limited to a magnitude that satisfies the existing ±0.2 LU gate | nonzero but `abs(LUFS drift) <= 0.2` | The protocol's unspecified “gain” control cannot be a multi-dB gain under this gate; record its actual small magnitude. |

The current implementation already supplies feature measurements needed to
verify the manipulation: `crest_factor_db`, `integrated_lufs`,
`silence_fraction`, `spectral_slope_db_per_khz`,
`low_high_energy_ratio_db`, `group_delay_var`, and
`inst_freq_dispersion` in `src/audio_features.py`. It should be reused rather
than defining an arm-specific incompatible measure.

## Quality gates and required records

The H2 protocol already requires, per paired waveform: STOI >= 0.95,
original-versus-transformed ASR WER <= 5%, absolute loudness drift <= 0.2 LU,
and no clipping; an arm is dropped below 90% pass rate. The environment has
`pystoi`, `pyloudnorm`, `jiwer`, `scipy`, `soundfile`, and `ffmpeg`
(including `acompressor`, `loudnorm`, and `alimiter`). It currently has
**no ASR runtime**: `whisper`, `faster_whisper`, `transformers`,
`speechbrain`, and `vosk` are all absent. H2 cannot claim those quality
gates have passed until a fixed ASR model and its revision are installed and
recorded.

For each proposed pair, retain a quality row even when it fails:

```text
sample_id, dataset, label, source_id/speaker_id, arm, arm_parameters_json,
seed, original_wave_sha256, transformed_wave_sha256, sample_rate,
original_samples, transformed_samples, detector_samples,
stoi, original_transcript, transformed_transcript, wer,
original_lufs, transformed_lufs, lufs_delta, peak, clipping_fraction,
target_feature_before, target_feature_after, all_28_feature_deltas,
pass_stoi, pass_wer, pass_loudness, pass_clip, pass_target, retained, reason
```

Compute all gates before loading/scoring the detector and enforce the 90% rule
overall and report its dataset-by-label breakdown. Also require the
predeclared target-feature direction; a perceptually acceptable transform that
does not alter its intended cue is a failed manipulation, not an H2 sample.

## Efficient, reproducible execution layout

- Generate and quality-gate pairs on CPU; use deterministic seeds and bounded
  workers. Avoid re-decoding the same clip for every model.
- After quality freeze, run one independent model per GPU: CUDA 0 Spectra,
  CUDA 1 AASIST, CUDA 2 Res2TCNGuard; reserve CUDA 3 for W2V2 acquisition,
  parity, or a second pass. Do not use DDP.
- H2 is inference, not new model training. The project's BF16 requirement
  applies to newly trained variants. Use reference float32/CUDA first; only
  enable TensorRT/FP16 after a recorded parity check. No model should silently
  use BF16 simply because a GPU supports it.
- Measure the fastest safe batch size separately for every scorer (candidate
  values 1, 2, 4, 8, 16, 32) on an unmodified calibration batch, recording
  throughput, peak VRAM, provider, and exact model hash. Do not reuse the
  Res2TCNGuard card's historical `batch_size=4` without a local sweep.
- Store large derived audio, engines, temporary data, and logs below
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/<run_id>/`.
  Commit only recipe/config hashes, manifests, compact quality tables, parity
  reports, results, and plots.

## Blocking items before a confirmatory H2 result

1. Freeze H1 candidate features and exact arm set before viewing H2 scores.
2. Finish the active ASVspoof2021_DF acquisition; download the confirmation
   datasets only according to the H1 result plan.
3. Implement the fixed ASR WER gate and pin its model/runtime revision.
4. Acquire the tiny Res2TCNGuard inference bundle and validate score parity.
5. Implement the generic fixed-window ONNX scorer and validate Spectra/AASIST
   parity. Do not assume external pre-emphasis for Spectra's ONNX graph.
6. Acquire and validate W2V2 ONNX only after the three-model path works, or
   audit and freeze the predeclared XLSR-SLS fallback before H2 responses;
   record any continuing artifact/package failure explicitly.
7. Record the gain-control magnitude before results: multi-dB gain conflicts
   with the existing ±0.2 LU quality threshold.

## Audited sources

- Local dataset cards and Parquet schema under the pinned paths in
  `data/arena-index.yaml`.
- Local Spectra card/model source and local Spectra/AASIST ONNX graph signatures.
- Pinned Hub file listings and raw source for
  [Res2TCNGuard](https://huggingface.co/SpeechAntiSpoofingBenchmarks/Res2TCNGuard/tree/4624265fa5e88c0abe425e37c278f3a9288aa914),
  [AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST/tree/16774d458d86d2a021ae31646c1bf66a5331b53e),
  [W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST/tree/196128e5a5101d5cb6ac7701597891bc7de7e7b5),
  and [Spectra-AASIST](https://huggingface.co/lab260/Spectra-AASIST/tree/eb65c2662d9e646d72557b3f4bdd08b000068c7f).
- Existing H2 protocol at `experiments/h2_causal_interventions/protocol.md`,
  study configuration at `configs/study.yaml`, and frozen feature registry at
  `src/audio_features.py`.
