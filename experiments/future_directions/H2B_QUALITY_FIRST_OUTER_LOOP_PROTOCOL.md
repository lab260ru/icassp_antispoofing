# H2B: Quality-first cue-intervention outer loop

**Status:** protocol only; no H2B audio, ASR, detector, or training run has
been launched.  This is a new outer loop, not a modification, rerun, or rescue
of `h2_quality_full_002`.

## Why a new outer loop is necessary

The original score-independent crest/control panel is permanently closed:
three arms retain fewer than 90% of pairs at the committed quality gate, so its
quality freeze intentionally produced no detector-score manifest.  Changing a
threshold, DSP parameter, arm count, sample, candidate cue, or model panel
would be post-result adaptation.  It therefore requires this separately
versioned protocol, a new score-blinded calibration cohort, and a fresh input
freeze.

This direction addresses a more basic empirical question before any detector
is allowed to read a transformed waveform:

> **Can a cue-directed waveform transformation attain the locked content and
> signal-integrity contract across speech domains while producing a measured,
> nontrivial change in its registered target feature?**

It is a problem-first feasibility study.  Its immediate stakeholder is the
anti-spoofing researcher who needs an interpretable perturbation but must not
confuse an audible or position-shifting change with causal feature reliance.

## Confirmatory boundaries

- H2B **does not** revive the failed Spectra-AASIST/full-waveform/spoof/crest
  candidate.  That candidate failed the completed five-corpus H1 portability
  condition.
- H2B **does not** promote any of the 19 already observed H1 association units.
  A detector-sensitivity follow-up needs a cue chosen under a future
  independently declared H1 discovery/confirmation split.
- The H2B quality result itself is not a detector result, an EER result, or
  evidence that any feature is causal.
- No score artifact, logit, detector implementation, or learned model may be
  loaded during parameter selection, feature-manipulation calibration, or
  quality-gate evaluation.

## Phased design

### Q0 — declare an independent waveform calibration source

Before any waveform decoding, declare a public, pinned speech corpus or split
that was not used to choose the completed five-corpus H1 candidate.  It must
provide stable IDs and 16-kHz mono waveform provenance.  Freeze a
label-stratified, source/speaker-clustered 256-clip calibration manifest from
metadata only (128 clips per class where labels exist; otherwise declare the
sampling frame explicitly).  Store the audio only under the HDD root and
commit its manifest, revision, labels/source metadata, and SHA-256 hashes.

**Gate Q0:** the manifest builder rejects response-like fields, detector files,
and score paths.  It must be committed before decoding a clip.

### Q1 — quality-only transform-family calibration

The candidate families are intentionally selected before calibration:

| Family | Calibration grid | Registered target | Collateral checks |
|---|---|---|---|
| Endpoint silence, fixed length | energy/VAD threshold and zeroed endpoint extent | `silence_fraction` | interior speech-core byte hash unchanged; duration/position unchanged |
| Spectral tilt | $\pm0.5$, $\pm1.0$, $\pm1.5$ dB/oct fixed-phase tilt | `spectral_slope_db_per_khz` and `low_high_energy_ratio_db` | crest, integrated LUFS, F0, phase, duration |
| All-pass phase | predeclared stable first-order coefficient cascades with $|a|\leq0.15$ | `group_delay_var` or `inst_freq_dispersion` | spectral-magnitude tolerance, LUFS, duration |
| Negative controls | polarity and a bounded gain whose *actual* LUFS drift is within 0.2 LU | invariance as appropriate | all v1_28 deltas and exact waveform/hash diagnostics |

For every grid point and clip, run the same locked metrics used by H2:
STOI $\geq0.95$, normalized original-to-transformed WER $\leq0.05$, absolute
loudness drift $\leq0.2$ LU, no transform-induced clipping, and the specified
target-direction check.  Record all failures; an unavailable metric fails.
The implementation must reuse `src/h2_waveform_transforms.py`,
`src/h2_asr_wer.py`, `src/audio_features.py`, and the current transform-induced
clipping definition rather than fork incompatible metrics.

**Selection rule, locked before Q1:** for each family retain at most one grid
point only if its *lower* Wilson 95% confidence bound for retained-pair rate is
at least 0.90 and its median target-feature change clears a family-specific
minimum declared in the Q1 manifest.  Among eligible points, select
lexicographically by (1) largest lower retention bound, (2) largest absolute
median target change in the registered direction, and (3) smallest median WER;
use canonical JSON parameter bytes as the final tie-breaker.  If no point is
eligible, declare that family unavailable—do not add a new grid point.

This gate is deliberately stricter than observing a raw 90% rate on 256
examples, so it does not carry an underpowered quality arm into scoring.

### Q2 — independent waveform-quality confirmation

Freeze a second, disjoint 1,000-clip (500-per-class where applicable)
multi-corpus panel from metadata only, after Q1 selection but before decoding
the panel.  Its candidate families and selected parameters are read-only from
the Q1 result.  Repeat Q1 quality metrics, require at least 90% retained in
**each** arm and corpus, and freeze the full table and hashes.  Do not launch a
detector unless this gate passes for every arm.

### Q3 — independent H1 candidate declaration

In parallel but without using Q1/Q2 detector-free outcomes to select a
feature, use new, explicitly declared H1 discovery corpora and a held-out
confirmation corpus to choose at most one `(feature, view, class, model)`
identity.  The candidate must satisfy its registered H1 rule before it can be
paired with a Q2-approved family.  The current five-corpus tables can be used
as background only, not as the H2B selector.

### Q4 — only then, multi-model paired scoring

Freeze an exact intersection of the Q2-quality-approved panel and the Q3 H1
identity.  Before any detector response, validate **four** executable runners
against their pinned Arena ordering on an untouched calibration manifest;
record preprocessing, orientation, sample-window rules, batch/worker sweep,
and GPU/provider.  The fourth runner may not be silently omitted.  Score
original/transformed pairs, retain logits and waveform hashes, and analyze the
predeclared paired delta with source/speaker clustered intervals.

H2B can use the word *causal sensitivity* only if all Q0--Q4 gates pass and
the original registered multi-dataset/multi-model effect criterion is met.

## Measurable pilot and stop rules

The next executable inner-loop run is Q1 on 256 score-blinded clips.  Its
proxy metric is the minimum, across a family's locked grid points, of the
Wilson lower retention bound subject to the target-direction check.  It is not
an EER, rank correlation, detector score, or model-selection metric.

Stop a family after its finite grid is exhausted without an eligible point.
Stop H2B before Q4 if Q2 or Q3 fails.  A quality-passing negative control is
useful only to validate the gate and must not be interpreted as feature
causality.

## Pre-execution acquisition and engineering checklist

1. Choose and pin the independent Q0/Q2 waveform source; if it is unavailable
   publicly, request access rather than silently substituting a current H1
   corpus.
2. Implement a score-free Q1 calibration CLI with immutable manifest/output
   and dedicated rejection tests.  It must never import detector modules.
3. Pin/verify the Whisper checkpoint and GPU-3 runtime as the existing H2
   quality runner does.  Record speed, worker count, and actual GPU mapping.
4. Acquire and parity-validate the W2V2-AASIST ONNX artifact, or separately
   audit and preregister a fourth-score fallback before Q4.  The three current
   runners cannot support the registered four-model causal conclusion.
5. Run the full test suite, record artifact hashes, and write a compact
   quality-result note before any detector process is started.

## Literature rationale

Raw-waveform augmentation can improve anti-spoofing robustness, but generic
augmentation does not establish that an intervention isolates a feature
([RawBoost](https://arxiv.org/abs/2111.04433)).  Independently applied speech
enhancement can also damage anti-spoofing-relevant information
([Wang et al., 2023](https://www.isca-archive.org/interspeech_2023/wang23v_interspeech.html)).
These observations motivate Q1/Q2's measured quality and collateral-feature
checks.  Dataset artifacts such as silence remain a separate concern
([Müller et al., 2021](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.pdf));
they motivate Q3's new H1 selection rather than converting the current atlas
into a new intervention list.
