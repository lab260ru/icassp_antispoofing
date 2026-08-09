# Research Findings

## Research Question

Which waveform-level signal cues associate causally, rather than merely
correlationally, with speech anti-spoofing detector scores across datasets and
architectures, and can the resulting evidence improve robustness?

## Current Understanding

The five-corpus H1 screen now establishes that some registered feature/view/class
units satisfy the study's descriptive cross-corpus association rule. It also
rules out the central frozen crest-factor candidate as a portable adjusted
association: its ASVspoof5 adjusted estimate is null. No causal
shortcut-sensitivity claim has been established; only a completed,
quality-frozen paired detector intervention can test that third question.

## Key Results

- Input-integrity validation: all 71,237 published Spectra-AASIST scores joined
  exactly to ASVspoof 2019 LA labels. The stored score is increasing bonafide
  evidence, so analysis uses its negation as spoof evidence. This establishes a
  valid artifact interface, not a feature or causal result.
- Feature-pipeline validation: the v1_28 registry extracted complete
  non-voice descriptors and 95% voice-quality coverage over 200 balanced
  ASVspoof 2019 LA samples and three views. Missing unvoiced measures will be
  modeled explicitly rather than filled with zeros.
- Score-panel validation: the ASVspoof 2019 LA analysis panel now covers all
  eight prespecified architectures (569,896 model–utterance rows). Every model
  has exactly one score for each of the 71,237 published trials and the stored
  analysis variable is normalized to increasing spoof evidence. This is input
  validation, not a score-feature result.
- Full-feature validation: the locked ASVspoof 2019 LA discovery subset is
  balanced (5,000 trials per class), complete across three waveform views, and
  has no duplicate utterance–view keys. The manifest does not provide attack
  IDs; the analysis records that control as a single explicit missing category
  instead of synthesizing attack metadata.
- First H1 discovery screen: 1,344 exact-join, within-class associations across
  eight architectures and three waveform views have been computed. Crest factor
  has large negative adjusted associations with spoof evidence for
  Spectra-AASIST on all three views (partial $\rho$: -0.454, -0.445, -0.327),
  but its signs and magnitudes vary across other architectures, class strata,
  and views. This is evidence for heterogeneous sensitivity, not a claim of a
  universal crest-factor shortcut.
- Second H1 discovery screen: on ASVspoof2021 LA, Spectra-AASIST's spoof-class
  full-waveform crest association has the same negative direction but attenuates
  to partial $\rho=-0.099$ (BH $q=4.9\times10^{-12}$). Other artifacts remain
  heterogeneous. Two corpora are therefore consistent with neither a uniform
  effect size nor a universal detector claim; the final discovery corpus must
  complete before candidates are frozen.
- H2 runner-parity validation: on a frozen 128-clip, score-independent
  calibration set, AASIST's raw ONNX path matches the published Arena ordering
  (Spearman 0.999994); Spectra-AASIST matches only when external 0.97
  pre-emphasis is used (0.999971). This validates model-specific waveform
  preprocessing for later paired scoring, not feature sensitivity or causality.
- Held-out H1 confirmation: the preregistered Spectra-AASIST spoof/full-waveform
  crest-factor test is small and adjusted-negative on InTheWild
  (rho=-0.032606, q=0.000523; 95% clustered CI [-0.070045, -0.001718]), but its
  raw association is null. On ASVspoof5 the raw association is negative
  (rho=-0.155568; 95% CI [-0.185310, -0.124442]), whereas the registered
  adjusted estimate is null (rho=0.001383, q=0.937089; 95% CI
  [-0.029092, 0.032219]). Neither estimator is substituted for the other.
- Five-corpus H1 aggregation: all 168 registered units were evaluated under the
  fixed partial-Spearman/BH rule without reselection. Nineteen units meet the
  descriptive cross-corpus association criterion; the discovery-frozen
  Spectra/crest/spoof candidate does not, because it fails the required
  confirmation significance and direction conditions. These associations do
  not justify post-hoc H2 expansion or a causal interpretation.
- H2 quality-gate result: the full 1,000-clip ASVspoof2019 LA crest/control
  panel completed detector-free, but three frozen arms miss the 90% retained
  threshold (DRC-3 18.1%, DRC-6 0.6%, and +0.1 dB gain 64.6%). The formal
  quality freeze refused before any transformed waveform reached a detector.
  This is evidence that the registered transformations are incompatible with
  the content/loudness/clipping gate at this setting, not evidence about model
  sensitivity.
- H2B Q1 quality result: the independent DeepVoice 256-clip × 9-arm,
  detector-free calibration completed all 2,304 pairs. The precommitted
  selection rule returns no non-control family. The closest arm,
  -0.5 dB/oct spectral tilt, has a 95% lower Wilson retention bound of
  0.89624, below the locked 0.90 threshold; endpoint zeroing retains only
  15/256 pairs. This is a negative transform-feasibility result, not a
  detector, feature, EER, score-delta, or causal result.
- H2B Q0 input integrity: DeepVoice is now pinned as an independent
  quality-calibration source, and its balanced 256-identity calibration
  manifest is frozen from labels only. This establishes a clean score-blind
  input boundary, not waveform-quality feasibility.

## Patterns and Insights

The study is designed to prevent label, corpus, duration, loudness, codec, and
speaker effects from being mistaken for detector reliance. Existing Arena score
artifacts provide broad architecture coverage without spending GPU time on
baseline reproduction.

## Lessons and Constraints

- Existing Arena outputs are trusted artifacts; do not rerun published baseline
  inference merely to reproduce them.
- All material claims require a protocol, pinned inputs, and an analysis note.
- The first draft must not contain invented or placeholder numerical findings.
- A tested feature cannot also be treated as an independent control in its own
  partial correlation. The analysis removes that impossible self-control before
  computing any estimate.
- With 5,000 samples per within-class test, multiplicity-adjusted p-values can
  be extremely small. Candidate selection must prioritize the fixed
  cross-corpus and cross-model portability gate, then held-out confirmation and
  interventions, rather than significance alone.
- Corpus-scoped result paths are required for reproducibility: a rerun for one
  dataset must never overwrite another dataset's association or score catalog.
- H2 must retain its model-specific parity contract: raw AASIST and
  pre-emphasized Spectra inputs are not interchangeable, and no transformed
  waveform can be scored until the same contract is applied.
- A failed quality gate is an experiment boundary. A revised arm family needs
  a new protocol, a disjoint score-blinded calibration cohort, and a fresh
  quality-confirmation freeze; it cannot inherit the completed H2 panel.
- Q1 selected no H2B arm. Do not relax its Wilson threshold or tune its arm
  grid after inspection; Q2--Q4 are not licensed by this outcome.

## Open Questions

- Which public score artifacts expose stable sample IDs that can join audio
  manifests without heuristic matching?
- Does the already-frozen exploratory crest intervention show paired score
  sensitivity after quality gates? No: its frozen arms fail before scoring;
  a future test would require a new, independently frozen outer-loop protocol.
- Is the strongest paper a principled negative crest-factor result plus an
  architecture-sensitivity atlas rather than a mitigation story?
- Can a newly predeclared transformation family, on a newly frozen cohort,
  clear the quality-first gate without material collateral changes? The first
  H2B grid does not.

## Optimization Trajectory

| Run | Hypothesis | Metric | Status |
|---|---|---|---|
| bootstrap | setup | n/a | active |
