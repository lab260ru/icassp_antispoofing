# Research Findings

## Research Question

Can documented content-aligned natural/synthetic pair-ranking supervision
improve the external transfer of a compact speech anti-spoofing detector beyond
same-data BCE and an equal-budget random-pair ranking control?

## Current Understanding

The original feature/intervention investigation remains valuable boundary
evidence, but its crest result is negative and cannot be the paper's primary
contribution. H9-PCR supplies a clean positive controlled-transfer result:
under a locked ODSS-only training protocol, content-aligned pair ranking lowers
two-target macro EER to 45.02%, versus 51.78% for same-pool BCE and 52.01% for
an equal-budget random-pair control. The effect is lower on both fixed targets
and both 2,000-replicate shared-ID bootstrap intervals lie below zero. This is
a narrow training-method result, not a statement about universal cue causality
or state-of-the-art anti-spoofing.

## H8 positive-pivot status

The prior feature/intervention loops are closed and cannot be promoted into a
positive paper result. H8-SF was a separate, pre-result cross-corpus study:
it tests whether source-only, corpus-by-class robust fusion of eight published
detector score streams can improve blind external-corpus EER over uniform and
ordinary learned fusion. It did improve those two weak fusion comparators on
every target, but it fails its stronger source-selected single-expert
falsification: Spectra-AASIST has 0.185% mean target EER while GroupDRO has
9.309%. H8-SF is terminal and cannot support the paper.

## H9 paired-counterfactual status

H9-PCR is a new, protocolled training study, not a reinterpretation of any
earlier detector-score or feature result. It uses 7,961 documented ODSS
natural/VITS/FastPitch--HiFi-GAN groups (23,883 shared B1/B2/P trials), a
voice-disjoint source development split, and 15,922 edges for both the
content-aligned P and the language/corpus/generator-stratified random-pair B2
control. A fresh 172,102-parameter Res2TCNGuard is trained in BF16 from four
fixed initializations; no historical pretrained checkpoint is loaded.

After the source-only lambda/checkpoint choices were sealed, the terminal
evaluator materialized SONAR and ArAD, found no exact canonical source--target
waveform collision, wrote raw predictions before label access, and evaluated
all 12 method--seed checkpoints in one call. P's EER is 47.07% on SONAR and
42.98% on ArAD, below both controls on each. Its macro improvement is -6.70 pp
versus BCE (95% CI [-8.25, -5.25]) and -6.97 pp versus random-pair ranking
([-8.54, -5.30]). The full interpretation and hashes are in
`experiments/h9_paired_counterfactual/results/H9_TERMINAL_EVALUATION_001.md`.

## H10 fresh-target follow-up status

H10 replayed the sealed H9 twelve-checkpoint panel on the independently
declared CD-ADD target after a two-pass label firewall. P improves EER to
41.66% versus B1 45.42% and B2 48.92%, and both fixed paired bootstrap
intervals are below zero. However, its 8.30% relative reduction versus B1
misses H10's predeclared 10% practical-effect rule. H10 is therefore a
terminal no-retune follow-up, not an added paper claim or a reason to select a
new target. See
`experiments/future_directions/results/H10_CDADD_TERMINAL_EVALUATION_001.md`.

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
- H9 controlled transfer: all eight predeclared terminal gate conditions pass
  after a fully source-only selection stage. P's four-seed ensemble improves
  relative to both same-pool controls on both fixed external targets, and its
  two fixed paired bootstrap intervals exclude zero. Absolute target EER is
  still weak and individual seed results vary; this supports a narrow
  controlled-transfer claim, not SOTA, a causal representation explanation,
  or arbitrary-corpus generalization.
- H4 score-free label--cue atlas: all 420 locked feature--label AUROC/bootstrap
  cells and 84 aggregations completed from revalidated feature tables only;
  23 units meet the terminal descriptive rule. Full-waveform crest factor has
  negative signed AUROC in the three ASVspoof corpora, positive signed AUROC in
  InTheWild, and near-zero separation in ASVspoof5. This is corpus dependence
  of label separation, not evidence about any detector or causal cue.
- H5 score-free paired view-invariance atlas: all 420 locked feature/view/corpus
  cells and 84 aggregations completed from a fresh seed-2610 sample freeze,
  without materializing labels or score artifacts. Seventeen units satisfy the
  strict all-corpus descriptive view-stability rule. Six InTheWild
  silence/clipping cells have no finite paired values and remain explicit
  degenerate cells; they are not repaired. View measurement agreement or
  disagreement is not evidence about a detector, labels, causality, or a
  preferred preprocessing choice.

## Patterns and Insights

The decisive H9 pattern is that the same rank-margin term is not sufficient:
random pairing (B2) fails to match content-aligned P despite identical source
pool, edge count, rank weight, architecture, optimizer, batch schedule, source
selection, and target evaluation. This is consistent with pair alignment being
useful, while the current data do not identify which learned representation
property causes that transfer effect. Existing H1--H7 evidence remains a
separate caution against overgeneralizing waveform-cue stories.

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
- H4's 23 stable label--cue units are terminal descriptive atlas entries. They
  must not be used as new H1/H2/H2B/H3 candidates or a model-training signal.
- H9's target panel is complete. Do not tune its rank weight, source split,
  epochs, checkpoint, seed ensemble, loss, target subset, or bootstrap after
  observing the result. Report the heterogeneous individual seeds and weak
  absolute EER alongside the passing ensemble gate.

## Open Questions

- Can a fresh independently frozen replication add a third external target or
  a second transparent architecture without using the completed H9 targets to
  tune that extension?
- How can content alignment be characterized more directly (for example with
  independently obtained transcript/content metadata) without turning the
  filename-derived ODSS pair key into an overclaimed causal mechanism?
- If a future study examines an H4 unit, can it declare a fresh score-independent
  discovery/confirmation/intervention design without using the H4 atlas for
  post-hoc cue selection?
- H5 is complete and sealed: only 17/84 registered view-pair/feature
  aggregations satisfy the strict all-corpus descriptive stability rule. It
  contextualizes feature measurement dependence but cannot alter H1/H2/H2B/H3/H4
  or select a feature for any later study.
- H6 has a complete independent label-only freeze: 50,000 selected IDs and 40
  raw-score byte identities were sealed before score parsing. Its completed
  agreement matrix is descriptive only and cannot reopen any feature or causal
  decision.
- H6 score-agreement atlas: all 280 planned within-class model-pair cells and
  28 summaries completed with 5,000 exact joins per cell. Agreement ranges from
  -0.629261 to 0.868770 across the fixed registry, demonstrating descriptive
  corpus/class/model-pair heterogeneity. It neither rates models nor links a
  score pattern to a waveform cue, causal mechanism, or training action.
- H7 fixed-vector feature transfer: one independently frozen 28-feature,
  full-waveform logistic baseline was trained on four corpora and evaluated on
  the fifth in all five leave-one-corpus-out folds. AUROC is 0.907785,
  0.845107, and 0.780957 when holding out the three ASVspoof corpora, but
  0.534590 on InTheWild and 0.588320 on ASVspoof5 (median 0.780957). This
  illustrates feature-only label-transfer dependence across corpus families;
  it reads no detector score and is not evidence of detector reliance,
  causality, a selected cue, or mitigation performance.

## Optimization Trajectory

| Run | Hypothesis | Metric | Status |
|---|---|---|---|
| bootstrap | setup | n/a | active |
