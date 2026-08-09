# Quality-first follow-up literature note

**Purpose:** contextualize a new outer-loop protocol after the current H2
crest/control panel failed its waveform-quality gate.  This is background for
future work; it does not alter the completed H1/H2 claims or add a citation to
the current paper.

## Sources checked

1. Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, and Nicholas
   Evans, *RawBoost: A Raw Data Boosting and Augmentation Method applied to
   Automatic Speaker Verification Anti-Spoofing* (2021),
   [arXiv:2111.04433](https://arxiv.org/abs/2111.04433).  RawBoost motivates
   raw-waveform augmentation as a practical route to robust anti-spoofing, but
   its goal is robustness to nuisance variability rather than identification
   of a single causal acoustic feature.
2. Xingming Wang, Bang Zeng, Suo Hongbin, Yulong Wan, and Ming Li, *Robust
   Audio Anti-spoofing Countermeasure with Joint Training of Front-end and
   Back-end Models* (Interspeech 2023),
   [doi:10.21437/Interspeech.2023-1166](https://www.isca-archive.org/interspeech_2023/wang23v_interspeech.html).
   The paper reports that an independently trained enhancement front end may
   distort anti-spoofing-relevant information; this supports recording all
   collateral feature deltas instead of treating an intended DSP parameter as
   a pure feature manipulation.
3. Nicolas M. Müller et al., *Speech is Silver, Silence is Golden: What
   do ASVspoof-trained Models Really Learn?* (ASVspoof 2021),
   [PDF](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.pdf).
   This establishes that a seemingly simple acoustic property can be a
   dataset-level confound and motivates preserving the distinction among label
   separation, within-class association, and controlled perturbation.

## Synthesis

The current H2 outcome is not evidence that waveform interventions are
unusable, or that any detector is invariant.  It is evidence that this
particular DRC/gain arm set cannot meet this project's strict content,
loudness, clipping, and target-direction contract.  Existing augmentation
work makes a quality-first redesign worthwhile, while the enhancement and
silence-artifact literature argues against choosing a replacement arm based on
detector scores or on a requested DSP setting alone.

The corresponding protocol is
`experiments/future_directions/H2B_QUALITY_FIRST_OUTER_LOOP_PROTOCOL.md`.
