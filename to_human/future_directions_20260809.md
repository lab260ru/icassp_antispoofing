# Future direction after the H2 quality-gate failure

## Decision

The current ICASSP draft will keep its conservative conclusion.  We will not
weaken the 90% quality gate, rescale the failed DRC arm, or turn the 19 observed
H1 associations into post-hoc interventions.

The next research loop is **H2B: quality-first cue intervention**.  It first
answers whether a waveform transform can both change a registered feature and
preserve content/loudness/peak integrity across domains, while all detector
scores remain unavailable.  Only after a disjoint quality-confirmation panel,
a separately declared H1 candidate, and four parity-validated scorers can it
measure detector sensitivity.

## Why it is worth doing

- It converts the H2 failure into a methodological result: transformation
  validity is a prerequisite, not a detail after a score change.
- It prevents a common ambiguity in augmentation studies: a desired DSP
  setting is not proof that only the intended feature changed.
- It preserves the current paper's credibility while creating a larger,
  publishable next study if the quality and causal gates eventually pass.

## First executable pilot

Run a 256-clip, score-blinded calibration of fixed endpoint-silence, mild
spectral-tilt, mild all-pass, polarity, and bounded-gain families.  Choose at
most one parameter point per family only by a predeclared retention-bound /
target-change / WER rule, then confirm it on a newly frozen 1,000-clip panel.
No detector or Arena score artifact will be read during these stages.

The locked public calibration source is the 5,053-trial DeepVoice corpus at
its Arena-pinned revision. It is independent of the five completed H1 corpora
and can be downloaded directly to the designated HDD location.

The full protocol and stop rules are in
`experiments/future_directions/H2B_QUALITY_FIRST_OUTER_LOOP_PROTOCOL.md`.
