# H2 clipping-gate correction before detector scoring

## What happened

`h2_quality_full_001` began a detector-free quality run on the committed
ASVspoof2019 1,000-clip panel. It was intentionally stopped after 133 of 4,000
pair checkpoints when inspection found that some original source clips already
had full-scale samples. The first implementation required *both* the source
and transformed clipping fractions to be zero. That criterion rejected polarity
and +0.1 dB gain controls even when they introduced no clipping at all.

No detector was imported, loaded, or scored. The incomplete checkpoints,
original-transcript cache, and run provenance remain under the HDD run directory
`runs/h2_causal_interventions/h2_quality_full_001/`; they are a diagnostic
implementation artifact, not a panel-gate result or causal finding.

## Corrected pre-score rule

The gate now records source clipping, transformed clipping, and
`added_clipping_fraction = transformed - source`. It accepts an example only
when the added fraction is at most zero. This tests the relevant intervention
property—whether the transform introduces clipping—without treating an
immutable pre-existing recording artifact as harm caused by the treatment.

The correction is made before any H2 detector response is viewed. It does not
change the frozen input identities, cue, arms, target direction, ASR runtime,
STOI threshold, WER threshold, or loudness threshold. The amended protocol is
committed before launching the distinct full run ID `h2_quality_full_002`.

## Regression coverage

`tests/test_h2_pre_score_pairs.py` now verifies that a peak-clipped source
whose polarity-reversed waveform has the same clipping fraction passes this
gate, while all clipping fractions remain visible in the quality row.
