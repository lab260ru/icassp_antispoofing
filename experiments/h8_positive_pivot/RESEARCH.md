# H8 research card

## Candidate contribution

**Corpus-Robust Copula Fusion for Frozen Speech Anti-Spoofing Systems.** Train
one small nonnegative score fuser on three source corpora only, with
source-corpus × class GroupDRO and monotone rank/probit score features. Test it
once on a fixed set of untouched external corpora and compare it to every
individual system, uniform rank fusion, and matched ERM fusion.

## Why it may work

Individual frozen systems specialize to different corpora and their
within-class scores disagree. Raw scores have incompatible calibration across
systems/corpora. Rank-space mapping removes monotone scale mismatch; GroupDRO
downweights a source-specialist solution when it harms an underperforming
source group. Neither target labels nor target-specific fusion weights are used.

## What would make it strong

The proposed fuser must improve corpus-macro mean EER and worst-target EER over
both learned ERM fusion and uniform rank fusion, with paired target bootstrap
support and no target-dependent method/weight selection. The paper would then
be about robust integration of heterogeneous frozen systems, not detector
retraining or waveform causality.

## What would falsify it

No consistent win over both fusion baselines on the blind target panel, or a
win concentrated in one target while worst-target EER worsens. In either case
we stop H8-SF and move to the separately protocolled SSL-probe fallback.
