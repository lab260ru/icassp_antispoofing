# Supplementary H1 adjustment and bootstrap disclosure

## Scope

This compact table documents covariate availability for the completed H1
rank-residual partial-Spearman design. It is a display-only copy of the sealed
score-free reporting artifact at
`experiments/h1_feature_association/results/covariate_availability_audit_001/`.
It computes no association, p-value, confidence interval, candidate, ranking,
or intervention result.

The source CSV SHA-256 is
`70cd65c2608181c23eac2e4d673fcfc78e9397409543a5d5ea53e8531ee1f070`;
the source-provenance JSON SHA-256 is
`ed8a42bddce44403efda0df86792e53450fc6fef4ce1d0367bebd041907d9152`.
That JSON identifies the five full feature-table inputs, their byte hashes and
schemas, the strict metadata projection, and the forbidden operations.

## Adjustment construction

For every H1 cell, `partial_spearman` uses an intercept; ranked duration and
integrated loudness with median fill; categorical speaker and attack dummies
with an explicit missing category; and removes a tested feature or response
from the controls if it would otherwise control itself. Label `0` is bona fide
and label `1` is spoof.

## Coverage by exact H1 cohort

| Dataset | Label | Samples | Duration finite | LUFS finite | Speaker nonempty / unique | Attack nonempty / unique | Source ID nonempty / unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ASVspoof2019_LA | 0 | 5,000 | 5,000 | 5,000 | 5,000 / 67 | 0 / 0 | 5,000 / 5,000 |
| ASVspoof2019_LA | 1 | 5,000 | 5,000 | 5,000 | 5,000 / 48 | 0 / 0 | 5,000 / 5,000 |
| ASVspoof2021_LA | 0 | 5,000 | 5,000 | 4,999 | 5,000 / 67 | 5,000 / 1 | 5,000 / 5,000 |
| ASVspoof2021_LA | 1 | 5,000 | 5,000 | 5,000 | 5,000 / 48 | 5,000 / 13 | 5,000 / 5,000 |
| ASVspoof2021_DF | 0 | 5,000 | 5,000 | 5,000 | 5,000 / 93 | 5,000 / 1 | 5,000 / 5,000 |
| ASVspoof2021_DF | 1 | 5,000 | 5,000 | 5,000 | 5,000 / 62 | 5,000 / 110 | 5,000 / 5,000 |
| InTheWild | 0 | 19,963 | 19,963 | 19,962 | 19,963 / 54 | 0 / 0 | 19,963 / 19,963 |
| InTheWild | 1 | 11,816 | 11,816 | 11,816 | 11,816 / 54 | 0 / 0 | 11,816 / 11,816 |
| ASVspoof5 | 0 | 5,000 | 5,000 | 5,000 | 5,000 / 737 | 5,000 / 1 | 5,000 / 5,000 |
| ASVspoof5 | 1 | 5,000 | 5,000 | 5,000 | 5,000 / 367 | 5,000 / 16 | 5,000 / 5,000 |

## Held-out confirmation bootstrap

The two already sealed H1 confirmation intervals use 2,000 label-stratified
clustered percentile replicates, seed 2609, and `speaker_id` when nonempty
(otherwise `source_id`). The selected spoof slice has 54 speaker clusters in
InTheWild and 367 in ASVspoof5. This supplement does not rerun that bootstrap.

This document is prepared for a future authorized submission supplement; it is
not evidence that the conference accepts a supplementary package or that a
permanent public archive has been created.
