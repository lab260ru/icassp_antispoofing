# H6 protocol — within-class published-score agreement atlas

Status: locked on 2026-08-10 before any H6 input freeze, score analysis, model
execution, waveform/feature access, or H6 result inspection.

## Question and prediction

Do the eight independently published anti-spoofing score artifacts produce
homogeneous rankings within each registered class across the five core corpora?
H6 predicts heterogeneous pairwise agreement. This is a descriptive
architecture-diversity audit; it cannot identify a feature, attribute a score
to a cue, establish model independence, or alter H1/H2/H2B/H3/H4 decisions.

## Immutable inputs and firewall

H6 may read only the pinned five dataset label indexes and the original eight
published Arena score artifacts per core corpus. It must use exact sample-ID
joins and normalize every artifact to the existing registered increasing-spoof
orientation. It may not read waveform files, feature products, H1 association
tables, H4 matrices, H2/H2B outputs, model code/weights, ASR, detector runners,
or any result summary. All input paths, revisions, byte sizes, and SHA-256
values are sealed in an H6 freeze before pairwise correlations are computed.

## Fresh label-only source freeze

The freeze verifies the dataset-label revision and selects at most 5,000 unique
sample IDs per dataset/class using SHA-256 seed 2611. It records the complete
label-only selection manifest before score values are read. A missing score or
duplicate `(model, sample_id)` key remains an unavailable matrix cell; no
intersection re-selection, imputation, model substitution, or repaired ID
matching is permitted.

## Registered matrix

For five datasets, two classes, and all 28 unordered model pairs, materialize
exactly 280 cells. Each complete cell reports exact joined count, within-class
Spearman score agreement, and a 95 percent sample-cluster percentile bootstrap
interval from 200 resamples. The per-cell generator seed is SHA-256 derived
from the base seed and full cell identity. A source identifier may be used only
for clustering if the pinned label index provides it; otherwise each stable
sample ID is the bootstrap cluster. No label or model pair is selected after
results are observed.

## Summaries, interpretation, and stop rule

The exact 28 model-pair summaries report all ten dataset/class cells, their
median correlation, and cross-cell range. The matrix is descriptive only;
neither high nor low agreement denotes architecture independence, detector
reliance, causal sensitivity, or model quality. Stop and record failure on any
hash/revision/join/orientation validation failure or incomplete 280-cell matrix.
Do not modify the sample cap, seed, models, datasets, bootstrap count, or
correlation interpretation after inspection.
