# H8 data contract

## Development sources

- ASVspoof2019_LA
- ASVspoof2021_LA
- ASVspoof2021_DF

These are frozen-system source domains, not claimed official training splits.
Training rows are selected with a new deterministic corpus × class manifest.

## Blind primary targets

- CFAD
- CVoiceFake_small
- DECRO
- LibriSeVoc
- XMAD

The freezer must verify each target's eight-model common score coverage and
revision identity from `data/arena-index.yaml` before reading target labels.
Target score CDFs may be materialized label-free only after the protocol is
committed. Target labels are read only by the final evaluator.

Final point estimates use all retained common-score target rows. The evaluator
uses a separate method-independent uncertainty panel of at most 10,000 stable
IDs per target × label, selected with the exact `H8SFBOOT|2608` SHA-256 rule
in `PLAN.md`, solely for its fixed 2,000-replicate bootstrap.

## Storage

Large source panels remain under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing`; H8 manifests and compact
metrics are preserved under `experiments/h8_positive_pivot/`.
