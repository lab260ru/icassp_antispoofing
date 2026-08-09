# Data and artifact registry

All large data remain outside Git at
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`. This directory stores the
versioned registry, sample manifests, small tables, and provenance notes.

`arena-index.yaml` is the source of truth for every Hugging Face dataset/model
revision, file path, license, score artifact, checksum, label convention, and
local HDD destination. A sample joins a published score only through its pinned
`sample_id`; filename-only heuristic joins are prohibited.

Published Arena metrics are retained as canonical baseline metrics. Per-sample
score files may be parsed for association analyses, but no baseline model is
rerun solely to reproduce those scores.
