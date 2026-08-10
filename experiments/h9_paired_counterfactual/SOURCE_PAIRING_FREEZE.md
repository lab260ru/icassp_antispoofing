# H9-PCR ODSS source pairing and split freeze

`scripts/freeze_h9_odss_pairs.py` is the only H9-PCR source-pair builder. It
is metadata-only: it accepts the pinned ODSS `labels.parquet` (or an equivalent
two-column metadata CSV) plus the pinned ODSS `README.md` and
`build_parquet.py`. It does not enumerate an audio shard, decode audio, load a
model, compute a source score, or touch SONAR/ArAD.

The ODSS documents pin the source-relative layout
`generator/corpus/speaker/stem.wav` and define `utterance_id` as that path with
`/` replaced by `__`. The builder reconstructs the path from the identifier,
and verifies optional `path`/JSON `notes` fields if supplied. It uses:

- voice key: `corpus/speaker`;
- content key: `corpus/speaker/stem`;
- conservative H9 group key: `voice_key::content_key`;
- pair ID: `source_corpus/speaker/stem` (with a separate `spoof_generator`
  column because one natural item may have both synthetic renderings).

Only a content group with exactly one `natural` (`label=0`) row and at least one
`vits` or `fastpitch-hifigan` (`label=1`) row is retained. This intentionally
excludes unpaired ODSS VCTK synthetic rows. A duplicate rendering for one
`(content_key, generator)`, a generator/label disagreement, or optional
path/notes disagreement is a `SOURCE_PAIRING_UNAVAILABLE` hard stop.

The split is frozen at seed `2909`, a 20% development fraction, and SHA-256
ordering of `voice_key` within documented language strata. Every complete group
from one `source_corpus/speaker` voice is assigned to the same source split;
thus the split is voice-disjoint as well as pair-disjoint. The builder hard
stops unless each observed language occurs in both splits and at least 1,000
complete groups occur overall **and in each split**.

The retained complete groups are the only source trial universe for B1, B2,
and P. Every unmatched ODSS row, including the VITS-only rows, is written to a
separate exclusion ledger with its IDs, reason, and hashes. It is unavailable
to all three conditions. B2 retains every P spoof edge but replaces its natural
partner with a deterministic content-unmatched natural row from the same
split, language, source corpus, and spoof-generator stratum. The emitted B2
edge table is hash-bound and directly auditable against P.

Run this only after the protocol commit and before any source audio decode:

```bash
PYTHONPATH=. python3 scripts/freeze_h9_odss_pairs.py \
  --metadata /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/odss/1968e6d0ef141c4572073695bdc1d17a8706177f/metadata/data/labels.parquet \
  --odss-readme /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/odss/1968e6d0ef141c4572073695bdc1d17a8706177f/metadata/README.md \
  --odss-build-script /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/odss/1968e6d0ef141c4572073695bdc1d17a8706177f/metadata/build_parquet.py \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9odss_source_pairing_001
```

The output directory must be new and below the project HDD root. It contains
`h9_odss_source_trials.csv`, `h9_odss_source_pairs.csv`,
`h9_odss_b2_random_pairs.csv`, `h9_odss_excluded_unmatched.csv`, and a
provenance JSON with byte hashes for compact metadata inputs, ODSS
path-semantics documents, all four H9 protocol documents, deterministic row
hashes, the eligible/excluded ID sets, B2/P edge sets, split counts, and emitted
CSV hashes. It is a preliminary source metadata freeze only: the later source
materialization must append byte-level audio hashes and the mandatory
source--target canonical-audio fingerprint audit before fitting can begin.
