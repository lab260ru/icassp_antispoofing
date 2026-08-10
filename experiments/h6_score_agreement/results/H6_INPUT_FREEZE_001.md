# H6 input freeze 001 — label-only published-score agreement panel

Status: complete label-only freeze. No raw score text, waveform, feature,
model, ASR, detector runner, or prior-result table was read by this stage.

The committed H6 freeze loaded only the exact pinned label projection, selected
at most 5,000 normalized sample IDs per `(dataset, label)` with seed 2611, and
then sealed byte identities for the 40 original published score files. File
hashing establishes immutable provenance but does not parse, rank, join, or
materialize score values.

| Dataset | Selected label 0 | Selected label 1 | Label rows | Bootstrap cluster |
|---|---:|---:|---:|---|
| ASVspoof2019_LA | 5,000 | 5,000 | 71,237 | stable sample ID |
| ASVspoof2021_LA | 5,000 | 5,000 | 181,566 | stable sample ID |
| ASVspoof2021_DF | 5,000 | 5,000 | 611,829 | stable sample ID |
| InTheWild | 5,000 | 5,000 | 31,779 | stable sample ID |
| ASVspoof5 | 5,000 | 5,000 | 680,774 | stable sample ID |

All five label inputs and all 40 score-file byte identities revalidated against
the locked Arena index; `freeze_status` is `complete` with no failed dataset.
The score-value firewall is intact (`raw_score_values_read_during_freeze:
false`). A frozen-manifest-only analyzer is now the sole authorized next step.

## HDD ledger

Artifacts remain under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_input_freeze_001/`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `h6_label_selection_manifest.parquet` | 4,042,772 | `159351198de98f4a3821495a3e40012a2cd152e6a2d7f796c36f910a6b907c39` |
| `h6_input_freeze_provenance.json` | — | `9ebc7ade28fac40769f745249b037c761d0b3cdf4c7e781f300a4e0d2e2cc4da` |

The provenance also binds protocol SHA-256
`755678790d3f87cb5123f87f200d4174aa8c7121e06f4a642315f18cf553f1bc`,
Arena-index SHA-256
`4c0b5e905cc9aeb96d179f21dd0c6efcbc5f5f894ea773540a89c497a03ede2a`,
the seed, cap, and 200-resample H6 configuration. Do not alter this freeze,
replace a score artifact, or use a prior score panel; an integrity failure must
be recorded rather than repaired.
