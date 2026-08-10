# H9-PCR source freeze 001

**Status:** complete source-only metadata freeze; no H9 training, source-audio
decode, target access, score, or target metric occurred in this stage.

## Immutable output

HDD directory:
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9odss_source_pairing_001/`

| File | SHA-256 |
| --- | --- |
| `h9_odss_source_freeze.json` | `289675d647ac379e8eabe0e56431cc4fec488d769362abb6c2e667aafadb2f93` |
| `h9_odss_source_trials.csv` | `a953a6476dc7b623fc1f27e969fa8401a53da9426ad4873dfd93c1fd33ab588a` |
| `h9_odss_source_pairs.csv` | `d9df508a5cbce8484742208e92b744e89178baf5e58cce99f42aaa257bb336bb` |
| `h9_odss_b2_random_pairs.csv` | `cced5cc79a60a5b08e7f7cadb57d6ab9f0e0307ad08506cbe2517fc29a7b1957` |
| `h9_odss_excluded_unmatched.csv` | `390c2a55eb1ef144a5d6c36982a80ab4b5e27cd83a12fdf57ecb6b8af92638af` |

The freeze binds ODSS revision
`1968e6d0ef141c4572073695bdc1d17a8706177f`, the input label metadata
(`1e4466cce807222807fffba83709b6250e0defa8431a441e2bbc6aec15d46162`),
and the pinned README/build-script semantics documents.

## Eligibility outcome

- Input metadata: 26,954 rows.
- Retained paired-only pool: 23,883 trials from 7,961 complete groups.
- P matched-pair edges: 15,922 (one natural-to-VITS and one
  natural-to-FastPitch--HiFi-GAN edge per complete group).
- B2 random-pair edges: 15,922. The mapping is frozen and stratified by
  split, language, source corpus, and spoof generator.
- Excluded unmatched source trials: 3,071, all `missing_bonafide` VITS-only
  rows. They are absent from B1, B2, and P.
- Source split: seed `2909`, voice-disjoint and language-stratified; 6,051
  complete groups in train and 1,910 in development. English, German, and
  Spanish are each represented in both splits.

This passes H9's source-pairing eligibility prerequisite. It is not a detector
result or a claim about cross-corpus performance. The next permitted stage is
the fresh-initialized BF16 source-training harness after its synthetic tests and
throughput contract are committed.
