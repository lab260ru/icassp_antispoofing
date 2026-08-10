# H1 covariate-availability reporting audit — run 001

This score-free reporting audit describes metadata availability in the completed H1 feature cohorts. It does not read feature values, scores, models, audio, or ASR; it does not compute an association, bootstrap, candidate, or intervention result.

## Partial-adjustment implementation

`partial_spearman` uses an intercept; ranked duration and integrated loudness with median fill; speaker and attack one-hot dummies with an explicit missing category; and removes a tested feature or response if it would otherwise control itself. The full coverage table and byte-bound source manifest are in the adjacent CSV and JSON.

## Held-out bootstrap context

The already sealed confirmation intervals use 2,000 label-stratified clustered percentile replicates with seed 2609 and speaker ID when nonempty, otherwise source ID. The selected spoof slice has 54 speaker clusters in InTheWild and 367 in ASVspoof5. No bootstrap was rerun for this report.

## Compact coverage

| dataset | label | n_samples | duration_seconds_finite_n | integrated_lufs_finite_n | speaker_id_nonempty_n | speaker_id_unique_n | attack_id_nonempty_n | attack_id_unique_n | source_id_nonempty_n | source_id_unique_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASVspoof2019_LA | 0 | 5000 | 5000 | 5000 | 5000 | 67 | 0 | 0 | 5000 | 5000 |
| ASVspoof2019_LA | 1 | 5000 | 5000 | 5000 | 5000 | 48 | 0 | 0 | 5000 | 5000 |
| ASVspoof2021_LA | 0 | 5000 | 5000 | 4999 | 5000 | 67 | 5000 | 1 | 5000 | 5000 |
| ASVspoof2021_LA | 1 | 5000 | 5000 | 5000 | 5000 | 48 | 5000 | 13 | 5000 | 5000 |
| ASVspoof2021_DF | 0 | 5000 | 5000 | 5000 | 5000 | 93 | 5000 | 1 | 5000 | 5000 |
| ASVspoof2021_DF | 1 | 5000 | 5000 | 5000 | 5000 | 62 | 5000 | 110 | 5000 | 5000 |
| InTheWild | 0 | 19963 | 19963 | 19962 | 19963 | 54 | 0 | 0 | 19963 | 19963 |
| InTheWild | 1 | 11816 | 11816 | 11816 | 11816 | 54 | 0 | 0 | 11816 | 11816 |
| ASVspoof5 | 0 | 5000 | 5000 | 5000 | 5000 | 737 | 5000 | 1 | 5000 | 5000 |
| ASVspoof5 | 1 | 5000 | 5000 | 5000 | 5000 | 367 | 5000 | 16 | 5000 | 5000 |

All paths, input hashes, schemas, and prohibited operations are recorded in `h1_covariate_availability_provenance.json`.
