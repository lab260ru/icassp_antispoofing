# H8-SF label-free target feature materialization

After the committed source freeze and acquisition of the pinned target score
files, create the only target input representation permitted to training:

```bash
PYTHONPATH=. python3 scripts/materialize_h8_target_features.py \
  --source-orientation /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_freeze_001/source_orientation.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_target_features_001
```

The command reads each target's eight raw score files and the frozen
source-derived orientation multipliers. It calculates the empirical rank/probit
coordinate within the full target score batch, then retains only exact
eight-model common IDs. It never opens a dataset label table, evaluates a
metric, fits a fuser, or writes raw scores. Its immutable JSON ledger records
all 40 raw-score hashes, coverage counts, the source-orientation hash, and the
derived-Parquet hash.
