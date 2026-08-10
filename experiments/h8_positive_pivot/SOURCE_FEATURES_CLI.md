# H8-SF source rank/probit feature materialization

After the v2 source freeze, create the revalidated source training panel:

```bash
PYTHONPATH=. python3 scripts/materialize_h8_source_features.py \
  --source-manifest /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_freeze_002/source_manifest.csv \
  --source-orientation /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_freeze_002/source_orientation.json \
  --source-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_freeze_002/source_freeze_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_features_001
```

The materializer re-hashes the index, source manifest, orientation record, and
every source raw-score input. It recomputes ranks on each complete source score
batch before applying the selected IDs. Its output contains source labels only
for the frozen rows; it has no target-data read path.
