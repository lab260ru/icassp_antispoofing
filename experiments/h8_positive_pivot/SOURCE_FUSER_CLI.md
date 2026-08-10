# H8-SF source-only fuser fit

The following command trains only on the sealed three-source feature panel and
writes target predictions without opening any target label or computing any
target metric:

```bash
PYTHONPATH=. python3 scripts/fit_h8_source_only_fusers.py \
  --source-features /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_features_001/source_rank_probit_features.parquet \
  --source-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_features_001/source_feature_provenance.json \
  --target-features /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_target_features_001/target_rank_probit_features.parquet \
  --target-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_target_features_001/target_feature_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_only_fit_001
```

The runner verifies both derived-input hashes and contracts. It fits B2, the
primary GroupDRO fuser P, and V-REx A1 with the fixed optimizer/seeds and
source-only inner leave-one-source-corpus-out selection. B0 is selected solely
by source corpus-macro EER; B1 is the untrained equal rank mean. It records all
source selection tables, seeds, fitted nonnegative weights, source predictions,
and **label-free** target predictions. It cannot accept target labels as an
argument.
