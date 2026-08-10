# H8-SF final target evaluation

This is the first and only H8 command that reads target labels. Run it only
after the committed evaluator and its tests pass:

```bash
PYTHONPATH=. python3 scripts/evaluate_h8_targets.py \
  --target-predictions /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_only_fit_001/target_predictions.parquet \
  --fit-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_only_fit_001/source_only_fit_provenance.json \
  --model-record /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_only_fit_001/source_fuser_models.json \
  --target-feature-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_target_features_001/target_feature_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_target_evaluation_001
```

The evaluator first checks every derived-input/model/prediction hash and proves
that the fit consumed no target labels or metrics. It then validates the pinned
target label hashes and one-to-one joins, reports every target × method EER and
AUROC, freezes the predetermined uncertainty panel, runs exactly 2,000
stratified sample-ID bootstrap replicates, and applies the five-part P gate
without a model or target-selection branch.
