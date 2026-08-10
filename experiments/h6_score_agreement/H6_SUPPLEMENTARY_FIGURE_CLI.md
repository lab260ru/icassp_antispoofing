# H6 supplementary figure CLI

Run this only after the implementation protocol and renderer are committed.
It is a display-only operation over the three completed compact H6 artifacts;
the renderer validates their literal paths, hashes, schemas, internal summary
identities, and completed 280-cell registry before creating an output.

```bash
PYTHONPATH=. python3 scripts/plot_h6_score_agreement_heatmap.py \
  --matrix /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001/h6_within_class_score_agreement_matrix.csv \
  --aggregation /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001/h6_model_pair_agreement_summary.csv \
  --provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001/h6_analysis_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001/figures_within_class_agreement_v1
```

The output directory must not already exist. Successful output is one vector
PDF, one 300-DPI PNG, and one metadata JSON. The metadata records all input
and output byte identities and the fixed no-ranking/no-selection display plan.

This command does not evaluate an anti-spoofing model, train a model, access a
waveform, or emit a raw model response. A successful display must still be
described only as within-class rank agreement among the fixed published score
artifacts.
