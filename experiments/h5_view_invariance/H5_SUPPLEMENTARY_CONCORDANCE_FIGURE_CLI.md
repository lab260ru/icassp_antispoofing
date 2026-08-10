# H5 supplementary median-concordance heatmap CLI

Use this command only after a complete `h5_analysis_001` result and its
provenance have been committed. The renderer accepts no raw H5 input, model,
audio, label, or response path. It accepts exactly the two sealed compact CSVs
and derives their sibling provenance for hash validation.

```bash
PYTHONPATH=. python3 scripts/plot_h5_view_invariance_heatmap.py \
  --matrix /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/h5_view_invariance_matrix.csv \
  --aggregation /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/h5_view_invariance_aggregation.csv \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/figures_median_concordance
```

The input names, parent run directory, hashes, complete 420/84 matrix, view
pair order, feature registry, and exact CSV headers are all checked before any
figure value is read. The destination must not already exist. The output is a
three-panel, 28-row-per-panel supplementary heatmap of the sealed five-corpus
median Spearman concordance using a fixed `[0, 1]` color scale, explicit gray
`×` unavailable cells, and green terminal-descriptive markers.

This visualization is not a feature ranking or selection mechanism. Do not use
it to reopen H1/H2/H2B/H3/H4, change H5, train a model, or make a detector or
causal claim.
