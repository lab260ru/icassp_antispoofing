# H1 bootstrap confirmation CLI

`analyze_associations.py` always writes the full **screen** to
`association_summary.csv`. It never promotes a feature to confirmation on the
basis of that screen. To request confidence intervals, pass an explicit frozen
CSV or Parquet manifest with exactly these required provenance fields:

```text
dataset,model,view,feature,class_label,selection_status,selection_split,selection_basis,frozen_at_utc
```

Every row must have `selection_status=frozen`; `selection_split` cannot be a
confirmation/test/evaluation split. The manifest is checked against the exact
screened rows, hashed into the output, and is the only source of bootstrap
candidates. A confirmation run uses 2,000 deterministic percentile replicates
by default, sampling speaker clusters where present and otherwise source
utterances, stratified by label.

```bash
rtk python3 scripts/analyze_associations.py \
  --dataset ASVspoof2019_LA \
  --bootstrap-candidates /absolute/path/to/frozen_candidates.csv \
  --bootstrap-replicates 2000 --bootstrap-seed 2609
```

The manifest-selected output is
`association_confirmation_bootstrap.csv`; its valid and invalid replicate
counts are reported separately for raw and partial Spearman estimates.
