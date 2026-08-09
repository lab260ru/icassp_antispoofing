# H1 cross-dataset aggregation CLI

`scripts/aggregate_h1_associations.py` makes an exhaustive, descriptive audit
from only the association CSV files named with `--input`. It does **not** find
files automatically, fit another model, select/freeze candidates, or impute
absent model/dataset cells.

Use it only after each supplied table was generated under the locked H1
protocol. A confirmation table is read only when its path is explicitly named
on the command line.

```bash
PYTHONPATH=. python3 scripts/aggregate_h1_associations.py \
  --input /path/to/asv2019/association_summary.csv \
  --input /path/to/asv2021-la/association_summary.csv \
  --input /path/to/asv2021-df/association_summary.csv \
  --output-dir /path/to/new_aggregation_report
```

This discovery-only invocation writes an explicit
`not_evaluable_requires_all_5_core_datasets` status. It cannot assert that any
feature is portable. After the two held-out results are intentionally supplied,
repeat with five explicit inputs:

```bash
PYTHONPATH=. python3 scripts/aggregate_h1_associations.py \
  --input /path/to/asv2019/association_summary.csv \
  --input /path/to/asv2021-la/association_summary.csv \
  --input /path/to/asv2021-df/association_summary.csv \
  --input /path/to/inthewild/association_summary.csv \
  --input /path/to/asvspoof5/association_summary.csv \
  --output-dir /path/to/new_aggregation_report
```

The default estimator is `partial_spearman` with its existing BH-adjusted
`partial_spearman_q` values at `alpha=0.05`. `--statistic spearman` evaluates
the corresponding raw-rank columns instead; it is a separate report, not a
license to choose the more favorable estimator.

Outputs are newly written in the requested directory:

- `portable_association_model_report.csv`: all configured model × feature ×
  view × class units, including explicit missing-core cells.
- `portable_association_feature_report.csv`: feature/view/class counts across
  all eight configured models. `final_portable_criterion_met` is always false
  unless all five configured core datasets were explicitly supplied.
- `portable_association_aggregation_report.json`: scope, estimator, criterion,
  input coverage, and a `selection_or_freezing_performed: false` guard.

For a model to count toward the configured `>=5/8` condition, it needs a finite
estimate for all five core datasets, one common direction in at least four of
them, and BH-adjusted significance (`q <= alpha`) with that direction on both
confirmation datasets. No missing cell can count as a fourth or fifth corpus.

The H1 analyzer itself now keeps generated source tables corpus-scoped:
`results/<dataset>/association_summary.csv` for a single `--dataset`, and
`results/combined/association_summary.csv` only when multiple datasets are
explicitly supplied together. Pass those per-dataset source paths explicitly to
this aggregation tool; never rely on a mutable flat result filename.
