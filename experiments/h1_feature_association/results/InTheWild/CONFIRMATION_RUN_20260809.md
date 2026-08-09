# InTheWild H1 confirmation run — 2026-08-09

## Scope and lock

This run used the sole identity declared before target-data analysis in
`../declared_confirmation_candidates_20260809T204004Z/InTheWild_candidates.csv`:
`Spectra-AASIST`, `full_waveform`, `crest_factor_db`, spoof (`class_label=1`).
It is a confirmation of that identity only; it neither selects new features nor
establishes the five-corpus portability criterion or causal sensitivity.

## Invocation

```bash
PYTHONPATH=. python3 scripts/analyze_associations.py \
  --dataset InTheWild \
  --bootstrap-candidates \
    experiments/h1_feature_association/results/declared_confirmation_candidates_20260809T204004Z/InTheWild_candidates.csv \
  --bootstrap-replicates 2000 --bootstrap-seed 2609
```

The finite-value-safe AUROC diagnostic implementation was committed before the
run in `bdebb8b`. The source feature audit is recorded in
`../../IN_THE_WILD_H1_FEATURE_FINITE_VALUE_AUDIT_20260809.md`; it did not
modify the source parquet.

## Integrity and outputs

- Feature samples: 31,779; three views yield 95,337 feature rows.
- Published score rows: 254,232 (eight models); every model has 95,337
  exact stable-ID feature joins across views.
- Screen rows: 1,344 unique `(dataset, model, view, feature, class)` cells.
- Candidate bootstrap: 2,000/2,000 valid label-stratified, speaker-cluster
  percentile replicates (54 clusters in the selected spoof slice).

| Artifact | SHA-256 |
|---|---|
| `association_summary.csv` | `fca2212ce2ee026f632e56d2f4386a98b844519a1310e10f0db076d19a1b89e4` |
| `association_confirmation_bootstrap.csv` | `d45ef60da53e7a35ba19a6a7e91595ed3a1637abccc7cfce0d48e029c64dd825` |
| `association_join_report.json` | `b8e18968eb12ce3fb9e510671e1ab35d036ea480395b54e6984d5ea27a7c73db` |
| `feature_label_metrics.csv` | `4cb72f4936cc98ebd553b9e18e9df60bf6700fe3fda5cd283b7aba161e850d42` |

## Manifest-selected result

For the declared Spectra-AASIST spoof/full-waveform/crest-factor test, the
unadjusted Spearman estimate is 0.003753 (`p=0.683305`, `q=0.696256`) with a
95% clustered interval of [-0.132051, 0.139498]. The prespecified adjusted
partial Spearman estimate is -0.032606 (`p=0.000406`, `q=0.000523`) with a
95% clustered interval of [-0.070045, -0.001718].

The discrepancy between unadjusted and adjusted estimates is reported rather
than reconciled post hoc. This is a small, held-out association under the
registered control model. The ASVspoof5 confirmation and the fixed five-corpus
aggregation remain required before any portability assessment.
