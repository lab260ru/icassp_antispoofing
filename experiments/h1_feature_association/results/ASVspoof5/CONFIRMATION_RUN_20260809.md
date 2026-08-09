# ASVspoof5 H1 confirmation run — 2026-08-09

## Scope and lock

This run used the sole identity declared before target-data analysis in
`../declared_confirmation_candidates_20260809T205526Z/ASVspoof5_candidates.csv`:
`Spectra-AASIST`, `full_waveform`, `crest_factor_db`, spoof (`class_label=1`).
It is a confirmation of that identity only; it neither selects new features nor
establishes the five-corpus portability criterion or causal sensitivity.

## Invocation

```bash
PYTHONPATH=. python3 scripts/analyze_associations.py \
  --dataset ASVspoof5 \
  --bootstrap-candidates \
    experiments/h1_feature_association/results/declared_confirmation_candidates_20260809T205526Z/ASVspoof5_candidates.csv \
  --bootstrap-replicates 2000 --bootstrap-seed 2609
```

## Integrity and outputs

- The locked score-independent feature cohort has 10,000 samples (5,000 per
  label); three views yield 30,000 feature rows.
- Published score rows: 5,446,192 across eight models. Every model has 30,000
  exact stable-ID feature joins across the three views.
- Screen rows: 1,344 unique `(dataset, model, view, feature, class)` cells.
- Candidate bootstrap: 2,000/2,000 valid label-stratified, speaker-cluster
  percentile replicates (367 clusters in the selected spoof slice).

| Artifact | SHA-256 |
|---|---|
| `association_summary.csv` | `d6539ee8194d9d87f28ca46a00716e172fa9cacb98a3f8095b8d35b836be7acb` |
| `association_confirmation_bootstrap.csv` | `ff9d1b9690f1bcebc09765dc0d02e206b19138c7c1255b23ce2ed188b98966b9` |
| `association_join_report.json` | `7387ea61ef41bb79ed94779649ba6250788cc4d3ef86c4050356bff599f1f3a5` |
| `feature_label_metrics.csv` | `694e1af27c3936c30e382c8d18ae1dacd1a5d1c3e4bf3e6684f09f4b4af2b8ce` |

## Manifest-selected result

For the declared Spectra-AASIST spoof/full-waveform/crest-factor test, the
unadjusted Spearman estimate is -0.155568 (`p=1.854067e-28`,
`q=6.077722e-28`) with a 95% clustered interval of [-0.185310, -0.124442].
The prespecified adjusted partial Spearman estimate is 0.001383 (`p=0.925134`,
`q=0.937089`) with a 95% clustered interval of [-0.029092, 0.032219].

The unadjusted association is reported as a descriptive result; it is not
substituted for the registered adjusted estimand. The adjusted estimate is
indistinguishable from zero under the registered controls. The fixed
five-corpus aggregation is required for the formal H1 portability assessment;
no causal claim follows from this result.
