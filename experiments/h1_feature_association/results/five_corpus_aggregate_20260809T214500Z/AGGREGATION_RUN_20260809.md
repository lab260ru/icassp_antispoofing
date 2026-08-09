# Five-corpus H1 aggregation — 2026-08-09

## Fixed evaluation

This is the preconfigured, descriptive five-corpus H1 evaluation. It reads the
five explicit `association_summary.csv` inputs named in the invocation below,
uses the registered `partial_spearman` / BH-`q` columns at `alpha=0.05`, and
does not select, rank, or freeze a feature.

```bash
PYTHONPATH=. python3 scripts/aggregate_h1_associations.py \
  --input experiments/h1_feature_association/results/ASVspoof2019_LA/association_summary.csv \
  --input experiments/h1_feature_association/results/ASVspoof2021_LA/association_summary.csv \
  --input experiments/h1_feature_association/results/ASVspoof2021_DF/association_summary.csv \
  --input experiments/h1_feature_association/results/InTheWild/association_summary.csv \
  --input experiments/h1_feature_association/results/ASVspoof5/association_summary.csv \
  --output-dir experiments/h1_feature_association/results/five_corpus_aggregate_20260809T214500Z
```

All five configured corpora and all eight configured score artifacts are
present. The output evaluates 168 registered `(view, feature, class)` units
and 1,344 model/unit rows under the rule: common direction in at least four of
five corpora, BH `q <= 0.05` in both held-out corpora with that direction, and
at least five of eight models.

## Primary frozen-candidate disposition

The exploratory H2 candidate is
Spectra-AASIST / full-waveform / spoof / `crest_factor_db`. Its adjusted
association has a negative direction in four of five corpora, but it fails both
held-out-significance and held-out-direction requirements: InTheWild is small
and negative (`rho=-0.032606`, `q=0.000523`) while ASVspoof5 is effectively
zero and positive (`rho=0.001383`, `q=0.937089`). It therefore **does not meet
the configured portable-association subcriterion**.

The already-frozen H2 crest-factor run remains an explicitly exploratory
follow-up from the three-discovery operational freeze. The five-corpus result
does not license adding any of the other units below to H2 after looking at
confirmation outcomes, and it does not provide causal evidence.

## Registry-wide descriptive result

Nineteen of 168 pre-registered units meet the fixed five-corpus association
criterion, with five to seven qualifying models each. Most are bona-fide-class
energy or spectral descriptors, distributed across all three waveform views.
This registry-wide count is an exhaustive screen result, not a new intervention
candidate list; individual bootstrap intervals and causal transformations would
need their own predeclared follow-up.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `portable_association_model_report.csv` | `6ba7ec2ffa63aab2d076770f1e5b8cef3105ed1c310d7d978a239db396a27321` |
| `portable_association_feature_report.csv` | `153a08438ee08c6ffb52159569dd33a593314fd78f6bc01aec89da9d70c3d356` |
| `portable_association_aggregation_report.json` | `91d3101a66b6046dda7d894d12aeb8eafb610f5f3ab48998c303cdd2f9872e21` |
