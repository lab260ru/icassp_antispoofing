# Discovery-only crest-factor comparison plot

`scripts/plot_h1_discovery_crest_factor.py` makes a reproducible grouped-bar
comparison for the pre-specified H1 slice only: `crest_factor_db`,
`full_waveform`, spoof class (`class_label=1`), and partial Spearman association
with canonical `score_spoof`. It includes all eight models from
`configs/study.yaml` and the three configured discovery corpora.

It reads exactly the three paths passed to it. Each CSV must contain one and
only its named dataset, a complete score panel in the fixed slice, no duplicate
H1 key, and no unconfigured model. The script rejects a reused path, an absent
input, an unknown dataset/model, or a missing panel cell. It does not search for
files and cannot read InTheWild, ASVspoof5, or any fallback corpus.

```bash
PYTHONPATH=. python3 scripts/plot_h1_discovery_crest_factor.py \
  --asvspoof2019-la /path/to/ASVspoof2019_LA/association_summary.csv \
  --asvspoof2021-la /path/to/ASVspoof2021_LA/association_summary.csv \
  --asvspoof2021-df /path/to/ASVspoof2021_DF/association_summary.csv \
  --output-dir /path/to/new_discovery_plot
```

It writes a vector PDF, 300-DPI PNG, and a JSON sidecar with hashes of the three
explicit source tables. The figure and sidecar state that the result is
**discovery-only and non-portable**: it is not a causal claim, a confirmation
analysis, or a candidate-selection/freeze mechanism. Do not run the command
until all three complete discovery CSVs are available.
