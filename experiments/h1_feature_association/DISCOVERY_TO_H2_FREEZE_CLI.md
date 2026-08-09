# Discovery-to-H2 freeze CLI

`scripts/freeze_discovery_to_h2_candidates.py` is the deterministic executor
for the locked exploratory rule in
`DISCOVERY_TO_H2_FREEZE.md`. It reads exactly three explicit discovery H1
`association_summary.csv` paths and no confirmation or H2 artifact.

```bash
PYTHONPATH=. python3 scripts/freeze_discovery_to_h2_candidates.py \
  --asvspoof2019-la /path/to/ASVspoof2019_LA/association_summary.csv \
  --asvspoof2021-la /path/to/ASVspoof2021_LA/association_summary.csv \
  --asvspoof2021-df /path/to/ASVspoof2021_DF/association_summary.csv \
  --frozen-at-utc 2026-08-10T12:00:00Z \
  --output-manifest /path/to/frozen_discovery_to_h2_candidates.csv \
  --selection-report /path/to/frozen_discovery_to_h2_candidates.report.json
```

The timestamp must be explicit UTC (`Z`). Both output paths must be new; the
command refuses to overwrite a freeze artifact. The manifest has exactly the
bootstrap schema from `BOOTSTRAP_CLI.md`:

```text
dataset,model,view,feature,class_label,selection_status,selection_split,selection_basis,frozen_at_utc
```

For each fixed feature family and only the two parity-validated runners
(`Spectra-AASIST`, `AASIST`), it requires three finite full-waveform spoof
partial-Spearman cells, BH q <= 0.05 in every discovery corpus, identical
nonzero signs, and |rho| >= 0.05 in every corpus. There is no top-k choice or
strongest-row selection. A qualifying runner--feature pair contributes all
three source cells to the manifest.

The JSON report records source/config/rule hashes, all 8 fixed eligibility
cells, thresholds, and a guard that no confirmation or H2 data was read. This
freeze is operational and exploratory only; it is not a portable-association,
causal, or confirmation claim. Do not execute it until the third discovery H1
screen is complete and commit its generated artifacts before any bootstrap or
H2 launch.
