# Discovery-to-H2 candidate freeze rule

**Status:** locked on 2026-08-09 before ASVspoof2021_DF H1 output exists and
before either held-out confirmation corpus is downloaded or read for analysis.

## Purpose and scope

This is an explicit, exploratory follow-up rule for choosing a small set of
waveform-feature *families* for H2 paired interventions after all three H1
discovery screens complete. It is not the five-corpus H1 portable-association
criterion, does not use confirmation data, and cannot turn an H2 result into a
confirmatory causal claim by itself.

The rule is deliberately limited to the fixed slice `full_waveform`, spoof
class (`class_label=1`), and the `partial_spearman` screen estimator. This
aligns H2 with the actual waveform that will be transformed and avoids choosing
a crop-only effect for a full-waveform intervention.

## Fixed candidate feature families and arms

| Feature | H2 transformation family | Targeted diagnostic |
|---|---|---|
| `crest_factor_db` | `drc_cf3`, `drc_cf6` | achieved crest-factor reduction |
| `silence_fraction` | endpoint-silence standardization | endpoint/silence change and detector-window position |
| `spectral_slope_db_per_khz` | spectral-tilt positive/negative arms | spectral-slope and low/high-energy-ratio change |
| `group_delay_var` | stable all-pass phase arms | phase-feature change with magnitude-spectrum tolerance |

The implementation may reject a specific pair or arm through the already
locked STOI, WER, loudness, clipping, feature-change, and retention-rate gates.
An accepted family therefore does not imply every listed pair is eligible.

## Eligibility calculation after the third discovery screen

Read exactly the three explicit discovery `association_summary.csv` files and
no other H1 table. For each feature/model cell in the fixed slice:

1. The row must be finite and have BH-adjusted
   `partial_spearman_q <= 0.05` in **all three** discovery datasets.
2. The three partial-Spearman signs must be identical and nonzero.
3. The absolute partial-Spearman effect must be at least `0.05` in **all
   three** discovery datasets.

A feature family is frozen for H2 if at least one of the two
parity-validated runnable models (`Spectra-AASIST` or `AASIST`) satisfies all
three conditions. Every satisfying model--feature--dataset H1 cell is written
to the provenance-bearing H1 bootstrap manifest; no strongest-row or top-$k$
selection is allowed. The H2 runner nevertheless scores every available
validated H2 model for each frozen feature family; it does not drop a model
because its discovery direction differs.

If no family qualifies, H2 is not run as a causal follow-up. The paper reports
the negative discovery result and retains only the pre-intervention engineering
artifacts. If a family qualifies but fewer than three transformed scorers later
pass parity, H2 remains a two-model feasibility result and cannot meet the
registered multi-model causal criterion.

## Freeze artifact requirements

The generated manifest must use the exact schema in `BOOTSTRAP_CLI.md`, have
`selection_status=frozen`, record `selection_split=discovery`, name this file
as `selection_basis`, contain a UTC freeze timestamp, and be SHA-256 hashed by
the bootstrap command. The table must be committed before launching any
bootstrap or reading confirmation score-feature tables.

## Disclosure

The general staged protocol was registered before result generation. This
specific H2 follow-up rule was added after the first two discovery screens and
before the final discovery screen; it is consequently documented as an
outer-loop, exploratory choice rather than presented as pre-registered.
