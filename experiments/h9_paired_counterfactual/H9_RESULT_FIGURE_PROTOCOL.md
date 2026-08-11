# H9-PCR terminal-result figure protocol

**Status:** fixed after the one terminal H9 evaluation completed; display-only
and incapable of changing the H9 decision gate.

## Purpose

Render one compact, publication-quality figure that reports the sealed
three-method comparison on the two predeclared external targets and the two
predeclared paired-bootstrap contrasts. The figure is an explanation of the
terminal result, not an additional analysis, selection rule, or new
experiment.

## Exact inputs

The renderer may read only these four terminal artifacts, all from
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_terminal_evaluation_001/`:

| Artifact | SHA-256 | Allowed fields/use |
| --- | --- | --- |
| `h9_terminal_target_metrics.csv` | `2e744154d25b3f515d1b0121db7dc8658cd9b4fb5ff4a9a45dca79b1d8f35ee4` | six terminal EER values only |
| `h9_terminal_bootstrap_macro_eer_differences.csv` | `771a4f3b6b794503b725e447dcd655cde65a3a2dbdedc851ee55fa71c938e9c0` | fixed 2,000-replicate contrast interval validation only |
| `h9_terminal_decision_gate.json` | `a4e5f118cd34af6ae2e728e094d2ad9e9e6ebf1a8ebe1fb2d841b191c3753852` | terminal macro EER and frozen contrast intervals |
| `h9_terminal_evaluation_provenance.json` | `af0b85c8f05ad9f2b70cad83b45db7922cd9422732c4630dbb17d1ad2821d2f4` | version, artifact linkage, BF16/device, and zero-collision audit |

It must not read raw predictions, target labels, source-development metrics,
checkpoints, audio, or a score artifact.

## Locked rendering

- Panel A: grouped bars of terminal EER (%) for the fixed order `SONAR`,
  `ArAD`, and unweighted two-target macro EER; methods are B1 (BCE), B2
  (random-pair ranking), and P (same-item ranking).
- Panel B: the two fixed two-sided 95% percentile intervals for the shared-ID
  bootstrap macro-EER differences P--B1 and P--B2, in percentage points. The
  vertical zero reference is descriptive; there are no newly calculated p
  values or significance markers.
- Palette: Okabe--Ito blue/orange/green, with P in green; PDF is vector and
  PNG is 300 DPI. The plot carries a clear lower-EER-is-better label and the
  four-seed probability-ensemble / 2,000-replicate context.
- The renderer rejects input hash/schema/cardinality/provenance drift and
  refuses to overwrite its output directory. It writes PDF, PNG, and a JSON
  metadata record with all input/output hashes. Rendering revision 003 changes
  only reader-facing terminology from ``content-aligned'' to ``same-item'' and
  neutralizes the contrast-panel title; it uses the exact same four sealed
  inputs and makes no numerical or analytical change.

## Claim boundary

The display may report the locked terminal result only: under this fresh-init
compact Res2TCNGuard/ODSS protocol, same-item ranking had lower EER than both
same-pool controls on both fixed external targets. It cannot claim
causality, state-of-the-art performance, generalization beyond these targets,
or that a representation discarded any particular shortcut.
