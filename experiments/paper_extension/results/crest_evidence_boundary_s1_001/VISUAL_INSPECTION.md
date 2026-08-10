# Visual inspection — Supplementary Fig. S1

**Artifact inspected:** `s1_crest_evidence_boundary.png`, original raster
dimensions 2207 × 935 pixels, embedded 300-DPI resolution. The companion PDF
is vector output from the same Matplotlib draw call and its SHA-256 is recorded
in `s1_crest_evidence_boundary.metadata.json`.

## Final inspection outcome

**Pass.** The original-resolution raster was inspected locally after the final
deterministic render. All three panel titles, five corpus labels, axes, point
estimates, held-out confidence intervals, four H2 arm labels, percentages,
locked 90% gate, and descriptive-only boundary text are legible. The legend is
unnecessary because each panel has direct labels. Zero reference lines are
visible in panels A and C, the fixed 90% gate is visually distinct in panel B,
and the H4 InTheWild sign reversal is visible without any supplementary
interpretation.

The only formatting correction made after an initial readability check was to
move the **locked 90% gate** annotation above panel B and remove the redundant
90 tick label, which had collided with the 100 tick label. The encoded values,
panel order, colors, source set, axis limits, confidence intervals, and every
data scale were unchanged. No source was substituted and no numerical or
statistical quantity was recomputed after inspection.

## Boundary check

- Panel A says `Fixed H1 association` and states that only held-out
  confirmations have intervals; it does not say portable or causal.
- Panel B states `Detector-free quality screening; scoring unavailable (panel
  not frozen)`; it does not contain detector scores or score deltas.
- Panel C states `Score-free raw label separation; not detector reliance or
  cue selection`; it does not promote H4 as a candidate.
- The caption and metadata declare no pooled estimate, new test, ranking,
  selection, causal claim, or cross-panel inference.

No paper source was changed by this supplementary-only artifact.
