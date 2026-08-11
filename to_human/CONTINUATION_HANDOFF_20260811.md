# Continuation handoff — 2026-08-11

## Readable paper

The current compiled working draft is `paper/build/main.pdf`, sourced from
`paper/main.tex`. It reports only H9: fresh 172k-parameter Res2TCNGuard trained
on a sealed ODSS same-item pool. Its four-seed ensemble reaches 45.02% macro
EER on the predeclared SONAR/ArAD panel, versus 51.78% for BCE and 52.01% for
an equal-edge random-pair ranking control. The locked paired bootstrap
intervals for P--B1 and P--B2 are below zero. The claim is deliberately narrow:
it is neither SOTA nor proof of a semantic/causal representation.

`paper/main.tex` has not been changed in this handoff. The last paper revision
was compiled immediately and passed the local five-page US-letter, landmarks,
and embedded-font preflight. Never edit `main.tex` without recompiling its PDF
in the same change.

The 2026-08-11 handoff rebuild writes PDF SHA-256
`80ee1f1ee6ce87d3a6e8cff65b48ae8718c50b22c341ab31395472b664ca6eca`.
The full repository suite passed **200 tests** in 87.06 seconds; the local PDF
preflight again confirms five US-letter pages, Table 1 on page 2, references
on page 5, and 13 embedded fonts. Tectonic emitted only non-fatal underfull
box/bibliography-rerun warnings.

## Closed follow-up

H10 applied the exact frozen H9 checkpoint panel to fresh CD-ADD. P was lower
than both controls and both intervals were below zero, but its 8.30% relative
reduction versus BCE missed the H10 protocol's fixed 10% practical threshold.
The independent red-team replay found no defect. H10 is a terminal no-retune
result and must not be added to the paper claim or used to select another
target.

## Strongest prospective study

H11 is the **Counterpart Specificity Ladder**:

1. Use one new relation-complete ODSS source pool for BCE, stratified random
   pairs, same-voice/different-item pairs, and documented same-item pairs.
2. Use fresh AASIST and RawNet2 implementations with source-only selection,
   fresh BF16 fits, fixed seeds, and all controls at the same edge budget.
3. Evaluate every frozen cell on a fresh all-trial terminal panel with a
   two-pass label firewall. The desired conclusion requires same-item ranking
   to beat every comparator in both architectures and both targets; it still
   may not be called semantic or causal proof.

The prospective details and hard stops are in:

- `experiments/future_directions/H11_IDEA_AUDIT_20260811.md`
- `experiments/future_directions/H11_TARGET_PANEL_AUDIT_20260811.md`

The recommended panel is HABLA plus J-SPAW_LA. J-SPAW_LA is explicitly for
non-commercial research only. Await the project owner's confirmation that the
study and any release meet those terms (or rights-holder permission) before
committing an H11 protocol, downloading data, or accessing any target content.
If confirmation is unavailable, H11 remains a future direction rather than a
target-shopping exercise.

## Resume sequence

1. Read `AGENTS.md`, then the two H11 audits above.
2. Obtain and record the licence decision before running any H11 command.
3. If authorized, first commit a new H11 pre-data protocol and synthetic-only
   architecture/preprocessing qualification; do not reuse H9/H10 targets,
   checkpoints, or model choices.
4. Keep datasets, caches, checkpoints, raw predictions, and logs on the HDD;
   commit only code, small manifests, hashes, notes, and compiled paper
   artifacts.
