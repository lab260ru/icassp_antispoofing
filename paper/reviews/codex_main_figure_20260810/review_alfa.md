# Methods and claim-integrity review — Fig. 1 substitution

**Reviewer:** alfa (Codex-only internal review)
**Scope:** methodological and claim-integrity audit of the proposed main-paper
Fig. 1 replacement only. This is an internal review, not external peer review.
It evaluates the current `paper/build/main.pdf`, `paper/main.tex`, the
main-figure protocol/manifest/metadata, and the sealed H1/H2 compact result
notes. It does not review H4--H6, H2B, S1, novelty, or submission policy.

## Summary

The proposed two-panel figure is methodologically appropriate for the narrow
paper narrative. Its inputs are exactly the sealed five H1 association
summaries, the two discovery-frozen held-out bootstrap records, and the one
detector-free H2 quality summary. The manifest, protocol, metadata, and live
source-file SHA-256 values agree. The renderer's focused synthetic suite passes
14/14 tests, including source-substitution, forbidden-source, exact-slice,
no-detector, and authorization refusal checks.

The plotted H1 values reproduce the sealed fixed slice in the declared corpus
order: -0.4538848, -0.0991428, -0.0649497, -0.0326060, and 0.0013835. The two
displayed confirmation intervals also reproduce the sealed records
(InTheWild [-0.070045, -0.001718]; ASVspoof5 [-0.029092, 0.032219]). Panel B
reproduces the detector-free retention fractions 18.1%, 0.6%, 64.6%, and
99.9% for the four fixed arms. These are faithful display copies, not a new
test, pooled estimate, or bootstrap.

The caption and manuscript correctly avoid a causal feature-reliance claim:
they state that no detector scoring, paired H2 delta, EER, or causal inference
was produced. The input contract excludes H4, H5, H6, H2B, S1, raw score
artifacts, and any training evidence; thus the replacement preserves an H1/H2
scope rather than expanding the paper. No methodological evidence supports
interpreting the side-by-side panels as a cross-panel relationship, and the
caption expressly excludes such an inference.

## Major concerns

None that invalidate the figure's internal-draft use. The following
clarifications should nevertheless be made before external circulation because
they prevent a precise, avoidable ambiguity about the H2 gate.

1. **"Panel was not frozen" is underspecified and can conflict with the
   manuscript's use of "frozen H2 crest/control quality panel."** The arm set
   and score-blind input panel were frozen and executed; what the quality gate
   refused to freeze was a *score-eligible retained-pair manifest*. The sealed
   H2 summary's `panel_gate_status: not_frozen` has that latter operational
   meaning. A reader could otherwise mistakenly think that the design itself
   was post hoc or that the quality run never occurred.

   **Required wording:** replace the figure caption sentence
   "Three arms fall below the gate, so the panel was not frozen and no detector
   scoring was performed" with
   "Three arms fall below the gate, so no score-eligible retained-pair manifest
   was frozen and no detector scoring was performed."

   The Panel B annotation should likewise read:
   "No score-eligible manifest; detector scoring unavailable."

2. **The graphical green polarity bar weakly conflicts with the protocol's
   no-winner/highlight rule.** It accurately distinguishes the single
   individually passing control from the three failures, but color alone can
   be read as promoting polarity. Since the final conclusion depends on the
   *panel* gate and not a winning arm, use one neutral/color-blind-safe bar
   color for all four arms and let the dashed 90% line plus the numeric labels
   convey the gate outcome. If the green encoding is retained, add an explicit
   legend or caption phrase: "Polarity individually clears the arm gate but
   does not open the panel."

## Minor concerns

1. **Make the H1 portability reference operationally explicit.** Panel A shows
   one model's five associations, whereas the paper's portability rule also
   contains BH, held-out direction, and a 5-of-8-model subcriterion. The
   current phrase "The slice fails the portability condition" is true, but a
   reader cannot reconstruct that multi-model rule from the panel alone.
   Prefer: "Under the paper's fixed model-level portability rule, this frozen
   slice fails." This preserves the correct negative result without implying
   that the plotted five signs alone define the rule.

2. **Avoid any possible preregistration implication.** The figure calls the
   association values "Registered," while the manuscript properly describes
   the instantiated crest follow-up as exploratory and discovery-frozen. The
   source H1 analysis was registered/fixed, but the visualized candidate was
   operationally frozen during the discovery loop. To keep the terminology
   aligned, change "Registered adjusted associations" to "Fixed-analysis
   adjusted associations" (or "Sealed adjusted associations") and retain
   "discovery-frozen" in the caption.

3. **Preserve the explicit side-by-side boundary in the final caption.** The
   machine-readable metadata states `cross_panel_inference: false`, while the
   human caption lists several excluded inferences. The existing language is
   adequate, but the strongest compact final sentence would be: "The panels
   are descriptive displays only; no cross-panel relationship, pooled
   estimate, ranking, new test, or causal inference is computed."

## Verdict

**Accept as an internal draft, with the two targeted wording/encoding
corrections above before external circulation.** The substituted figure is
faithful to the sealed H1/H2 evidence, does not create a causal claim, and
does not broaden the paper beyond H1/H2. Its main remaining risk is semantic:
distinguish the frozen exploratory design from the absent score-eligible
manifest, and do not make the polarity control look like a selected successful
intervention.

## Verification record

- Read and visually inspected `paper/build/main.pdf` and
  `paper/figures/fig_crest_evidence_boundary_main.png`.
- Checked the fixed contract in
  `experiments/paper_extension/main_crest_evidence_boundary_protocol.md` and
  `main_crest_evidence_boundary_input_manifest.json`.
- Checked the output metadata and every declared source/output SHA-256; each
  matched the live file.
- Checked the source values against
  `AGGREGATION_RUN_20260809.md`, the two H1 confirmation rows, and
  `H2_QUALITY_RUN_002.md` / `h2_quality_full_002.summary.json`.
- Ran `PYTHONPATH=. python3 -m pytest -q
  tests/test_main_crest_evidence_boundary.py`: **14 passed**.
