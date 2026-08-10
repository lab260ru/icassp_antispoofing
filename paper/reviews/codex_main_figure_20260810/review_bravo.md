# Codex review bravo — main-figure narrative and interpretation audit

## Summary

The replacement figure improves the manuscript's hierarchy: it now places the
actual result—the failure of the discovery-frozen
`Spectra-AASIST/full_waveform/spoof/crest_factor_db` slice to satisfy the
five-corpus rule—and the pre-score H2 stop in the same, readily inspectable
location. Its five H1 values, held-out intervals, four H2 retentions, fixed
orders, and 90% line agree with the hash-bound metadata and the H1/H2 reports.
The renderer's eight-input contract also excludes H4/H5/H6/H2B/S1, raw scores,
and detector outputs. The caption and surrounding prose explicitly avoid a
causal, score-delta, ranking, or pooled claim.

The figure is therefore an improvement over a pass-count atlas for this paper's
narrow negative result. Two presentation boundaries nevertheless need repair
before treating it as the final main-paper figure: the five-corpus H1 result
and the one-corpus H2 quality screen can be conflated, and the plot does not
visibly separate discovery from held-out confirmation.

## Major concerns

1. **The shared title can overstate the H2 scope.** “Crest factor: an evidence
   boundary across five corpora” spans both panels, but panel B is only the
   detector-free ASVspoof2019 LA, 1,000-clip quality screen. It was not a
   five-corpus H2 result. A skimming reader can incorrectly transfer the
   five-corpus qualifier from panel A to the H2 gate failure. Revise the title
   or add an unavoidable panel-B qualifier, for example “ASVspoof2019 LA,
   n=1,000, detector-free,” and state the same scope in the caption. This is a
   labeling correction, not a new analysis.

2. **Panel A does not make the staged design legible at figure-reading speed.**
   The fixed order and “95% CI only for held-out confirmations” footer are
   correct, but the first three discovery corpora and final two held-out
   confirmations look like one undifferentiated five-row series. The paper's
   evidential point depends on this separation. Add fixed group labels and/or a
   non-data separator between the third and fourth rows (e.g., “discovery” and
   “held-out confirmation”); do not recolor, sort, or otherwise re-emphasize
   the observations. Record this as a new versioned display contract rather
   than modifying the inspected asset in place.

## Minor concerns

1. The main title says only “Crest factor,” while the decisive scope
   (Spectra-AASIST, spoof class, full waveform, discovery-frozen) is in a
   small footer. Promote at least “discovery-frozen crest slice” into the title
   or panel-A subtitle so the visual cannot be mistaken for a feature-wide or
   model-wide result.

2. The green 99.9% polarity bar is visually more salient than the three failed
   arms and can look like a successful result despite the panel-level stop. It
   is factually correct, but a neutral single palette (with the dashed gate and
   explicit “panel not frozen” statement) would better honor the protocol's
   “no winner” intent. If pass/fail color remains, add an explicit legend that
   the color is arm-level gate status and cannot authorize scoring.

3. Replacing the atlas appropriately narrows the narrative, but the abstract
   and Results still invoke a “reproducible association atlas” / “complete
   descriptive table” without an in-paper locator. Add a concise companion
   artifact pointer in the final archival package, or call it a hash-sealed
   companion table rather than implying that it appears in the manuscript.

4. The figure is floated alone after the conclusion in the current compiled
   PDF. That is readable, but it weakens the Results-to-figure connection. If
   page budget allows, place it adjacent to the Results text; otherwise keep
   the first Results sentence highly specific, as it is now, and avoid adding
   more detached visual material.

## Verdict

**Major revision before final submission.** The current figure is evidence
consistent and materially improves the paper's central hierarchy; it neither
imports supplementary analyses nor makes a causal claim. The two major scope
labels should be fixed so a reader cannot mistake the one-corpus, pre-score H2
quality failure for a five-corpus detector result or mistake discovery rows for
confirmatory replications.
