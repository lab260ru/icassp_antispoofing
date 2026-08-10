# Codex presentation audit — main evidence-boundary figure

## Scope and materials inspected

This is a Codex-only presentation/layout audit, not an external peer review
and not an evidentiary re-analysis. I inspected the current compiled
`paper/build/main.pdf` (five US-letter pages; the first four are technical
pages and page five is references), the vector
`paper/figures/fig_crest_evidence_boundary_main.pdf`, its 300-dpi PNG preview,
the figure metadata, and the current `paper/main.tex`. For comparison, I read
the immediately preceding main-figure caption at commit `7d430c4`; it was a
registry-wide pass-count atlas.

## Summary

The new full-width Fig. 1 is a clearer main-paper visual for the present
negative-result narrative than the former pass-count atlas. It places the two
facts that bound the paper's claim next to each other: the discovery-frozen
crest association weakens through the fixed five-corpus sequence, and three of
four pre-score quality arms do not clear the 90% retention gate. The rendered
values and order agree with the hash-recorded metadata: the H1 order is 2019
LA, 2021 LA, 2021 DF, In-the-Wild, ASV5; the two confidence intervals occur
only on the held-out rows; and the H2 percentages are 18.1, 0.6, 64.6, and
99.9 against a 90% gate.

The presentation is readable at the full two-column target width. The source
is vector PDF, the published preview is 300 dpi, the two panels have clearly
separated axes, and the right-panel percentage labels and dashed gate retain
their meaning without relying on colour alone. Figure, caption, and manuscript
all consistently state the important caveat: the panel was not frozen and no
detector score, pooled estimate, ranking, new test, or causal inference was
produced. It fits on the fourth technical page with its caption and does not
move references outside the dedicated fifth page.

## Major concerns

None within this presentation-only audit. The figure does not introduce a
layout collision, raster-quality issue, scale manipulation, or an apparent
causal/score-result claim.

## Minor concerns

1. The distinction between the three discovery rows and the two held-out rows
   is presently inferred from their position and from the note that only the
   latter have intervals. A subtle `discovery` / `held-out confirmation`
   grouping marker (or a thin separator) would make the evidence boundary
   immediately legible without changing any data, scale, or analysis. This is
   useful because the absence of discovery intervals is a design property, not
   an uncertainty claim.

2. Replacing the prior pass-count atlas is the right choice for the central
   narrative, but it leaves the 19/168 descriptive registry outcome represented
   only by Table 1 and prose in the main paper. Keep the former atlas readily
   available as a supplementary/repository artifact and, if a submission
   package is prepared, point readers to it. Do not enlarge the main paper or
   re-rank the registry to compensate.

3. The compact axis labels (`ASV19 LA`, `ASV21 LA`, `ASV21 DF`, and `ASV5`)
   conserve space and are unambiguous after the methods section, but a short
   expansion in the caption or nearby text would improve standalone reuse of
   the figure. The current caption already gives the full semantic scope of
   the slice, so this is optional rather than a correctness defect.

4. The entire fourth technical page is intentionally devoted to Fig. 1 and
   its caption. That is acceptable under the present four-page technical
   layout and makes the negative boundary easy to inspect; it also means this
   figure should remain concise. The current two-panel design earns that space.

## Verdict

**Minor revision (presentation polish only).** The main figure is suitable
for the current draft as a conservative, accurately caveated evidence-boundary
visual. A discovery/held-out grouping cue is the only recommended pre-submission
polish; it is not required to preserve the evidence boundary or page budget.
