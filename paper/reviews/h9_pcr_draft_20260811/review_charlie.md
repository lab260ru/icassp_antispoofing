# Internal Codex-only review — Charlie (presentation and template)

**Verdict: Minor revision.** The manuscript is a readable, conservative ICASSP
working draft with a coherent visual hierarchy. It needs a few reader-facing
traceability and polish fixes before it should be treated as a submission-ready
version; none requires changing the sealed result.

## Material inspected

- `paper/build/main.pdf` (Tectonic build; 4 pages, generated 2026-08-11)
- `paper/main.tex`, the pinned ICASSP-2026 `spconf`/`IEEEbib` inputs, and
  `paper/figures/FIGURE_H9_PCR_TERMINAL.md`
- `paper/references.bib`, `paper/citation-verification.md`, and the local PDF
  readiness material

The current local readiness command passed in single-anonymous mode with the
H9 landmarks: four US-letter pages, Table 1 on page 2, references on page 4,
and all 13 recovered fonts embedded. The pinned 2026 template provenance
appropriately says that it still requires reconciliation with the eventual
2027 kit; this review does not claim official-template compliance.

## Priority fixes

1. **P1 — remove the unexplained internal label from the reader-facing
   figure.** Figure 1 is captioned “Sealed H9 terminal result,” but `H9` is a
   repository experiment identifier that the paper never defines. Rename it
   to “Sealed terminal evaluation” (or define the identifier once) so the
   first result is intelligible without repository context.

2. **P1 — make the model provenance citable in the manuscript.** The method
   calls Res2TCNGuard a public 172k-parameter architecture but gives no
   citation, archive URL, or revision in the paper. The local citation ledger
   says its model-card provenance lives elsewhere, which is useful for the
   repository but not enough for a conference reader. Add a stable software
   / model-card citation or a concise availability pointer with the exact
   public revision. This is especially important because fresh initialization
   versus an excluded checkpoint is central to the claim.

3. **P2 — add an explicit reproducibility location/commit.** The prose says
   that the repository preserves ledgers and hashes, but contains no URL or
   release/commit identifier. A short code-and-artifact availability statement
   would let a reader connect the well-described terminal firewall to the
   stated evidence. It should identify the frozen H9 materials without making
   a claim that target data may be reused for tuning.

4. **P2 — clean the one visibly malformed bibliography name.** Reference [5]
   renders `Mathew Magimai.-Doss`; the full stop before the hyphen is a
   citation-quality defect. Correct the BibTeX author token while retaining
   the already verified DOI metadata.

5. **P2 — define EER at first use for readers outside the immediate
   anti-spoofing subfield.** A parenthetical such as “equal error rate (EER;
   lower is better)” in the abstract or first results reference is sufficient.
   The figure itself already says lower is better and does not need a redesign.

## Presentation and layout assessment

- The 3 technical pages plus a references-only fourth page are visibly below
  the stated maximum rather than trying to conceal overflow. The updated local
  readiness text and checker defaults match that layout. This is appropriate
  for an initial draft, though later additions should use any remaining
  technical-page capacity only for audit-relevant detail rather than padding.
- The full-width Figure 1 is the right primary visual: it gives the two target
  values, macro endpoint, direction, controls, and bootstrap intervals in one
  place. The bar labels, direct method names, and caption make it interpretable
  in grayscale/for readers who do not distinguish the colors. The previously
  reported legend/interval overlap is absent in the inspected v2 PNG/PDF
  source.
- Table 1 clearly establishes source-only lambda selection; Table 2 correctly
  labels the seed values as secondary to the predeclared probability ensemble.
  This ordering prevents the seed table from visually competing with the
  primary endpoint.
- Section 4 and the discussion make the weak absolute EER, one-architecture
  scope, non-blind historical exposure, and non-causal interpretation
  conspicuous rather than burying them in a footnote. That restraint is a
  presentation strength.
- The author block, two-column layout, index terms, numbered section style,
  reference style, US-letter geometry, and embedded fonts are consistent with
  the supplied `spconf` build inputs. The user-provided 2026 template is not
  evidence of final ICASSP-2027 compliance; retain the documented final-kit
  reconciliation step.

## Positives

The title identifies the controlled variable and evaluation setting without a
SOTA implication. The abstract reports both absolute EERs and the controls,
then explicitly limits the claim. Figure, tables, and text all agree on the
same four-seed ensemble endpoint and confidence intervals. The draft is
unusually clear about what its result does *not* establish, which improves
reader trust.
