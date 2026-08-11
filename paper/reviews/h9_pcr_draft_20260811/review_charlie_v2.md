# Internal Codex-only post-revision check — Charlie

**Verdict: Accept with one minor presentation cleanup.** The revised PDF
addresses the requested reader-facing provenance and bibliography issues and
is a coherent five-page ICASSP working layout. No result or protocol change is
needed.

## Checks passed

- `paper/build/main.pdf` has **five US-letter pages**. The local
  single-anonymous readiness check passes: Table 1 is recoverable on page 2,
  References on page 5, and all 13 recovered fonts are embedded. Pages 1--4
  contain technical material; page 5 is references only.
- The PDF has a useful reading order: method and Tables 1--2 on page 2, the
  full-width terminal figure and seed table on page 3, then scope/limitations
  and conclusion. The three compact tables are labeled, have units, and make
  the ensemble-versus-seed status explicit.
- Figure 1 is legible in both its inspected PNG source and the PDF text layer:
  method labels, bar values, confidence intervals, endpoint direction, and
  frozen-ensemble condition are all present without the previously observed
  overlap. Its caption now says “Sealed terminal evaluation,” so the
  unexplained `H9` label is no longer reader-facing there.
- Res2TCNGuard now has an in-text citation to
  `borodin2024res2tcn`; the citation ledger links it to DOI
  `10.48084/etasr.8906` and a publisher PDF. The manuscript also gives a
  repository/artifact location. These resolve the first-review traceability
  concerns.
- The displayed Reference [7] now correctly renders **Mathew Magimai-Doss**;
  the prior erroneous full stop is gone. EER is expanded in the abstract and
  precisely defined in the method.

## One minor cleanup before the next compiled revision

The rewritten paper consistently uses **same-item** to avoid treating the
metadata key as proof of linguistic content. The figure still says
“Content-aligned advantage” and its legend says “P: content-aligned.” Rename
those two graphic labels to “Same-item advantage” and “P: same-item” (or
explicitly define the equivalence) so the primary visual does not reintroduce
the stronger terminology the text carefully disavows. The body also retains
two internal `H9` mentions (“H9 loss hyperparameter” and “H9 protocol”);
replace them with “study”/“protocol” or define H9 once if a fully
reader-independent version is desired.

The pinned 2026 template remains correctly documented as a working input, not
proof of final ICASSP-2027 template compliance. Retain that reconciliation
step.
