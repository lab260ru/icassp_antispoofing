# Scope and layout audit — revised four-page draft

**Status:** internal read-only audit, 2026-08-09. It applies the repository's
paper-review criteria but is not an external peer review; the authenticated
three-reviewer launcher remains unavailable as recorded in `REVIEW_STATUS.md`.

## Findings and disposition

| Finding | Disposition in `main.tex` |
|---|---|
| The completed H2 evidence is a 1,000-clip, four-arm score-blind ASVspoof 2019 LA quality screen, whereas the former text read as if the planned four-model/multi-corpus design had been run. | Rewritten as a planned confirmatory H2 plus its completed exploratory quality-only component. |
| Generic “crest factor fails” risks conflating the failed frozen spoof-class candidate with a separate descriptive bona-fide crest unit. | Abstract, limitations, and conclusion now name the failed `Spectra-AASIST/full-waveform/spoof` candidate. |
| The notation `T(f)` suggests an isolated causal intervention on a feature, while the DSP transformations change a waveform family. | Rewritten as `T_\theta(x)` and limited to waveform-family sensitivity under quality proxies. |
| Table I was stranded after the paper text in the previous PDF. | Removed engineering/pilot rows, compressed its rule statement, and rebuilt it on page 2 before References. |
| H2B Q1 is a useful negative feasibility outcome but has no detector, score, EER, candidate-selection, or causal result. | Kept out of the four-page paper; retained in future-direction artifacts. |

## Compile validation

`tectonic --outdir build main.tex` succeeds after the revision. The rebuilt
`paper/build/main.pdf` has four pages. Programmatic PDF inspection finds
`Table I` and its `Artifact` header on page 2 and `References` on page 3, so
the former post-reference table placement is resolved. Remaining TeX warnings
are known underfull boxes and Tectonic's bibliography-rerun warning; there is
no unresolved-citation or overfull-box warning.
