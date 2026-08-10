# Codex-only claim-integrity review — H7 supplementary figure

**Scope:** completed H7 supplementary forest plot only. This is an internal,
read-only audit of the named H7 protocols, sealed result notes, authoritative
figure assets/metadata, and the H7 figure renderer/tests; it is not external
peer review and does not assess H1--H6, H2B, or the main paper.

## Result and claim check

**Pass.** The figure is a faithful display of the sealed, complete five-cell
H7 all-28-feature leave-one-corpus-out matrix. Its plotted AUROCs match
`H7_ANALYSIS_001.md` and `h7_leave_one_corpus_out.csv` in fixed corpus order:
0.907785, 0.845107, 0.780957, 0.534590, and 0.588320, with the corresponding
sealed 95% intervals. The `40k` leave-one-corpus-out train pool, `10k`
held-out cohort, all-28 full-waveform feature scope, 500/500 bootstrap
completion, and converged fits also match the result matrix.

The title and footer state the essential boundary directly: **score-free,
feature-only** transfer and **no detector score or causal claim**. The H7
protocol, analysis provenance, and output metadata consistently exclude
detector reliance, causality, cue selection, model ranking, and mitigation
performance. The rendering therefore supports only descriptive feature-label
transfer context; it does not select a cue or alter H1/H2/H2B/H3.

## Provenance and integrity check

**Pass.** Live SHA-256 checks agree with the figure note and metadata for all
four compact inputs, both authoritative figure outputs, and the metadata JSON.
The analysis provenance links the same freeze-manifest and freeze-provenance
hashes; the renderer validates those links, five-row order, finite ordered
intervals, fixed 40,000/10,000 cohort sizes, 500 valid bootstraps, convergence,
and the no-claim provenance before rendering. Focused renderer tests pass
**3/3**, covering output creation/non-overwrite, input-hash drift, and a
response-like-column firewall. The inspected PDF is one-page/vector and the
PNG's declared output is the authoritative `figures_feature_transfer_001`
version, not either preserved superseded layout trial.

## Visual check

**Pass.** The fixed-order corpus labels, blue AUROC points, 95% intervals,
chance line, numeric values, axis range, title, and scope footer are legible.
The chance annotation does not collide with a tick and no numeric annotation
is clipped in the authoritative PNG/PDF.

## Minor recommendations

1. The label "source-ID bootstrap CI" is technically correct, but the sealed
   analysis reports singleton source IDs in these cohorts, making it an
   utterance-level cluster bootstrap. Keep that explicit in any nearby
   supplementary caption/note so readers do not infer repeated-source
   dependence was available.
2. The output metadata pins the protocol by path but not by protocol hash.
   Input/output hashes and freeze links are already sufficient for the current
   artifact, but adding the protocol SHA-256 in a future version would make the
   display contract independently byte-verifiable. Do not modify this sealed
   rendering in place.

## Verdict

**Accept as a supplementary-only internal artifact.** The claims match the
sealed result, the score-free/no-causal boundary is clear, provenance links
are consistent, and the authoritative rendering is readable. The two notes
above are documentation refinements, not grounds to reopen or rerender H7.
