# Proposed main-paper Fig. 1 protocol — crest evidence boundary

Status: **drafted before reading any sealed H1/H2 input or rendering an output**
on 2026-08-10. The protocol, manifest, and renderer must be committed before
an explicitly authorized render. This is a display-only proposal for a
possible replacement of the current pass-count atlas; it is not a new
experiment, analysis, or submission-package decision.

## Fixed question and scope

Can a two-panel main-paper figure state the paper's narrow evidence boundary
more directly: the discovery-frozen
`Spectra-AASIST` / `full_waveform` / `spoof` / `crest_factor_db` H1 slice does
not pass the fixed five-corpus portability condition, and the registered H2
crest/control panel stops at the detector-free 90% quality gate?

The figure must not make a causal, detector-reliance, model-ranking, feature
selection, or training claim. It deliberately excludes H4, H5, H6, H2B, S1's
label-separation panel, and all supplementary-only results so that it does
not broaden the paper's scope.

## Immutable input contract

The versioned manifest must enumerate **exactly eight** hash-pinned inputs:

1. five H1 association summaries, one per corpus in this exact order:
   `ASVspoof2019_LA`, `ASVspoof2021_LA`, `ASVspoof2021_DF`, `InTheWild`,
   `ASVspoof5`;
2. two H1 held-out bootstrap rows, in this exact order: `InTheWild`,
   `ASVspoof5`; and
3. the single detector-free H2 quality summary.

The H1 scope is exactly `Spectra-AASIST`, `full_waveform`, spoof class
(`class_label=1`), and `crest_factor_db`. The H2 arm order is exactly
`drc_cf3`, `drc_cf6`, `small_gain_plus_0p1db`, `polarity`; the fixed gate is
90% retained pairs. The manifest may contain no other input and may use only
repository-relative source paths.

The implementation must reject H4/H5/H6/S1 inputs, raw score artifacts,
model files, audio, ASR outputs, H2B outputs, absolute paths, redirects,
source substitutions, unexpected response-like columns, malformed source
schemas, missing inputs, and a mismatched source or protocol hash. It must
not open any artifact outside that allow-list.

## Fixed display

Panel A is a five-corpus, fixed-order dot plot of the existing H1 partial
Spearman associations. Its only intervals are the two already-sealed
held-out confidence intervals. Its fixed horizontal scale is [-0.55, 0.08].
Panel B is a four-arm, fixed-order horizontal-bar display of existing H2
retained-pair percentages with a dashed 90% gate. Its fixed horizontal scale
is [0, 103] percent. The panel title and annotation must state that detector
scoring is unavailable because the panel was not frozen.

The renderer may copy the sealed values to the plot and machine-readable
metadata, but it may not calculate any statistic, bootstrap, pooled estimate,
new threshold, ranking, sort, filter, cross-panel relationship, or test.
All corpus and arm orderings remain as specified above. No point or bar is
highlighted as a winner.

## Output and authorization gate

After the protocol and its implementation are committed and a repository
maintainer explicitly authorizes a render, the only outputs are:

- `paper/figures/fig_crest_evidence_boundary_main.pdf` (vector PDF);
- `paper/figures/fig_crest_evidence_boundary_main.png` (300-DPI PNG); and
- `paper/figures/fig_crest_evidence_boundary_main.metadata.json`.

The renderer refuses by default. The command requires an explicit
authorization flag and a non-empty authorization note, both recorded in the
metadata. It must not be run before a maintainer verifies the committed
protocol and manifest. A failed input revalidation or unreadable output is a
stop condition: record the failure, do not substitute an artifact, alter a
scale, or amend the fixed source set after inspection.

## Required verification

Before any authorized render, focused synthetic tests must show the fixed
manifest contract, source-redirect/hash rejection, forbidden-source firewall,
schema enforcement, and refusal without explicit authorization. After an
authorized render, separately verify vector PDF creation, 300-DPI PNG
dimensions, hashes, and visual readability before proposing a `main.tex`
change. This protocol alone neither changes `paper/main.tex` nor authorizes a
paper rebuild.
