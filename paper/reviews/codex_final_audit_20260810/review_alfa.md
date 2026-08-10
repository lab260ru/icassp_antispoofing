# Codex methods and claim audit — reviewer alfa

**Review type:** internal, Codex-only evidence audit; this is not external
peer review.  It audits the compiled draft and the repository artifacts named
below.  No external website, model, or reviewer runtime was used.

## Material inspected

- Compiled paper: `paper/build/main.pdf` (five US-letter pages; technical
  material on pp. 1--4 and references on p. 5) and `paper/main.tex`.
- H1: `experiments/h1_feature_association/results/
  five_corpus_aggregate_20260809T214500Z/AGGREGATION_RUN_20260809.md`, its
  two CSV reports, and the five source `association_summary.csv` files.
- H2: `experiments/h2_causal_interventions/results/quality_runs/
  H2_QUALITY_RUN_002.md` and its compact JSON summary.
- H5/H6: their locked protocols, result notes, and the hash-named HDD matrix
  and summary CSVs cited by those notes.

The local static PDF preflight passes in `single-anonymous-submission` mode:
five pages, Table 1 on p. 3, references on p. 5, US-letter media boxes, and
embedded fonts.  This check is layout hygiene only, not official ICASSP or PDF
eXpress validation.

## Summary and assessment

This is an unusually careful negative-result paper.  Its strongest contribution
is not that crest factor is harmless or that any of the 19 descriptive units is
a shortcut; it is the locked separation of (i) feature/label separation,
(ii) within-class score association, and (iii) a quality-gated paired
intervention.  The draft consistently refuses to turn a failed quality gate or
a cross-corpus association screen into causal evidence.

The central numerical claims checked here are supported by the sealed result
artifacts.  The main revisions needed are methodological disclosure and sharper
terminology around the five-corpus decision rule.  H5 and H6 should remain
supplementary: they are valid, independently frozen descriptive atlases, but
neither advances the paper's primary causal-evidence chain and both would
dilute a constrained four-page narrative.

## Numerical-claim audit

| Paper claim | Artifact check | Result |
|---|---|---|
| 6,720 H1 cells = 5 corpora x 8 models x 2 classes x 3 views x 28 features | H1 aggregate reports 1,344 model/unit rows across the fixed 168 view/feature/class units; the product is 6,720. All five configured corpora and eight score artifacts are recorded as supplied. | Verified. |
| Frozen Spectra-AASIST/full-waveform/spoof crest partial correlations are -0.454, -0.099, -0.065 in the three discovery corpora | Exact rows are -0.4538847801, -0.0991427559, and -0.0649496949. | Verified (appropriate rounding). |
| In-the-Wild adjusted association is -0.0326 with q=0.000523; ASVspoof5 adjusted association is 0.0014 with q=0.937089 and raw association is negative | Exact rows are -0.0326059797/q=0.0005225356 and 0.0013834716/q=0.9370890683; the ASVspoof5 raw Spearman value is -0.1555684412. | Verified. |
| The frozen candidate fails the fixed portability rule; 19/168 registered units meet it | The H1 aggregation JSON records 19 of 168.  The candidate's model row has `both_confirmation_q_le_alpha=False` and `both_confirmation_match_direction=False`. | Verified. |
| Eighteen of the 19 units are bona-fide class; the only spoof unit is pre-emphasized spectral flatness | The feature report yields class counts 18 (label 0) and 1 (label 1); the latter is `preemphasized_crop/spectral_flatness`. | Verified, assuming the stated label convention (0 bona fide, 1 spoof). |
| H2 quality screen retained 18.1%, 0.6%, 64.6%, and 99.9% for DRC-3, DRC-6, gain, and polarity and did not score detectors | The H2 note/JSON record 181, 6, 646, and 999 of 1,000; `detector_scoring_allowed=false` and no score-eligible manifest exists. | Verified. |
| H5 has 420 cells/84 aggregations, 17 terminal descriptive stable units, and six explicit degenerate cells | The sealed H5 matrix has 414 `ok` plus six `failed_zero_or_nonfinite_pooled_iqr` cells; the aggregation has 17 stable units.  All six failures are InTheWild `silence_fraction`/`clipping_fraction` across the three view pairs. | Verified; not currently a paper claim. |
| H6 has 280 complete cells and agreement spanning -0.629261 to 0.868770 | The sealed matrix has 280 `ok` cells with 5,000 exact joins each; direct summary yields median 0.3324774211 and extrema -0.6292605821/0.8687696230. | Verified; not currently a paper claim. |

## H5/H6 scope decision

**Do not add H5 or H6 results to the four-page main paper in this revision.**

- **H5** explains why waveform view should be reported, but its 17/84
  score-free measurement-stability design is an ancillary descriptive result,
  drawn from a different independently frozen sample.  It cannot validate,
  select, or rescue H1/H2.  Keep the locked concordance figure and its result
  note as supplementary material.  At most, a one-sentence artifact pointer is
  appropriate after the main paper has adequate method disclosure.
- **H6** demonstrates that within-class score rank agreement varies greatly
  across corpus/class/model-pair cells.  This is useful provenance context, but
  it does not establish model independence or detector diversity and has no
  causal connection to crest factor.  Adding it would invite an avoidable
  second paper question.  Keep it supplementary and do not use it to strengthen
  claims about the eight-model panel.

## Major concerns

1. **The paper does not disclose the implemented partial-adjustment design
   precisely enough to reproduce or evaluate the central estimand.**  The text
   says controls include duration, loudness, and metadata "such as" speaker,
   attack, and codec, while later admitting that availability differs by
   corpus.  The H1 protocol is more specific: controls are conditional on
   metadata availability and the tested feature/response is removed from its
   own control set.  Because the paper's main conclusion rests on adjusted
   partial Spearman values, add a compact corpus-by-control availability table
   in the supplement (and a one-sentence pointer in Sec. 3.2), state the
   exact rank-residual procedure, and state the self-control exclusion rule.
   Also identify the source/speaker cluster used for each held-out bootstrap.

2. **The word “present” understates the five-corpus acceptance criterion.**
   Section 3.2 says a unit must be "present in at least five of eight models."
   The sealed criterion is stronger: at least five models must each meet the
   full model-level subcriterion (same direction in at least 4/5 corpora,
   BH-significant in both held-outs, and matching held-out direction).  Replace
   “present” with that precise formulation, and make the Fig. 1 caption say
   “models satisfying the full fixed subcriterion,” not merely “models
   satisfying ... partial-Spearman/BH.”  This avoids a material
   misinterpretation of the 19/168 count.

3. **Artifact availability is not yet submission-grade.**  “A citable archival
   locator will be added” is not a reproducibility mechanism for an accepted
   paper.  Before submission, deposit the source code, pinned manifests,
   compact CSVs, hashes, PDF figures, commands, and a README in a permanent
   archive; cite its versioned DOI/URL.  Keep raw audio, model weights, and
   large logs external as the repository policy requires, but specify the
   access conditions and expected directory layout.

4. **The interpretation is responsible but the title/abstract should narrow
   the claimed scope further.**  The paper disproves portability for one
   discovery-frozen *adjusted association* under one feature/view/class/model
   slice; it does not establish that crest factor is globally irrelevant or
   that the 19 remaining screen units are portable mechanisms.  The body is
   clear, but add “for the frozen Spectra-AASIST spoof/full-waveform slice” to
   the first conclusion sentence or title-adjacent framing to prevent a title
   reader from overgeneralizing “Beyond Crest Factor.”

## Minor concerns

1. Define the numerical label convention once (e.g., `0=bona fide,
   1=spoof`) before reporting the 18/1 split; the paper otherwise uses prose
   labels only.
2. State the deterministic-crop duration/location and the exact
   pre-emphasis operator in the supplement or feature-registry appendix.  The
   three-view result is difficult to interpret without these definitions.
3. Clarify that feature-label AUROC is reported as descriptive context and is
   not part of the H1 portability rule or H2 candidate reselection.
4. The figure has a defensible evidence-boundary caption, but a reader cannot
   recover the per-corpus values or uncertainty from it.  Keep the compact
   table in the archive/supplement and cite it explicitly in the caption or
   availability statement.
5. Preserve the current explicit statement that the crest H2 follow-up is
   exploratory.  Do not call the H2 screen “confirmatory” merely because its
   quality rules were locked; the cue freeze occurred during the discovery
   loop, as the manuscript correctly says.

## Actionable fix order

1. Correct the portability-rule wording and figure caption.
2. Add a compact adjustment/bootstrap disclosure pointer plus a complete
   supplementary control-availability table.
3. Add versioned artifact-access text before submission.
4. Make the central-slice scope explicit in the conclusion/abstract framing.
5. Leave H5/H6 out of the main text; link their sealed supplementary artifacts
   only after the primary reproducibility disclosures are fixed.

## Verdict

**Major revision (methods/reporting), with no numerical contradiction found in
the main H1/H2 claims.**  The conservative evidence boundary is credible and
worth preserving.  Address the adjustment/portability disclosure and archive
availability before treating this as submission-ready; do not add H5/H6 merely
to make the main paper appear broader.
