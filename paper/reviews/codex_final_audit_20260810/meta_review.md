# Meta-review — Codex final audit

## Consensus

All three reviewers find the paper's central H1/H2 evidence credible and
appropriately conservative: the exploratory discovery-frozen
Spectra-AASIST/full-waveform/spoof crest slice fails the locked five-corpus
portability rule, and the frozen H2 crest/control panel stops at its quality
gate before detector scoring. No review found a numerical contradiction in
those claims. All reviewers also agree that H5 and H6 are useful sealed
supplementary diagnostics but should not broaden this constrained main-paper
question.

The common verdict is **major revision before submission**, driven by
method-reporting precision, narrative calibration, and submission provenance
rather than by a request for a new causal experiment. The paper is acceptable
as an evidence-conservative internal draft after the stated editorial fixes.

## Resolved in this revision

- The main text now calls the crest slice exploratory and discovery-frozen,
  rather than implying a public prospective preregistration.
- The contribution list distinguishes completed evidence from the conditional,
  unrun H3 protocol.
- The H1 partial-rank controls, self-control exclusion, within-corpus 1,344
  cell BH family, and held-out bootstrap clustering are stated explicitly.
- The five-corpus rule and atlas caption now name the full model-level
  portability subcriterion; the H2 table records polarity's 99.9% retention.
- The main text states that its estimands concern pinned released score
  artifacts, not current executable implementations or independent mechanisms.
- The conclusion foregrounds association portability and a future sensitivity
  test; it does not claim causality. The modified `main.tex` was rebuilt and
  passed the local five-page, US-letter, embedded-font preflight.

## Remaining submission-stage conditions

- Publish an authorized versioned artifact locator for code, manifests, compact
  results, commands, and hashes; raw audio/weights remain external under the
  repository policy.
- Reconcile the user-supplied ICASSP-2026 build inputs with the official
  ICASSP-2027 kit and run the official checker.
- Decide, under an authorized submission/supplement policy, whether to replace
  or compact the current pass-count atlas with the separately locked S1
  evidence-boundary display. Do not cite S1, H5, or H6 from the present main
  draft without that package decision.
- Retain the H5/H6 artifacts as supplementary-only context; neither can
  select a cue/model or revise H1--H3/H2B/H4 decisions.
