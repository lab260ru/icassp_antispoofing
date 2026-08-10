# Research and paper handoff — 2026-08-10

## Delivered

- Readable named-author ICASSP working draft: `paper/main.tex` and
  `paper/build/main.pdf` (SHA-256
  `4395f1e131e74e926c475a43c7a965a51ee46dcc73387dbf4e9166120b7fdd4c`).
  The current source uses the user-provided, hash-pinned ICASSP-2026
  `spconf`/`IEEEbib` inputs and was independently audited by three Codex-only
  perspectives. A further Codex-only figure review replaced the registry
  pass-count atlas with a hash-pinned H1/H2 evidence-boundary Figure 1 that
  explicitly separates discovery from confirmation and the five-corpus H1
  result from the one-corpus H2 quality screen.
- Complete H1 five-corpus association audit: the discovery-frozen
  Spectra-AASIST/full-waveform/spoof crest-factor association is **not
  portable** under the locked rule. The 19/168 other entries are descriptive
  and are not intervention candidates. A score-free covariate-availability
  report now makes the exact five-cohort adjustment coverage and confirmation
  bootstrap cluster context directly inspectable.
- Complete H2 and H2B quality-first stops: H2's DRC-3, DRC-6, and gain arms
  retain 18.1%, 0.6%, and 64.6%, respectively, below the 90% gate; H2B Q1
  selects no non-control family. No transformed detector was scored and H3
  training was not run.
- Completed bounded descriptive extensions: H4 label-cue transportability
  (420 cells/84 aggregates), H5 feature-view invariance (420/84, 17 terminal
  descriptive units and six explicit degeneracies), and H6 within-class
  published-score agreement (280 cells/28 summaries, 5,000 joins/cell). These
  do not supply causal, model-ranking, or cue-selection evidence.
- Completed H7 feature-only transfer baseline: a newly frozen all-28-feature
  leave-one-corpus-out logistic recipe scores 0.907785/0.845107/0.780957 on
  the held-out ASVspoof cohorts and 0.534590/0.588320 on InTheWild/ASVspoof5.
  Its hash-validated display and Codex-only review are supplementary-only;
  they read no detector score and cannot establish causality, cue selection,
  model ranking, or mitigation performance.
- Display-only supporting figures: sealed crest evidence-boundary S1, H5
  concordance atlas, H6 agreement atlas, and H7 transfer forest plot. They
  remain outside the current main paper until a submission-package decision is
  authorized.
- Resumption materials: current `AGENTS.md`, `ARTIFACT_INDEX.md`,
  `research-state.yaml`, full research/implementation logs, future roadmap,
  final verification, Telegram receipt, remote branch, and complete-history
  HDD Git bundle.

## Verified now

The full repository suite passes **149 tests**. The local paper checker reports
five US-letter pages, Table 1 on page 3, references on page 5, and ten embedded
fonts. See `verification/FINAL_VERIFICATION_20260810.md`.

## Do not reopen

Do not tune the completed H1/H2/H2B thresholds, feature choice, score
orientation, transform grids, quality gate, sample manifests, or model panels.
Any causal or training study must begin with the independent protocol and
freeze requirements in
`experiments/future_directions/POST_20260810_RESEARCH_PLAN.md`.

## External submission work

Before upload, reconcile the pinned ICASSP-2026 template with the official
ICASSP-2027 kit, run its official checker/PDF workflow, choose an authorized
permanent artifact locator, and decide whether a supplementary package is
allowed. If `paper/main.tex` changes, immediately compile it and commit the
refreshed `paper/build/main.pdf` with the edit.
