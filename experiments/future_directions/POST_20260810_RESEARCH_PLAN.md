# Post-2026-08-10 research plan

**Status:** prospective roadmap only; no experiment is authorized by this
document. It records the safe next decisions after the 2026-08-10 bounded
research loop, rather than attempting to rescue a failed gate.

## Evidence boundary at handoff

- The discovery-frozen crest-factor H1 candidate is not portable across the
  fixed five-corpus rule.
- H2 has no transformed detector score: three of four predeclared arms fail
  the detector-free 90% retained-pair quality gate.
- H2B Q1 selects no non-control waveform family under its locked Wilson-bound
  rule. Its Q2--Q4 stages are closed.
- H4, H5, H6, and H7 are complete descriptive audits. H7's fixed all-28-feature
  transfer baseline is strong on the held-out ASVspoof cohorts but weak on
  InTheWild/ASVspoof5; like the atlases, it cannot choose an intervention
  feature, a model, a scorer, or a training input.
- H3 is not authorized because the causal prerequisite does not exist.

Consequently, no threshold relaxation, transform-grid expansion, source-panel
reuse, scorer substitution, or model training may be described as a
continuation of H1/H2/H2B/H3.

## Direction A — fresh quality-first intervention feasibility

If a causal sensitivity study is still desired, start a **new protocol** before
opening any score artifact or detector code. It must:

1. declare a waveform source independent of the completed five-corpus H1
   results and H2B Q1 calibration selection;
2. freeze finite transform families, parameters, primary target diagnostics,
   retention criteria, and a source-clustered sample manifest from metadata
   only;
3. run quality/ASR/feature diagnostics without detector access, record every
   failure, and lock a retained-only panel before scoring;
4. independently freeze a feature identity using *new* discovery and held-out
   confirmation corpora, without consulting H4/H5/H6/H7 for selection; and
5. validate four executable scorers against their own untouched published-score
   baselines before any paired detector analysis.

Only a passed pre-score quality panel and an independently selected identity
would authorize a new causal paired-score protocol. A negative feasibility
outcome is final for its exact declared family.

## Direction B — prospective association discovery

If the question is instead association portability, create a new corpus split
and register a finite feature/model/view panel before reading new score data.
Use a discovery/confirmation firewall, fixed within-class controls, multiplicity
family, clustered intervals, and a written portability criterion. Existing H1
and H4/H5/H6/H7 registries may provide background but cannot be mined to name the
next candidate. Report the complete registered matrix regardless of selection.

## Direction C — training only after causal evidence

Feature-conditioned training remains contingent on a valid causal result from
Direction A and a new training protocol. That protocol must predeclare:

- train/dev/test datasets, source/speaker leakage controls, and split hashes;
- a matched baseline recipe, model implementation/revision, loss, optimizer,
  BF16 precision, seed set, batch/worker throughput sweep, and GPU allocation;
- the feature-conditioning mechanism and its ablations;
- held-out EER/min-tDCF (as applicable), calibration, and transformation
  consistency measures; and
- a primary comparison, uncertainty method, and stopping rule.

It must not use the current descriptive atlases as a post-hoc feature shortlist.

## Paper and archival work

The current manuscript is a conservative negative-result working draft. Before
submission, reconcile the pinned ICASSP-2026 template with the official
ICASSP-2027 kit, obtain the official PDF/checker outcome, create an authorized
permanent artifact/archive locator for the hash-bound outputs, and decide
whether supplementary figures S1/H5/H6/H7 are permitted in the submission
package. Any `paper/main.tex` modification must immediately rebuild and commit
`paper/build/main.pdf`.
