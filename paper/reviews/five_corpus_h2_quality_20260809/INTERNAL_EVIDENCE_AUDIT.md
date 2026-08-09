# Internal evidence audit — five-corpus / H2-quality draft

**Status:** completed 2026-08-09; internal artifact audit, **not** external
peer review and not an ARA Seal assessment.

## Why this audit exists

The external `paper-review` workflow could not authenticate (see
`REVIEW_STATUS.md`).  The repository also does not use the formal ARA
`PAPER.md` / `logic/claims.md` artifact layout required by the automated rigor
reviewer.  This note is a narrower replacement: it checks that the central
empirical statements in `paper/main.tex` are backed by immutable or committed
research artifacts and that the paper does not overstate the failed H2 stage.

## Claim-to-artifact check

| Manuscript claim | Supporting artifact | Audit disposition |
|---|---|---|
| H1 comprises 6,720 within-class screen cells across five corpora, eight score artifacts, two classes, three views, and 28 features. | `experiments/h1_feature_association/results/five_corpus_aggregate_20260809T214500Z/AGGREGATION_RUN_20260809.md` | Supported: 168 feature/view/class units and 1,344 model/unit rows; the stated matrix product is 6,720. |
| The frozen Spectra-AASIST/full-waveform/spoof/crest-factor candidate fails the five-corpus portability condition. | Same aggregation note and `portable_association_aggregation_report.json` (SHA-256 recorded in that note). | Supported: In-the-Wild partial rho -0.032606, q 0.000523; ASVspoof5 rho 0.001383, q 0.937089. The paper calls the latter null after adjustment and does not claim a causal effect. |
| Nineteen of 168 registered feature/view/class units meet the descriptive rule. | `portable_association_feature_report.csv`, with SHA-256 `153a08438ee08c6ffb52159569dd33a593314fd78f6bc01aec89da9d70c3d356`. | Supported, with correct scope: it is a registry-wide association screen, not an H2 candidate selection. |
| The frozen H2 crest/control quality panel fails before any detector scoring. | `experiments/h2_causal_interventions/results/quality_runs/H2_QUALITY_RUN_002.md`; compact `h2_quality_full_002.summary.json`; immutable HDD hashes recorded in the note. | Supported: DRC-3 18.1%, DRC-6 0.6%, gain 64.6%, and polarity 99.9% retained; the 90% per-arm gate blocks the panel. |
| No H2 paired deltas, causal sensitivity result, or H3 training result is reported. | H2 quality note plus absence of a score-eligible manifest. | Consistent: the paper explicitly withholds causal and mitigation claims. |

## Findings and required wording

1. The manuscript must continue to label the 19-unit atlas as *descriptive*
   and must not re-rank it into intervention candidates after confirmation.
2. The H2 result is a quality-gate incompatibility for this frozen panel.  It
   must not be characterized as detector robustness, lack of sensitivity, or a
   causal negative result.
3. The paper should not say confirmation is pending: both held-out H1 corpora
   have been analyzed.  This wording correction is included with this audit.
4. Exact data, full quality tables, transcripts, and checkpoints reside on the
   documented HDD path; committed notes retain their hashes and compact
   summaries.  The manuscript must retain this distinction rather than imply
   that large raw artifacts are versioned in Git.

## Remaining review limitation

This audit does not assess novelty, writing quality, citation completeness, or
statistical validity beyond verifying the paper's stated linkage to artifacts.
An authenticated independent paper review remains required before submission.
