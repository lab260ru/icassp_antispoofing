# Paper workspace

This directory will contain an ICASSP-format source tree, reproducible figures,
verified bibliography, compiled PDF, narrative history, and review bundles.
No table cell or citation is added without a source artifact. The current source
contains authorized working metadata for the ICASSP 2027 single-anonymous
policy; confirm the portal record before submission as documented in
`AUTHOR_BLOCK_REQUIRED.md`. It uses the user-provided ICASSP-2026
spconf/IEEEbib pair pinned under `template/ICASSP2026/`; its provenance and the
required future ICASSP-2027 reconciliation are in
`template/ICASSP2026/TEMPLATE_PROVENANCE.md`.

The current draft is **Same-Item Pair Ranking for Cross-Corpus Speech
Deepfake Detection**. It is based solely on the sealed H9 controlled-transfer
experiment: fresh Res2TCNGuard instances trained on the ODSS paired source
pool, evaluated once on fixed SONAR and ArAD targets. Its central comparison is
same-item ranking (P) versus same-pool BCE (B1) and an equal-edge,
stratum-matched random-pair ranking control (B2). Read
`../experiments/h9_paired_counterfactual/PLAN.md` and
`../experiments/h9_paired_counterfactual/results/H9_TERMINAL_EVALUATION_001.md`
before editing quantitative claims.

The current Figure 1 is the hash-validated H9 terminal EER comparison,
documented in `figures/FIGURE_H9_PCR_TERMINAL.md`. The prior crest/quality
materials remain repository history and are not blended into this paper.

The manuscript is an internal named-author working draft, not a submitted
paper or an uploaded supplement. It makes neither a state-of-the-art,
architecture-general, causal-representation, nor blind-evaluation claim.
