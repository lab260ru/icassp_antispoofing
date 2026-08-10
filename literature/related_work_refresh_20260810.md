# Related-work refresh — 2026-08-10

Purpose: a bounded primary-source refresh after the final Codex paper audit.
This note adds no experimental claim, paper result, or unverified citation.
It preserves sources that may be useful at camera-ready time once page budget
and an authorized archive/supplement plan are decided.

## Verified primary sources

| Source | Verified limited finding | Possible paper role | Current decision |
|---|---|---|---|
| Shim et al., [*Beyond Silence: Bias Analysis through Loss and Asymmetric Approach in Audio Anti-Spoofing*](https://www.isca-archive.org/syndata4genai_2024/shim24_syndata4genai.html), SynData4GenAI 2024, DOI `10.21437/SynData4GenAI.2024-12` | Extends anti-spoofing bias analysis beyond silence and reports class-wise training-dynamics differences. | Supports the paper's distinction between a label cue and model behavior. | Retain as an optional related-work citation; do not expand the four-page main draft without a focused sentence and recompilation. |
| Dao et al., [*Spoofing detection in the wild: an investigation of approaches to improve generalisation*](https://www.isca-archive.org/odyssey_2024/dao24_odyssey.html), Odyssey 2024, DOI `10.21437/odyssey.2024-21` | Evaluates generalisation strategies under domain mismatch, including In-the-Wild. | Context for cross-domain robustness approaches. | Retain for a future robustness/mitigation discussion; it does not validate the unrun H3 protocol. |
| Wang et al., [*ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale*](https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html), ASVspoof 2024, DOI `10.21437/ASVspoof.2024-1` | Describes ASVspoof5's crowdsourced, diverse acoustic conditions and challenge setup. | Contextualizes use of ASVspoof5 as one held-out corpus. | Retain for camera-ready dataset description if an extra reference fits; do not change the reported H1 estimator. |

## Screening decision

The current main draft already has source-verified citations directly supporting
its critical shortcut/generalization and H3 background statements. The final
Codex review found no citation-dependent numerical or causal defect. These
three papers should therefore remain in this literature ledger rather than be
inserted opportunistically into the locked five-page working draft. Any future
citation import must add a BibTeX entry and claim-specific verification row,
then recompile and rerun paper preflight.
