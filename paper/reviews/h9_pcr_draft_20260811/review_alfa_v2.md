# H9-PCR post-revision verification — alfa

## Scope

Read-only verification of the revised `paper/main.tex` and compiled
`paper/build/main.pdf` against the six mandatory findings in `review_alfa.md`.
The PDF text matches the revised source, is five US-letter pages with references
on page 5, has embedded fonts, and passes the local
`single-anonymous-submission` preflight. This is an internal Codex review, not
external peer review.

## Checklist

| Requirement | Status | Verification |
| --- | --- | --- |
| (a) Same-item/content-key wording | **Mostly resolved** | The title, abstract, method, limitations, and conclusion now correctly use *same-item* / documented metadata language. Lines 127--133 explicitly say the key is not semantic evidence and can bundle speaker, recording, duration, and phonetic correspondence. However, Figure 1 still displays `P: content-aligned` and the heading `Content-aligned advantage`; change these display-only labels to `P: same-item` / `Same-item advantage` (or explicitly define content-aligned as metadata-key-aligned) so the figure cannot revive the stronger interpretation. |
| (b) AUROC, counts, and orientation | **Resolved** | Table 2 reports all six EERs, all six AUROCs, and target bona-fide/spoof counts. Lines 153--155 and Table 2 specify mean spoof probabilities and spoof-positive AUROC. This makes the weak absolute regime visible. |
| (c) Ensemble-conditional bootstrap | **Resolved** | The abstract, figure caption, Results, and limitations now identify the 2,000 shared-ID trial bootstrap as conditional on the frozen four-seed ensemble. Lines 246--250 correctly state that it does not resample source voices, training seeds, or a new B2 draw. |
| (d) B2/B1 compute qualifications | **Resolved** | Lines 169--174 state that B2 inherits P's selected lambda and is not optimized independently; they also distinguish the matched training envelope from wall-clock compute because B1 has no rank-edge forwards. The equal-edge wording is appropriately limited. |
| (e) Availability locator | **Partly resolved** | Lines 280--288 give a concrete repository URL, H9 path, compact artifact inventory, and dataset-license boundary. Before submission, make this immutable: push the reviewed revision and cite a commit hash, release tag, or DOI/archival snapshot. A mutable root URL alone does not guarantee that the exact paper/code/ledger state will remain recoverable. |
| (f) Res2 provenance citation | **Resolved** | The text cites `borodin2024res2tcn` where it introduces the public 172k Res2TCNGuard architecture. The BibTeX entry and citation-verification ledger point to the published architecture paper and correctly distinguish it from the prohibited pretrained checkpoint. |

## Residual recommendation

After synchronizing the figure wording and creating an immutable public
revision, the prior **major-revision** concerns are adequately addressed. The
manuscript's remaining claim is appropriately narrow: a fixed same-item versus
different-item pairing comparison for a fresh compact Res2TCNGuard/ODSS
protocol, not a linguistic-content mechanism, SOTA result, or general
deployment claim. No post-terminal experiment or retuning is warranted.

## Verdict

**Accept after the two documentation/display fixes above.**
