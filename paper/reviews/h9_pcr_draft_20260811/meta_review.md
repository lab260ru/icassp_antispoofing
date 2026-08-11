# Codex-only meta-review — H9 same-item ranking draft

## Scope

This internal review synthesizes three independent Codex-only reviews of the
H9 draft: methods/statistics (`review_alfa.md`), novelty/narrative
(`review_bravo.md`), and presentation/template (`review_charlie.md`). It is
not conference peer review. The reviews checked the manuscript only against
the sealed H9 protocol, compact terminal artifacts, figure provenance, and
primary bibliographic records.

## Verdict

**Accept as a submission-ready internal working draft, with the stated narrow
claim.** The original reviews found the terminal numbers and target firewall
credible but required narrower wording and more transparent diagnostics. Their
post-revision checks accept the manuscript after the terminology-only figure
revision and publication of the immutable repository tag.

## Resolved material concerns

| Concern | Resolution in the working draft |
| --- | --- |
| Linguistic/semantic content claim exceeded the filename-derived metadata key. | The title and claims now use *same-item* pairing. The method and limitations state that the corpus/speaker/stem key is not transcript-level semantic evidence and can bundle other correspondence. |
| Pairwise-learning novelty was overstated. | The related-work paragraph cites Siamese, supervised-contrastive, and bonafide-pair learning. The contribution is the controlled same-item versus equal-edge random-pair comparison, not ranking itself. |
| AUROC, counts, score orientation, and EER meaning were absent. | Table 2 reports every terminal EER/AUROC and target bona-fide/spoof count; the text defines spoof-positive mean-probability orientation and EER. |
| Bootstrap could be misread as a seed-level replication interval. | Abstract, figure caption, results, and limitations say it is a shared-ID trial bootstrap conditional on the frozen four-seed ensemble; Table 3 exposes seed heterogeneity. |
| Control fairness and optimization were ambiguous. | The objective is written explicitly. The paper says B2 inherits P's source-selected lambda, matches P's edge budget, is not optimized separately, and BCE is not wall-clock-compute matched to ranked arms. |
| Architecture and artifact provenance were insufficiently discoverable. | The Res2TCNGuard paper is cited; repository, immutable tag, protocol, compact ledgers, licensing boundary, and target-tuning prohibition are named. |
| Figure terminology retained an obsolete claim. | Rendering revision 003 changes only `content-aligned` labels to `same-item` and neutralizes the contrast title. It reuses the exact sealed inputs; source/PDF/PNG hashes are recorded. |

## Residual limitations to retain

- The result is weak in absolute terms (P: 47.07% SONAR and 42.98% ArAD EER).
- It concerns one ODSS source, one fresh 172k-parameter architecture, two fixed
  targets, and the locked four-seed probability ensemble.
- It cannot isolate a linguistic-content mechanism, establish SOTA, or support
  arbitrary-corpus or deployment claims.
- The target panel was held out from H9 selection but is not described as
  blind because of the documented historical checkpoint context.

No post-terminal retuning or new target experiment is justified by this review.
