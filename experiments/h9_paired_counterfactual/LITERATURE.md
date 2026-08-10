# H9-PCR related-work and claim ledger

This is a scoped literature check for the active H9 protocol, not a systematic
review and not a priority claim.

## Closest evidence

| Source | Relevance to H9 | Boundary for this paper |
| --- | --- | --- |
| Yaroshchuk et al., ODSS, WIFS 2023, [DOI](https://doi.org/10.1109/WIFS58808.2023.10374863) | Establishes the ODSS natural/TTS counterpart structure and the VCTK unmatched-VITS caveat. | H9 must prove its own pairing and exclude unmatched items for every condition. |
| Xie et al., Interspeech 2021, [paper](https://www.isca-archive.org/interspeech_2021/xie21_interspeech.html) | Siamese anti-spoofing learns pair class agreement. | Different from real--TTS same-text ranking; do not call H9 the first pairwise anti-spoof method. |
| Tran et al., Interspeech 2024, [paper](https://www.isca-archive.org/interspeech_2024/tran24_interspeech.pdf) | Cross-entropy plus supervised contrastive anti-spoofing. | Motivates the random-pair margin control: pairwise regularization alone is not H9's contribution. |
| Kim et al., Interspeech 2025, [paper](https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.html) | Pair-aware anti-spoof objective that aligns augmented bona-fide pairs and disperses fakes. | Closest pair-aware contrast; its augmentation pairs differ materially from ODSS real--TTS pairs. |
| Wang et al., ACML 2025, [paper](https://proceedings.mlr.press/v260/wang25g.html) | Model-agnostic inter-instance compactness/separation. | Confirms that ordinary inter-instance objective changes are not novel by themselves. |
| Yin and Zhao, 2026, [arXiv](https://arxiv.org/abs/2606.02980) | Combines classification and pairwise ranking for anti-spoofing. | Makes B2 mandatory: a result must attribute any gain to **content alignment**, not ranking alone. |
| Paired differential-detection study, 2025, [record](https://invenio.nusl.cz/record/681406) | A trusted-reference test-time paired setting reports modest matching benefit. | It is not H9's training-only deployment setting, but it is a caution against assuming a large effect. |

## Claims allowed if the locked gate passes

> On ODSS, content-aligned natural--TTS pair-ranking supervision improved
> external SONAR and ArAD EER over equal-data BCE and random opposite-class
> ranking for a fixed, fresh-initialized compact Res2TCNGuard architecture.

## Claims prohibited even if it passes

- "first" paired/contrastive/ranking anti-spoofing method;
- causal removal of content or speaker representations;
- universal unseen-generator robustness, model-agnostic benefit, or SOTA;
- a blind target-evaluation claim (historical checkpoint benchmark rows were
  encountered in the architecture-provenance audit; the checkpoint is excluded
  and fresh H9 predictions remain metric-fenced);
- a claim that audio fingerprints prove absence of all data lineage overlap.

## Required reporting if H9 runs

1. Count and hash all matched and excluded ODSS rows; B1/B2/P share the
   paired-eligible source pool.
2. Report P versus B2 pairing marginals (language, source corpus, generator,
   voice, duration, pair count) and reject mismatch.
3. Report each target, each seed, and both comparator signs; a macro alone is
   insufficient.
4. Separate trial-bootstrap uncertainty from seed-to-seed variation.
5. State source, architecture, fresh initialization, and fixed target roles
   precisely in the paper.
