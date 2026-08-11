# H9-PCR literature audit — 2026-08-11

## Scope and conclusion

This is a bounded primary-source audit, not a systematic review or a priority
claim. It verifies that pairwise and contrastive anti-spoofing objectives
already exist. H9-PCR's narrower contribution is a controlled test of
*documented natural--synthetic counterpart* alignment: the same paired-eligible
ODSS pool is used for BCE (B1), content-aligned ranking (P), and an equal-edge
stratum-matched random-pair ranking control (B2), with source-only selection
and two fixed external targets.

All claims below were checked in official proceedings pages and linked official
PDFs on 2026-08-11. "Not reported" means only that the cited paper does not
report that design; it is not a claim about all prior literature.

## Direct comparison

| Source | Paired supervision | Explicit same-item natural/synthetic pair? | Equal-edge random-pair control? | Source-only selection + fixed held-out corpora? | H9 boundary |
| --- | --- | --- | --- | --- | --- |
| Yaroshchuk et al., WIFS 2023 (ODSS) | Dataset paper; documents natural and synthetic data construction, rather than a pairwise detector objective. | The published ODSS resource is the source for H9's independently materialized counterpart groups. H9 verifies its own groups and excludes unmatched VITS rows. | No detector comparison. | No detector-training protocol. | ODSS supplies the documented counterpart structure; it is not prior evidence that counterpart ranking improves transfer. |
| Xie et al., Interspeech 2021 | Siamese contrastive loss pulls same-class pairs together and pushes different-class pairs apart. Section 4 randomly selects 50 pairs from each balanced mini-batch. | No: pair identity is a class relation, not a documented natural/TTS counterpart. | No comparator that fixes edge count while randomizing pairing alignment is reported. | No: reports ASVspoof 2019 train/development/evaluation partitions, not a source-only checkpoint fence and two fixed external corpora. | Pairwise anti-spoofing is not new; H9 differs in the supervision relation and attribution control. |
| Tran et al., Interspeech 2024 | Cross-entropy plus supervised contrastive loss; positive pairs share a class label and negatives are other-class samples. | No: class-labelled pairs, not a natural/synthetic counterpart. | No equal-budget random-pair ranking comparator is reported. | The work trains on ASVspoof 2019 LA and reports 2019 LA plus 2021 LA/DF, without H9's predeclared target fence or source-only checkpoint ledger. | H9 must not attribute gains to ranking alone; P versus B2 is essential. |
| Kim et al., Interspeech 2025 | Bonafide-Pair Learning aligns two *augmentations* of the same bonafide waveform; extended one-class softmax separates spoofed samples. | No: same-utterance augmentation within the bonafide class, not a natural/synthetic counterpart. | No equal-edge random-pair control is reported. | Uses ASVspoof 2019 LA train/development/evaluation and additionally tests 2021 LA/DF; no source-only selection lock plus two pre-fixed external corpora is reported. | Closest positive-pair precedent. Explicitly distinguish it and avoid "first pair-aware" wording. |

## Primary-source evidence

- **ODSS.** The [IEEE WIFS paper](https://doi.org/10.1109/WIFS58808.2023.10374863)
  and the [pinned ODSS record](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ODSS)
  establish the public source dataset. H9's own hash-bound materialization,
  not a citation alone, establishes the exact 7,961 documented groups used.
- **Xie.** The [ISCA record](https://www.isca-archive.org/interspeech_2021/xie21_interspeech.html)
  and [PDF](https://www.isca-archive.org/interspeech_2021/xie21_interspeech.pdf)
  state that pairs are judged by whether samples have the same category;
  Sections 3.1 and 4 describe random selection and balanced mini-batches.
- **Tran.** The [ISCA record](https://www.isca-archive.org/interspeech_2024/tran24_interspeech.html)
  and [PDF](https://www.isca-archive.org/interspeech_2024/tran24_interspeech.pdf)
  define supervised contrastive positives by the same-class indicator
  (Section 2.4) and train on ASVspoof 2019 LA (Section 3).
- **Kim.** The [ISCA record](https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.html)
  and [PDF](https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.pdf)
  state that BPL aligns embeddings of augmented bonafide pairs; Section 3.1
  creates two augmentations from one selected bonafide waveform.

## Ready-to-use bibliography

`yaroshchuk2023odss`, `xie2021siamese`, and `tran2024contrastive` are already
verified in `paper/references.bib`. The following fully verified entry can be
added if the related-work paragraph cites Kim:

```bibtex
@inproceedings{kim2025bpl,
  title     = {Enhancing Audio Deepfake Detection by Improving Representation Similarity of Bonafide Speech},
  author    = {Kim, Seung-bin and Shin, Hyun-seo and Heo, Jungwoo and Lim, Chan-yeong and Koo, Kyo-Won and Son, Jisoo and Hong, Sanghyun and Jung, Souhwan and Yu, Ha-Jin},
  booktitle = {Interspeech 2025},
  year      = {2025},
  pages     = {2250--2254},
  doi       = {10.21437/Interspeech.2025-422},
  url       = {https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.html}
}
```

## Paper-safe paragraph and claim boundary

> Pairwise and contrastive supervision is already established in speech
> anti-spoofing. Xie *et al.* form randomly selected same-/different-class
> Siamese pairs, Tran *et al.* use class-supervised contrastive learning with
> cross-entropy, and Kim *et al.* align two augmentations of a bonafide
> utterance. Our question is narrower: whether a *documented natural--synthetic
> counterpart* relation improves transfer beyond the same paired-eligible data
> used with BCE and beyond an equal-edge random-pair ranking control. We select
> the ranking weight only on a source voice-disjoint split and evaluate frozen
> systems on two external corpora.

Use: **"For the fixed compact Res2TCNGuard and ODSS protocol, documented
content-aligned counterpart ranking improved transfer over same-pool BCE and
an equal-edge random-pair ranking control on two predeclared targets."**

Do not use: "first pairwise/contrastive/ranking anti-spoofing method," "first
content-aware anti-spoofing method," or language implying this bounded audit
proves no similar work exists.
