# H8 deep research — corpus-robust frozen-system fusion

## Gap and decision

Vanilla score averaging or a target-tuned logistic stacker is not a credible
paper contribution: anti-spoofing fusion, calibration, and SSL-CM fusion are
already established. H8 tests a narrower gap: **can a single source-only,
domain-robust rank-space fuser improve both average and worst external-corpus
error over ordinary fusion without any target labels?** The method is useful
only if it clears the locked external-target gate.

## Primary sources

| Source | Finding relevant to H8 | Consequence |
| --- | --- | --- |
| Serrano et al. (2025), *Improving OOD Audio Deepfake Detection via Layer Selection and Fusion of SSL-Based CMs*, [arXiv:2509.12003](https://arxiv.org/abs/2509.12003) | Fusion can help OOD audio deepfake detection, but a logistic fuser can lose to calibrated-score summation on a different target. | H8 must beat both learned and fixed fusion across several targets; ordinary logistic fusion is only a baseline. |
| Wang et al. (2024), *Revisiting and Improving Scoring Fusion for SASV*, [Interspeech DOI](https://doi.org/10.21437/Interspeech.2024-422), [arXiv:2406.10836](https://arxiv.org/abs/2406.10836) | Calibration materially affects fusion. | Calibrated/rank-normalized uniform fusion is mandatory, not an afterthought. |
| Zhang et al. (2022), *Deepfake Detection System Based on Score Fusion*, [DOI](https://doi.org/10.1145/3552466.3556528), [arXiv:2210.06818](https://arxiv.org/abs/2210.06818) | OOD fusion can fail when score scale and weights are unreliable. | Motivate monotone rank mapping and source-domain robustness rather than raw-score stacking. |
| Sagawa et al. (2020), *Distributionally Robust Neural Networks for Group Shifts*, [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) | GroupDRO minimizes high-loss group risk but needs controlled regularization. | Use source corpus × class groups, matched regularization, and source-only inner validation. |
| Müller et al. (2024), *Harder or Different?*, [arXiv:2406.03512](https://arxiv.org/abs/2406.03512) | Cross-dataset audio deepfake degradation is primarily distributional difference. | Primary endpoint includes worst-target EER, not pooled accuracy. |
| Pascu et al. (2024), *A Generalizable Approach to Speech Deepfake Detection*, [ISCA](https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html) | Strong multi-dataset SSL generalization/calibration is possible. | Frozen-system fusion is a fast H8 screen; a frozen-SSL GroupDRO probe is the predeclared fallback only if H8-SF fails. |
| Dowerah et al. (2026), *Speech Deepfake Arena*, [DOI](https://doi.org/10.1109/OJSP.2026.3652496) | Broad, reproducible released-score benchmarking exposes substantial model/corpus heterogeneity. | H8 uses the Arena-like score interface, but must disclose that it evaluates frozen published systems rather than retraining them. |

## Open-source implementation anchors

- [WILDS GroupDRO](https://github.com/p-lambda/wilds/blob/main/examples/algorithms/groupDRO.py)
  (MIT): simple adversarial source-group weighting; reimplement the small
  algorithm rather than import a heavyweight framework.
- [original group-DRO](https://github.com/kohpangwei/group_DRO/blob/master/loss.py)
  (MIT): algorithmic reference for adversarial group weights/EMA; do not reuse
  its obsolete runtime.
- [DomainBed](https://github.com/facebookresearch/DomainBed) (MIT, archived):
  methods/reference checklist only; its target-data Oracle selection is
  explicitly forbidden for H8.
- [nonparametric speaker-verification normalization](https://www.sri.com/publication/speech-natural-language-pubs/nonparametric-feature-normalization-for-svm-based-speaker-verification/):
  supports the monotone empirical-CDF score mapping. H8 names target-CDF use
  transductive, label-free adaptation rather than source-only generalization.

## Candidate levers

| Angle | Candidate | Mechanism | Status |
| --- | --- | --- | --- |
| C | Empirical-CDF/probit fusion features | Remove arbitrary score scale while retaining each expert's ordering. | Selected for H8-SF. |
| B/E | corpus × class GroupDRO, nonnegative L2 fuser | Prevent a source-specialist detector/corpus from dominating loss. | Selected primary method. |
| B/E | pooled, corpus-balanced logistic fuser | Establish whether DRO itself improves over matched ordinary learning. | Mandatory baseline. |
| I | uniform rank fusion | Establish whether learned weights beat robust fixed aggregation. | Mandatory baseline. |
| G | V-REx loss-variance fuser | Alternative domain-generalization mechanism. | Ablation only if H8-SF primary passes source validation. |
| J | disagreement-based abstention | High expert IQR may mark unreliable cases. | Separate, predeclared secondary analysis; never converts a rejected sample into a claimed EER win. |
| K | parallel target/seed evaluation | Exhaustive, fixed target evaluation and three source-training seeds are cheap. | Required verification, not method novelty. |

## Falsifiable decision

If the predeclared GroupDRO rank fuser fails to improve over both uniform rank
fusion and corpus-balanced ERM fusion on the locked blind-target gate, H8-SF is
not a paper direction. The fallback is a new frozen-SSL linear-probe protocol;
do not optimize fusion after target inspection.
