# Literature survey

Each paper receives its own note with a verified URL, metadata source, claim
used, and relevance. Citations are added to the manuscript only after two-source
verification and programmatic BibTeX retrieval.

## Seed questions

- What signal shortcuts have been documented in spoofing/deepfake detection?
- How are F0, phase, silence, dynamic range, and codec effects used or audited?
- Which prior studies distinguish correlation from causal waveform intervention?
- Which cross-dataset anti-spoofing evaluations are most relevant to this study?
## Signal-cue audit positioning

## Verified anchors

| Source | Relevant verified point | Consequence for this study |
|---|---|---|
| Müller et al. (ASVspoof 2021) | Silence duration can be label-correlated and strongly discriminative. | Separate label separability, score association, and causal intervention. |
| Shim et al. (Interspeech 2023) | Controlled shortcut conditions can yield misleadingly strong or worse-than-chance countermeasures. | Treat shortcut claims as causal and quality-gated, not correlational. |
| Müller et al. (Interspeech 2024) | ASVspoof generalization gaps are primarily a difference rather than hardness component. | Use multi-corpus replication; a one-corpus score-feature result is not portable. |
| Pascu et al. (Interspeech 2024) | Frozen SSL representations and a simple classifier can generalize/calibrate strongly in their benchmark. | Motivate, but do not prejudge, the conditional low-parameter H3 mitigation. |
| Pan et al. (Interspeech 2022) | Environmental classification clues can be attackable; spectral representation may improve noise robustness. | Test cue changes with paired waveform transformations and negative controls. |
| Yamagishi et al. (ASVspoof 2021) | 2021 adds channel/compression variability and unmatched evaluation conditions. | Keep 2019 LA, 2021 LA, and 2021 DF separate until explicit cross-corpus aggregation. |

## Positioning guardrails

- The paper is **not** a claim that crest factor is a universal shortcut. Its
  observed 2019/2021-LA association is heterogeneous and discovery-only.
- The paper's prospective novelty is the evidence chain: immutable published
  per-sample score artifacts, frozen waveform features, within-class and
  adjusted screen, held-out portability, parity-validated paired interventions,
  then only conditional mitigation.
- Existing robustness/generalization work motivates broad evaluation; it does
  not remove the need for the present model-specific parity and quality gates.

## Per-paper notes

- `asvspoof2021.md`
- `generalization_muller_2024.md`
- `calibrated_ssl_pascu_2024.md`
- `robust_features_pan_2022.md`
