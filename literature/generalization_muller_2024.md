# Generalization is primarily distribution difference, not only hardness

## Verified source

Müller et al., *Harder or Different? Understanding Generalization of Audio
Deepfake Detection*, Interspeech 2024, pp. 2705--2709,
DOI: `10.21437/Interspeech.2024-247`.

- Official record: <https://www.isca-archive.org/interspeech_2024/muller24b_interspeech.html>
- Verified 2026-08-09 from the official ISCA Archive metadata and abstract.

## What the source establishes

- The paper decomposes in-domain/out-of-domain performance gaps into hardness
  and difference components.
- Its ASVspoof experiments attribute the performance gap primarily to the
  difference component, not hardness, and caution that merely increasing model
  capacity may not address the generalization problem.

## Paper implication

This is the closest high-level motivation for the present multi-corpus audit.
Our narrower contribution must be stated carefully: we test whether
interpretable waveform-feature/score relationships differ across architectures
and corpora, then test selected relationships by controlled transformation. We
do not claim to reproduce the paper's hardness/difference decomposition.

## Citation status

Verified for later bibliography use; not yet inserted into `paper/references.bib`.
