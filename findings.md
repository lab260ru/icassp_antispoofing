# Research Findings

## Research Question

Which waveform-level signal cues associate causally, rather than merely
correlationally, with speech anti-spoofing detector scores across datasets and
architectures, and can the resulting evidence improve robustness?

## Current Understanding

No empirical claim has been established yet. The initial study separates three
questions that are often conflated: a feature may distinguish labels, correlate
with a detector score, or causally change a detector score under a matched
transformation. Only the last supports a shortcut-sensitivity claim.

## Key Results

- Input-integrity validation: all 71,237 published Spectra-AASIST scores joined
  exactly to ASVspoof 2019 LA labels. The stored score is increasing bonafide
  evidence, so analysis uses its negation as spoof evidence. This establishes a
  valid artifact interface, not a feature or causal result.
- Feature-pipeline validation: the v1_28 registry extracted complete
  non-voice descriptors and 95% voice-quality coverage over 200 balanced
  ASVspoof 2019 LA samples and three views. Missing unvoiced measures will be
  modeled explicitly rather than filled with zeros.

## Patterns and Insights

The study is designed to prevent label, corpus, duration, loudness, codec, and
speaker effects from being mistaken for detector reliance. Existing Arena score
artifacts provide broad architecture coverage without spending GPU time on
baseline reproduction.

## Lessons and Constraints

- Existing Arena outputs are trusted artifacts; do not rerun published baseline
  inference merely to reproduce them.
- All material claims require a protocol, pinned inputs, and an analysis note.
- The first draft must not contain invented or placeholder numerical findings.

## Open Questions

- Which public score artifacts expose stable sample IDs that can join audio
  manifests without heuristic matching?
- Which candidates survive held-out confirmation and quality-gated interventions?
- Is the strongest paper a causal-audit/fix story, an architecture-sensitivity
  atlas, a benchmark-artifact audit, or a principled negative result?

## Optimization Trajectory

| Run | Hypothesis | Metric | Status |
|---|---|---|---|
| bootstrap | setup | n/a | active |
