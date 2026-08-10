# H8-SF analysis 001 — robust rank-space fusion is not the paper result

**Status:** terminal, failed positive-result candidate. The run is preserved as
a transparent baseline/falsification artifact; it does not authorize fuser
tuning, a model-subset change, a target change, or a paper claim.

## Sealed inputs and firewall

- Source freeze v2: 57,355 balanced source trials; manifest SHA-256
  `e3fc7e000effba8ae8c76b7539b5593063373899c5b512baa87669877a50539a`.
- Label-free target rank/probit panel: 625,021 exact common trials across
  CFAD, CVoiceFake_small, DECRO, LibriSeVoc, and XMAD; SHA-256
  `e5fc34c692832726a0b5917e5342b18b54ae2665456af7339855dccfced7f2db`.
- Source-only fit: target-label and target-metric flags are both `false`; its
  target-prediction SHA-256 is
  `27e05d89b43f65b7b72127e0f4fd8a21a77bf9ae22970a9f86dc843b152c8501`.
- Final target labels were downloaded only when the committed evaluator first
  required them and exactly matched the five Arena-pinned label SHA-256 values.
  All 625,021 predictions joined one-to-one to those labels.

The first attempted evaluation halted before any target label value was read
because the five pinned label files were absent locally. Only those small label
files were then downloaded to the HDD; their hashes were checked before the
unchanged evaluator reran. The label-free target materializer previously halted
on CVoiceFake_small because generic `Path.stem` collapsed literal dotted IDs.
The v2 terminal-audio-suffix normalizer and a new v2 source freeze were
committed before the successful materialization. No model, target, loss,
hyperparameter, or score result was used to make either integrity correction.

## Frozen source-only choices

- B0: Spectra-AASIST selected by source corpus-macro EER.
- B1: equal rank/probit mean.
- B2: corpus × class-balanced nonnegative ERM (`L2=1e-2`).
- P: GroupDRO with source-LOO-selected `L2=1e-3`, `eta=0.01`.
- A1: V-REx with source-LOO-selected `L2=1e-4`, `lambda=0.1`.

## Exhaustive blind target outcome

| Method | Mean EER (%) | Worst EER (%) |
| --- | ---: | ---: |
| B0: Spectra-AASIST | **0.185** | **0.481** |
| B1: equal rank mean | 19.936 | 30.117 |
| B2: balanced ERM | 11.911 | 19.553 |
| P: GroupDRO | 9.309 | 15.010 |
| A1: V-REx | 8.975 | 14.363 |

P improves over the two weaker fusion comparators on every target: its mean
EER reduction is 53.30% versus B1 and 21.84% versus B2. The fixed bootstrap
mean EER difference `P - B2` is -2.551 percentage points (95% CI
[-2.671, -2.434]). These are real measurements but are not sufficient for a
strong result because B0 is dramatically better.

## Predeclared falsification and disposition

The H8 plan's B0 row states its falsification condition as: *any fusion cannot
beat this macro EER*. P’s 9.309% mean EER does not beat B0’s 0.185% (and its
15.010% worst EER does not beat B0’s 0.481%). This condition is false by a
large margin, so H8-SF fails as a positive paper candidate.

The generated `h8_decision_gate.json` reports `positive_result_gate_passed:
true` because its five literal numbered checks compare P only with B1/B2. That
mechanical Boolean is **not** the authoritative scientific conclusion: it
omits the already-written B0 falsification cell from the methods table. This
is a protocol-consistency defect discovered after final evaluation, not a
license to amend the gate or rerun the target analysis. The present result note
applies the more conservative predeclared constraint and closes H8-SF.

## Output identities

The final HDD output directory is
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_target_evaluation_001/`.

| Output | SHA-256 |
| --- | --- |
| `h8_target_metrics.csv` | `9b18d14d5b99ed3af533acca022f27b0a73385e75d42d844859cf110417e45d4` |
| `h8_bootstrap_panel.parquet` | `a13b311074ce9f1de40ae0faf0a10a102126b829f0b5a9fe9057e9756d79f9da` |
| `h8_bootstrap_mean_eer_differences.csv` | `f1f16a4f905c31b31d1b5ecb4e7d133a7e6afb36a39dae88a8642e923eb5c9b2` |
| `h8_decision_gate.json` | `0bc0d5c2b4ec3a00d637228171e21a2a5d780c6c404e8ea9302c4c411b1d2250` |
| `h8_target_evaluation_provenance.json` | `ac8d1493c671d9d63d5cae89059aee671c7881a12d9a33040b86b531293da60a` |

## Next boundary

Do not exclude Spectra-AASIST, reweight the panel, make a conditional gate,
change a rank mapping, add a loss, or move to another target under H8-SF. A
new hypothesis must first state why its available training data and model
provenance yield a meaningful independent baseline, then lock a new protocol
before its first result access.
