# H9-PCR terminal evaluation 001

**Status:** complete, confirmatory positive result under the locked H9-PCR
protocol. This note is the authoritative compact interpretation of the
terminal target evaluation.

## Frozen study chain

- Source: ODSS revision `1968e6d0ef141c4572073695bdc1d17a8706177f`;
  23,883 paired-eligible trials from 7,961 documented natural/VITS/
  FastPitch--HiFi-GAN groups. B1, B2, and P use this same pool. P and B2 each
  use exactly 15,922 frozen rank edges.
- Model: public 172,102-parameter Res2TCNGuard architecture, freshly
  initialized for every run; the historical pretrained checkpoint was never
  loaded. Four fixed BF16 seeds use batch 24, four workers, six maximum epochs,
  and the source-dev checkpoint rule. The source-only selection fixed rank
  weight 1.00 for P and B2.
- Controls: B1 is BCE; B2 adds the same number of language/corpus/generator-
  stratified random natural--spoof rank edges; P instead uses documented
  content-aligned natural--spoof edges.
- Terminal targets: SONAR revision
  `eca7c72ebdf0f7936a644605a56735ac8564dbd9` (3,948 trials) and ArAD revision
  `350184966eeb5b46ff2acdabd8f4d12e41e582da` (3,570 trials). They were
  materialized only after the complete source checkpoint ledger was sealed.

The evaluator byte-validated all target payloads, found **zero** exact
canonical source--target waveform-fingerprint collisions, wrote the complete
raw prediction table before opening the two separate target-label tables, and
then evaluated all 12 fixed method--seed checkpoints together on CUDA BF16.

## Terminal result

The primary score is unweighted mean EER over SONAR and ArAD after averaging
spoof probability over the four frozen seeds. Lower is better.

| Method | SONAR EER | ArAD EER | Two-target macro EER |
| --- | ---: | ---: | ---: |
| B1: same-pool BCE | 58.88% | 44.69% | 51.78% |
| B2: BCE + random-pair ranking | 56.09% | 47.93% | 52.01% |
| **P: BCE + content-aligned ranking** | **47.07%** | **42.98%** | **45.02%** |

P lowers macro EER by **13.05% relative** to B1 and **13.44% relative** to
B2. It has lower ensemble EER than both controls on both fixed targets.

The predeclared shared sample-ID, label-stratified 2,000-replicate bootstrap
(seed 2909) reports these macro-EER differences in percentage points:

| Contrast | Mean | 95% percentile CI |
| --- | ---: | ---: |
| P - B1 | -6.70 | [-8.25, -5.25] |
| P - B2 | -6.97 | [-8.54, -5.30] |

Both intervals are wholly below zero. The evaluator's complete gate therefore
passes all eight locked checks: both relative-reduction rules, both-target
point-estimate rules, both bootstrap rules, zero collision, and prediction /
manifest reconstruction.

## Seed disclosure and interpretation

The terminal decision is intentionally based on the predeclared four-seed
probability ensemble, not cherry-picked individual seeds. Individual seed
macro EERs are heterogeneous (B1/B2/P, respectively): seed 9101
50.00/53.05/53.30%, 9102 50.59/50.67/49.26%, 9103 53.78/54.47/45.92%, and
9104 46.26/47.58/41.10%. This is a robustness limitation, not a reason to
alter the frozen ensemble, checkpoint, target, or loss rule after observing
the terminal panel.

The supported claim is narrow: with this compact, fresh-initialized model and
ODSS source protocol, documented content-aligned pair-ranking supervision
improved transfer relative to both matched-only BCE and an equal-budget random
ranking control on the two predeclared external target corpora. The result is
not state-of-the-art, does not identify a causal internal representation
mechanism, and does not establish performance beyond these targets.

## Artifact ledger

Large outputs are on the HDD at
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_terminal_evaluation_001/`.

| Artifact | SHA-256 |
| --- | --- |
| frozen source checkpoint ledger | `3e0530e2b850cbb08329d14a8982a7448a319a4a30e75ee2d51dc4ecb7f2325d` |
| raw predictions (written before labels) | `f992baba8498ad7fae2a0c4c67073e6554ae073d5d189e387095295ead6965a1` |
| terminal metrics | `2e744154d25b3f515d1b0121db7dc8658cd9b4fb5ff4a9a45dca79b1d8f35ee4` |
| per-seed metrics | `06f14ff72d7a1e23e4447eb35fde838543218a0f2f1c644c6a556696152b8b8c` |
| bootstrap differences | `771a4f3b6b794503b725e447dcd655cde65a3a2dbdedc851ee55fa71c938e9c0` |
| terminal decision gate | `a4e5f118cd34af6ae2e728e094d2ad9e9e6ebf1a8ebe1fb2d841b191c3753852` |
| terminal provenance | `af0b85c8f05ad9f2b70cad83b45db7922cd9422732c4630dbb17d1ad2821d2f4` |

The final inspected display-only result figure is
`results/h9_terminal_evaluation_001/figures_terminal_002/h9_pcr_terminal_eer.pdf`
(PDF SHA-256 `165e0ad8e922878568d3d6033fd4b18076c10471808effc3b4211ebea57eda7b`)
and its 300-DPI PNG has SHA-256
`2bcae387a963f3144195c7351e19d50d7987fd0a297f8e9fc5e7b86e2211df16`.
It is bound to the four terminal input hashes in
`H9_RESULT_FIGURE_PROTOCOL.md`. The earlier `figures_terminal_001` layout is
preserved as a non-authoritative visual iteration; it has identical scientific
inputs and encoding but an overlapping legend/annotation layout.

## Independent review

The Codex-only red-team audit in
`reviews/H9_TERMINAL_RED_TEAM_AUDIT_20260811.md` independently replayed the
source-ledger and terminal-artifact hashes and accepted the controlled transfer
result with the limitations stated above. It is an internal technical audit,
not external peer review.
