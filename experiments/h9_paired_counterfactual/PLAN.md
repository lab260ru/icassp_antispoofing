# H9-PCR experiment plan — paired counterfactual ranking

**Status:** locked before H9 source/target data materialization, model fitting,
or target result access.

## Question

Can content-matched natural-to-synthetic ranking supervision reduce cross-corpus
speech anti-spoofing error beyond both ordinary BCE and an equal-budget random
opposite-class ranking control?

## Fixed model and optimization envelope

All methods fine-tune the same public, locally runnable **Res2TCNGuard**
PyTorch implementation from the same pinned checkpoint. It is a transparent,
compact (172,102-parameter) detector; its existing source-PyTorch inference
path has a documented baseline-parity check, but that check is not an H9
result.

- Input: mono 16-kHz waveform, deterministic fixed-length policy shared by all
  conditions.
- Precision: CUDA BF16 autocast with a finite-loss guard. No FP32 training
  fallback is allowed.
- Splits: H9 source groups are disjoint by the conservative
  `(speaker_or_voice_key, content_key)` grouping rule. A group never crosses
  train/dev. The deterministic split seed is `2909`.
- Batch construction: class-balanced, language-balanced where metadata
  permits, fixed source train groups. The same batch schedule and augmentations
  are shared by B1, B2, and P.
- Four independent seeds: `9101`, `9102`, `9103`, `9104`, one fixed GPU per
  seed at final fit. A batch-size throughput probe selects one safe batch size
  before final runs; the selected size is then shared by all conditions.
- Source-only loss-weight selection: choose `lambda` from `{0.10, 0.30, 1.00}`
  on the group-disjoint ODSS development split by mean EER over the four
  predeclared seeds. Tie-break lower lambda. The selected lambda applies to P
  and B2. Target data cannot enter this choice.
- Source-only stopping: fixed maximum epoch budget and a source-dev EER
  checkpoint rule, shared across conditions. The exact epoch count and
  checkpoint tie rule are written before the first training run; target data
  cannot select a checkpoint.

## Methods

Let `z(x)` be a spoof-evidence logit. Every condition uses identical BCE
training samples and objective scaling. A pair term is

`max(0, 1 - [z(x_spoof) - z(x_bonafide)])`.

| ID | Method | Difference from B1 | Purpose |
| --- | --- | --- | --- |
| B1 | Class-balanced BCE | None | Required ordinary label-training baseline. |
| B2 | BCE + random opposite-class rank term | A deterministic, label-valid but content-unmatched bonafide partner for every spoof pair; same term count and selected lambda as P. | Controls for pairwise-margin regularization itself. |
| **P** | **BCE + paired counterfactual rank term** | Natural and spoof clips in the same frozen content group; same term count and selected lambda as B2. | Primary method: hold content fixed while ranking synthesis evidence. |

No GroupDRO, codec augmentation, architecture change, target normalization,
or ensemble is part of H9-PCR. Those would confound the paired-supervision
test and require a new protocol.

## Source artifacts and target firewall

1. Build and commit a source-only manifest containing the ODSS revision,
   complete group/pair assignment, source hashes, split assignment, and the
   target-free lambda/checkpoint selection result.
2. Run all 12 final condition×seed source fits using that selection. Source
   development results may be read only to choose the predeclared lambda and
   source checkpoint.
3. Only after source fits are frozen and their checkpoint hashes written, build
   target prediction manifests and run every method/seed on both targets.
4. The terminal evaluator may then read target labels once to emit all
   predeclared metrics and paired uncertainty. It has no flag for model or
   hyperparameter selection.

## Metrics and decision gate

For every target and every method, report EER, AUROC, number of retained
trials, orientation provenance, and all four seed predictions. The H9 primary
endpoint is the unweighted two-target macro EER, computed from the mean
probability over the four frozen seeds. EER is lower-is-better.

The fixed uncertainty panel is all retained target trials. Use 2,000
stratified sample-ID bootstrap replicates, seed `2909`, shared resample indices
across P/B1/B2, and report the macro-EER difference `P - comparator`. If a
target lacks sample-ID uniqueness, hard-stop rather than silently resampling.

H9-PCR is a positive result only when all conditions hold:

1. P has at least **10% relative lower** two-target macro EER than B1 **and**
   B2;
2. P has lower point-estimate EER than B1 and B2 on **both** SONAR and ArAD;
3. both fixed 95% bootstrap CIs for macro-EER difference (P minus B1 and P
   minus B2) lie below zero;
4. no exact source--target canonical-audio fingerprint collision exists; and
5. the source/target manifests, output hashes, and fixed prediction
   reconstruction audit pass.

If any condition fails, H9-PCR stops. A weaker source result, one-target
improvement, or a different target/model/loss is not a positive result.

## Claim boundary

A passing H9 result supports a restricted claim that matched content
counterfactual supervision improved transfer of this transparent compact model
to the two predeclared external corpora under this protocol. It does not prove
why an internal representation changed, certify arbitrary future generators,
or establish a comparison with proprietary / independently trained systems.
