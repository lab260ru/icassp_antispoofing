# H9-PCR experiment plan — content-aligned pair ranking

**Status:** locked before H9 source/target data materialization, model fitting,
or target result access.

## Question

Can content-aligned natural-to-synthetic ranking supervision reduce cross-corpus
speech anti-spoofing error beyond both ordinary BCE and an equal-budget random
opposite-class ranking control?

## Fixed model and optimization envelope

All methods train the same public, locally runnable **Res2TCNGuard** PyTorch
architecture from a fresh seed-specific initialization. The public `_net.py`
architecture is compact (172,102 parameters); the bundled
`best_1.495.pth` checkpoint is deliberately not loaded because its declared
ASVspoof2019-LA pretraining would make ODSS no longer the sole training source
and its model card exposes historical benchmark rows. The architecture-file
hash and initialization rule are frozen in the source manifest.

Every method uses only the frozen **paired-eligible source pool**: complete
groups with one documented natural clip and at least one documented synthetic
re-synthesis. Unmatched source items are excluded from B1 as well as B2/P, so
the primary comparison does not conflate pair availability with the loss.

- Input: mono 16-kHz waveform, deterministic fixed-length policy shared by all
  conditions.
- Precision: CUDA BF16 autocast with a finite-loss guard. No FP32 training
  fallback is allowed.
- Splits: a `voice_key=(source_corpus,speaker)` is wholly assigned to train or
  dev under a deterministic language-stratified split, so a voice never crosses
  the source development boundary. The pair key is a documented
  source-corpus/speaker/utterance identifier after removing the generator
  prefix. The manifest reports its content-key overlap diagnostic but does not
  claim text-disjoint development without transcript-level evidence. The split
  seed is `2909`.
- Batch construction: class-balanced, language-balanced where metadata
  permits, fixed source train groups. The same batch schedule and augmentations
  are shared by B1, B2, and P.
- Four independent seeds: `9101`, `9102`, `9103`, `9104`, one fixed GPU per
  seed at final fit. Each seeds every initialization and data/random-pair
  schedule. A batch-size throughput probe selects one safe batch size
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
| B1 | Class-balanced BCE | None, on the same paired-eligible source pool. | Required ordinary label-training baseline. |
| B2 | BCE + random opposite-class rank term | A deterministic, label-valid but content-unmatched bonafide partner for every spoof pair; random pairing is stratified by language, source corpus, and spoof generator; same term count and selected lambda as P. | Controls for pairwise-margin regularization itself without a domain-composition confound. |
| **P** | **BCE + content-aligned rank term** | Natural and spoof clips in the same documented source-text group; same term count and selected lambda as B2. | Primary method: test whether documented content alignment helps beyond an equally sized random margin term. |

No GroupDRO, codec augmentation, architecture change, target normalization,
or ensemble is part of H9-PCR. Those would confound the paired-supervision
test and require a new protocol.

The source manifest must emit counts and hashes for all exclusions and a B2/P
marginal distribution table over language, source corpus, spoof generator,
voice, duration, and pair count. A mismatch in the locked strata or term count
is a run failure, not a reason to alter the random-pair rule.

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

A passing H9 result supports a restricted claim that content-aligned pair
supervision improved transfer of this transparent compact model
to the two predeclared external corpora under this protocol. It does not prove
why an internal representation changed, establish a causal content/speaker
mechanism, certify arbitrary future generators,
or establish a comparison with proprietary / independently trained systems.
