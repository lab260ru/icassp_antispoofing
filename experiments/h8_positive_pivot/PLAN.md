# H8-SF experiment plan — source-only corpus-robust copula fusion

**Status:** protocol locked before target score/label acquisition or any H8
metric computation. This is a fresh study; H1--H7 results cannot select an H8
model, feature, target, or hyperparameter.

## Question

Can a single nonnegative, rank-space GroupDRO fuser of eight frozen
anti-spoofing systems reduce both corpus-macro and worst-corpus EER on a blind
five-corpus external panel relative to fixed and ordinary learned fusion?

## Data roles and firewall

| Role | Corpora | Permitted before final evaluation |
| --- | --- | --- |
| Development source domains | ASVspoof2019_LA, ASVspoof2021_LA, ASVspoof2021_DF | labels and scores for new H8 source manifest, inner corpus leave-one-out tuning, and training |
| Blind primary targets | CFAD, CVoiceFake_small, DECRO, LibriSeVoc, XMAD | metadata/revision/score-file integrity only; labels are inaccessible to training, tuning, model/roster selection, and target-CDF computation |
| Excluded from H8 primary endpoint | ASVspoof5, InTheWild, DeepVoice | previously examined in H1--H7; retain only for future explicitly secondary retrospective checks |

The fuser roster is fixed before score acquisition: **Spectra-AASIST, AASIST,
Res2TCNGuard, RawTFNet, WhisperMFCCMesoNet, W2V2-AASIST, XLSR-SLS, RawBMamba**.
Every retained row must have all eight aligned sample IDs and finite raw scores.
No model subset is selected after an H8 result. Per-model orientation is learned
from sources only and then frozen. The target empirical CDF may use the entire
unlabeled target score batch; this is explicitly a transductive label-free
adaptation operation, not source-only deployment.

## Frozen representation and source sampling

For each corpus and model, convert the oriented spoof score to
`Phi^-1(clip((rank - 0.5) / n, 1e-4, 1-1e-4))`, where rank is computed within
that corpus using labels nowhere. This monotone rank/probit map is the H8
copula feature representation.

From each source corpus × class, freeze up to 10,000 stable IDs using
`sha256('H8SF|2608|dataset|label|sample_id')` order. A corpus/class below the
cap is retained whole; a missing class or score/model coverage fails the run.
Training batches contain equal examples from the six source corpus × class
groups, preventing ASVspoof2021_DF class imbalance from defining the loss.

## Methods

All models use the same eight rank/probit features, no corpus identity input,
and a scalar spoof logit. Bias is allowed; fusion weights are nonnegative.

| ID | Angle | Method | Mechanism | Expected move | Falsification |
| --- | --- | --- | --- | --- | --- |
| B0 | I | Source-selected single expert | Establish a non-fusion deployment baseline selected by source corpus-macro EER only. | no general guarantee | Any fusion cannot beat this on macro EER. |
| B1 | I/C | Uniform rank mean | Monotone mapping removes scale mismatch; equal aggregation exploits fixed roster diversity without learned weights. | improve scale-sensitive raw fusion | It is no better than B0. |
| B2 | B/E | Corpus × class-balanced ERM nonnegative L2 logistic fuser | Learn global complementary weights while equal group sampling resists source-size imbalance. | improve B1 macro EER | It is no better than B1. |
| **P** | B/E/C | **Corpus × class GroupDRO nonnegative L2 logistic fuser** | High-loss source groups increase adversarial weight, preventing one source-specialist expert from dominating; rank features preserve monotone expert evidence. | at least 5% relative lower macro EER and lower worst EER than B1/B2 | Fails either target aggregate or paired uncertainty gate. |
| A1 | G | V-REx nonnegative L2 logistic fuser | Penalizing source-group loss variance may generalize better than worst-group reweighting. | exploratory robustness ablation | Never replaces P as the primary claim. |

`P` is the only predeclared primary method. A1 is reported exhaustively but is
not promoted based on target results.

## Source-only selection and training

- Inner leave-one-source-corpus-out validation selects P's L2 coefficient from
  `{1e-4, 1e-3, 1e-2, 1e-1}` and GroupDRO step size from
  `{0.01, 0.05, 0.1}` by lexicographically minimizing source-validation
  worst-corpus EER then corpus-macro EER. Ties choose the smaller L2 then
  smaller step size.
- A1 uses the same L2 grid and variance coefficient `{0.01, 0.1, 1.0}` under
  the same inner procedure.
- B2 uses the fixed central grid value `L2=1e-2`; it is a mandatory baseline,
  not a target-selected method.
- Final source fit uses the selected settings and seeds `1701`, `1702`, and
  `1703`; report mean prediction across all three. No target metric may choose
  a seed, epoch, model, or hyperparameter.
- Fixed optimizer: full-batch AdamW, 1,000 steps, LR `1e-2`, weight decay 0;
  full-batch use makes score-fusion optimization deterministic within seed and
  avoids hidden dataloader variation. B2/P/A1 share this budget.

## Primary endpoint and decision gate

For each blind target, report EER, AUROC, retained common-score count, and all
methods. The primary endpoint is the unweighted mean target EER across all five
targets; the secondary co-primary safety endpoint is their maximum EER.

Point estimates use every retained common-score target row. For the fixed
uncertainty comparison only, the final evaluator freezes up to 10,000 target
IDs per target × label by ascending
`sha256('H8SFBOOT|2608|dataset|label|sample_id')`; smaller classes are used in
full. It then takes 2,000 stratified sample-ID bootstrap replicates on that
method-independent panel. This cap is an execution-bound clarification made
before any target score or label access; it cannot change the point-estimate
panel, model, target, hyperparameter, or decision threshold.

H8-SF is a positive-result candidate only if P:

1. reduces mean target EER by **at least 5% relative** versus both B1 and B2;
2. has lower worst-target EER than both B1 and B2;
3. wins or ties (within 0.05 absolute EER points) on at least 3/5 targets;
4. has a 95% stratified sample-ID bootstrap CI for mean-EER difference below
   zero versus B2 (2,000 fixed replicates, seed 2608); and
5. passes a no-leakage/provenance audit (target labels untouched before final
   evaluation; all source/target/model file hashes recorded).

Every target and method is reported even if P fails. No target is removed after
viewing a result. If P fails this gate, H8-SF stops and the frozen-SSL fallback
requires a new protocol rather than a fusion retune.

## Sanity and provenance gates

1. Verify labels and all eight score files join one-to-one on `sample_id`.
2. Verify source score orientation from source labels only and write a frozen
   orientation report before downloading or loading target labels.
3. Verify every target CDF feature is generated without its labels; write a
   separate label-free target feature hash.
4. Re-run final P from the saved source manifest/settings and require bytewise
   equal source prediction arrays and numerically equal target metrics.
5. Audit base-model metadata. A target stated to have been used in a base
   model's training/tuning is retained only as a clearly labelled retrospective
   benchmark, never as support for the external-generalization claim.

## Stop rule

This is a compact, one-generation go/no-go study. Do not add models, rank
transforms, target datasets, losses, or hyperparameters after target labels are
read. A failed P invokes the separately frozen SSL-probe direction; it does not
license fusion searching.
