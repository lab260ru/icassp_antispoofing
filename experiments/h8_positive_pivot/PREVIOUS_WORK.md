# H8 previous-work audit

## Current baseline opportunity

- **Available interface:** five byte-addressable score panels, each with all
  eight published anti-spoofing systems, normalized so larger values mean more
  spoof-like: `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/scores/<dataset>/score_panel.parquet`.
- **Potential H8 endpoint:** source-only training on ASVspoof2019_LA,
  ASVspoof2021_LA, and ASVspoof2021_DF; untouched final evaluation on
  InTheWild and ASVspoof5. Primary comparison must include individual experts,
  uniform rank fusion, and static regularized logistic fusion.
- **Why a fusion study is plausible:** H6 records large within-class
  disagreement across experts. For example, Spectra-AASIST/Res2 has median
  within-class agreement 0.0295, while the strongest single expert changes by
  corpus. This is a hypothesis generator, not an H8 tuning input.
- **Starting expert EERs (published artifacts):** Spectra-AASIST is 1.464% on
  InTheWild and 14.220% on ASVspoof5; XLSR-SLS is 7.456%/18.764%; W2V2-AASIST
  is 11.222%/16.248%. The new model must beat a locked fusion baseline on both
  held-out corpora, not merely choose the apparent strongest expert.

## What was tried and must not be recycled

- H1's frozen crest candidate failed its five-corpus portability criterion.
  The 19 descriptive units cannot select H8 features or hyperparameters.
- H2 and H2B stopped before score measurement because their quality criteria
  failed; they cannot be reinterpreted as augmentation evidence.
- H4--H6 are descriptive analyses only. H7's fixed feature-vector logistic
  baseline transfers poorly to InTheWild/ASVspoof5, so it is not an H8 method.
- The existing paper is a sealed negative-result working draft. H8 replaces
  rather than inflates its central empirical contribution if a positive result
  survives a new protocol.

## Data, code, and compute facts

- Core local corpora contain `test-*.parquet` partitions only. A fusion paper
  must accurately call ASVspoof corpora *source training domains*, not official
  train/dev splits; InTheWild/ASVspoof5 remain untouched targets.
- ASVspoof2021_DF is strongly imbalanced (22,617 bona fide versus 589,212
  spoof); source-domain-balanced sampling is required in every H8 baseline.
- No reusable BF16 trainer exists. Res2TCNGuard is the only immediately
  trainable local network, while most other local models are ONNX inference
  artifacts. A fresh waveform fine-tuning project is higher risk than fusion
  for the available time.
- Four idle RTX 6000 Ada GPUs (48 GB) and PyTorch 2.11/CUDA 13 are present.
  H8's score-vector candidates are small enough to run many seeds quickly;
  GPU use remains relevant for parallel bootstrap/model-selection sweeps but
  hardware alone is not treated as a contribution.

## Open hypotheses / TODOs

- Does a domain-balanced, regularized score-fusion baseline beat uniform and
  static logistic fusion simultaneously on both untouched target corpora?
- Does a predeclared disagreement-aware gated mixture improve the mean and
  worst-target EER without target-label tuning?
- Does group-DRO over source corpora produce a smaller held-out worst-target
  EER than ERM/static fusion?

## Known fragile areas

- Score orientation must be validated per model/corpus; use only the supplied
  `score_spoof` field after revalidating its panel hash and labels.
- A target-label peek, selecting a model per held-out target, or using corpus
  identity as an input would invalidate a deployment/generalization claim.
- A numerical win must be repeated across independently seeded source-domain
  training runs and evaluated on the frozen target panels with confidence
  intervals.

## Audit sources

- `data/arena-index.yaml`
- `src/arena_io.py`
- `experiments/h6_score_agreement/results/H6_ANALYSIS_001.md`
- `experiments/h7_feature_transfer/results/h7_analysis_001/H7_ANALYSIS_001.md`
- `experiments/h2_causal_interventions/results/quality_runs/H2_QUALITY_RUN_002.md`
- `experiments/future_directions/results/H2B_Q1_RUN_001.md`
- `research-state.yaml`, `findings.md`, `research-log.md`, and commits through
  `5ffbb08`
