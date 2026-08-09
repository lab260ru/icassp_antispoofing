# H3 protocol — feature-conditioned robustness fix

**Status:** CONFIRMATORY conditional on a causal H2 result; otherwise exploratory and explicitly labeled as such.

## Question and prediction

An audit-derived feature embedding and transformation-consistency loss improve
macro held-out EER without materially harming in-domain EER.

## Train/evaluation split

Train and tune only on ASVspoof 2019 LA train/dev. Evaluate without OOD tuning
on the configured held-out datasets. Use seeds 13, 37, and 73.

## Compared variants

1. Equivalent-recipe baseline head.
2. Logistic late score-feature fusion.
3. Learned feature-conditioned head.
4. Feature-conditioned head plus consistency loss on H2-accepted transforms.

Run first on Spectra-AASIST and AASIST. New training uses BF16 and independent
one-GPU jobs; choose measured fastest batch/worker settings before full runs.

## Success criterion

The fix succeeds only with at least 10% relative macro OOD EER improvement on
two or more datasets, a paired confidence interval excluding zero, and no more
than 0.2 percentage-point in-domain EER degradation. Otherwise write it as a
negative or exploratory result.

## Outputs

Per-seed configs, throughput probes, metrics, confidence intervals, checkpoints
on HDD, compact summaries, plots, and `analysis.md`.
