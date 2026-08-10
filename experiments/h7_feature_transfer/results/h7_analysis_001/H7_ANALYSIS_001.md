# H7 fixed-vector cross-corpus transfer audit — analysis 001

This complete five-cell, score-free leave-one-corpus-out matrix uses one locked all-28-feature logistic baseline. It is descriptive feature-label transfer context only: it does not test a detector, select a cue, establish causality, rank models, or evaluate mitigation.

## Complete matrix

| held_out_dataset | n_train | n_test | auroc | auroc_ci_low | auroc_ci_high | eer | post_imputation_feature_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ASVspoof2019_LA | 40000 | 10000 | 0.907785 | 0.902070 | 0.913709 | 0.162200 | 36 |
| ASVspoof2021_LA | 40000 | 10000 | 0.845107 | 0.837789 | 0.854125 | 0.202400 | 35 |
| ASVspoof2021_DF | 40000 | 10000 | 0.780957 | 0.771824 | 0.789614 | 0.289800 | 36 |
| InTheWild | 40000 | 10000 | 0.534590 | 0.522942 | 0.546265 | 0.452200 | 36 |
| ASVspoof5 | 40000 | 10000 | 0.588320 | 0.577296 | 0.599532 | 0.440600 | 36 |

The locked unweighted median held-out AUROC is 0.780957; the median held-out EER is 0.289800. These are summary descriptors without an acceptance threshold.

Every cell uses 40,000 frozen training rows, 10,000 frozen held-out rows, all 28 registered full-waveform features, train-only median imputation plus missing indicators and standardization, and the fixed L2 logistic recipe. AUROC intervals use 500 source-ID cluster bootstrap replicates; source IDs are singleton in these cohorts, so this is an utterance-level cluster bootstrap. The adjacent provenance JSON binds the two freeze artifacts, the model recipe, and the complete matrix. No score artifact, detector/model code, audio, ASR, feature coefficient, or H1/H4/H5/H6 result was used.
