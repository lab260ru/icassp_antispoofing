# H8 budget

```text
metric                 = corpus-macro target EER and worst-target EER
direction              = lower
seed_experiments       = 4 methods + 1 individual-model baseline family
seconds_per_experiment = 600 maximum wall time per seed/target evaluation
parallelism            = 4 CPU workers; no GPU required for 8-D fusers
compute_cap            = 6 CPU-hours for H8-SF; reserve 24 GPU-hours for SSL fallback
max_generations        = 2 for H8-SF before a go/no-go outer-loop decision
hypotheses_per_gen     = 4
proposers              = 3 Codex-only research perspectives
critics                = 3 Codex-only research perspectives
stagnation             = 1 generation
--- spent ---
generations_run        = 0
experiments_run        = 0
cpu_min_used           = 0
gpu_min_used           = 0
best_metric            = pending fresh baseline
champion               = none
```

The H8-SF compute cap is intentionally small: it is a rapid evidence gate.
If its locked target criterion fails, do not extend the fusion sweep; move to
the separately protocolled frozen-SSL fallback.
