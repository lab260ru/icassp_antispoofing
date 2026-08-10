# H9-PCR frozen checkpoint ledger

`scripts/freeze_h9_pcr_checkpoints.py` is the source-only handoff between the
locked source training loop and the one terminal SONAR/ArAD evaluation. It has
no target argument and must run exactly once after all twelve final source
fits are complete: B1, B2, and P for seeds 9101--9104.

It consumes, but never rewrites, the materialized source manifest, frozen P
and B2 pair tables, source-freeze and source-materialization provenance,
source-only lambda-selection JSON, public Res2TCNGuard architecture bundle,
and the twelve final training sidecars/checkpoints. Before publication it
replays the source-manifest/P/B2 validators, verifies both provenance seals,
and checks fresh BF16 initialization, the complete method×seed matrix, fixed
optimization envelope, selected lambda, checkpoint rule, and every byte hash.

The lambda-selection JSON is not trusted as an aggregate alone. It must name
exactly the 12 hash-pinned P grid sidecars (three locked lambdas × four seeds).
The builder rereads each sidecar and its checkpoint, and rejects drift in its
source-artifact triple, CUDA-BF16 provenance, fresh seed-specific
initialization, device, P-only grid identity, lower-EER/lower-epoch source
checkpoint rule, frozen optimization envelope, or source-development EER.
Each selection-candidate seed value must equal the corresponding hashed
sidecar/checkpoint EER before the selected lambda can enter the final matrix.
It then validates the candidate JSON with the terminal evaluator's own ledger
loader and publishes the JSON create-only. An existing output is always an
error; no replacement ledger is possible.

```bash
PYTHONPATH=. python3 scripts/freeze_h9_pcr_checkpoints.py \
  --source-manifest /absolute/HDD/h9_odss_source_manifest.csv \
  --p-pairs /absolute/HDD/h9_odss_source_pairs.csv \
  --b2-pairs /absolute/HDD/h9_odss_b2_random_pairs.csv \
  --source-freeze-provenance /absolute/HDD/h9_odss_source_freeze.json \
  --source-materialization-provenance /absolute/HDD/h9_odss_source_materialization.json \
  --source-selection /absolute/HDD/h9_p_source_lambda_selection.json \
  --res2-bundle /absolute/HDD/Res2TCNGuard_bundle \
  --training-record /absolute/HDD/B1_seed9101.json \
  --training-record /absolute/HDD/B1_seed9102.json \
  --training-record /absolute/HDD/B1_seed9103.json \
  --training-record /absolute/HDD/B1_seed9104.json \
  --training-record /absolute/HDD/B2_seed9101.json \
  --training-record /absolute/HDD/B2_seed9102.json \
  --training-record /absolute/HDD/B2_seed9103.json \
  --training-record /absolute/HDD/B2_seed9104.json \
  --training-record /absolute/HDD/P_seed9101.json \
  --training-record /absolute/HDD/P_seed9102.json \
  --training-record /absolute/HDD/P_seed9103.json \
  --training-record /absolute/HDD/P_seed9104.json \
  --plan experiments/h9_paired_counterfactual/PLAN.md \
  --data-contract experiments/h9_paired_counterfactual/DATA.md \
  --output /absolute/HDD/h9_pcr_frozen_checkpoint_ledger.json
```

The resulting `h9_pcr_frozen_checkpoint_ledger` uses version
`h9-pcr-terminal-evaluation-v1` and is the only permitted checkpoint input to
`scripts/evaluate_h9_pcr_targets.py`. The ledger is source-only and records
`target_labels_read`, `target_audio_read`, and `target_metrics_read` as false;
the terminal evaluator later becomes the first component permitted to open
target labels.
