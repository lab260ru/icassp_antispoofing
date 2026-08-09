# H2 post-quality score-eligibility freeze

`scripts/freeze_h2_quality_manifest.py` is the only boundary that can open the
current H2 panel to detector scoring. It consumes an **already completed**
detector-free quality-run directory; it does not decode audio, run ASR, load a
detector, or produce scores.

Before emitting anything, it checks the committed pre-score manifest, arm
ledger, and Arena index; their hashes against the quality-run provenance; all
four registered arms; every expected immutable pair checkpoint; the derived
Parquet table; and the completed run summary. Pilot, bounded, incomplete,
duplicate, missing, provenance-mismatched, or arm-incomplete runs are refused.
The fixed, predeclared criterion is at least 90% retained pairs in **each**
frozen arm. Failed quality rows are required in the source run but never enter
the emitted manifest.

After a completed pass, use a fresh repository-relative output directory:

```bash
PYTHONPATH=. python3 scripts/freeze_h2_quality_manifest.py \
  --quality-run-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/RUN_ID \
  --output-dir experiments/h2_causal_interventions/results/quality_freezes/RUN_ID
```

The command refuses to overwrite `--output-dir` and writes:

- `score_eligible_pairs.csv` and `score_eligible_pairs.parquet`: only
  `retained=True`, `quality_status=passed` rows, marked
  `detector_stage=eligible_after_quality_freeze`. They retain `pair_id`,
  sample identity/selection fields, arm definition and parameters, original
  and transformed waveform hashes, sample counts/rate, transform diagnostics,
  and all quality measurements/gates. They contain no detector-response field.
- `quality_freeze.json`: source pre-score and quality-run hashes, an aggregate
  immutable-checkpoint hash, arm-wise 90% gate accounting, logical plus
  byte-level manifest hashes, and the explicit guard that this artifact is not
  a score result or causal estimate.

A scorer must take these file paths explicitly, verify
`score_eligible_manifest_sha256` against the CSV bytes, and consume no other
pair from the quality-run directory.

## Focused verification

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_h2_quality_freeze.py \
  tests/test_h2_quality_runner.py \
  tests/test_h2_pre_score_pairs.py
```
