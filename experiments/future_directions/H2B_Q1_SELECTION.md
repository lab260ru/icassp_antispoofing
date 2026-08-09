# H2B Q1 quality-family selection

After—and only after—the full detector-free Q1 table is complete,
`scripts/select_h2b_q1_quality_families.py` implements the policy frozen in
`h2b_q1_quality_arm_manifest.json`.

For each non-control arm it records the retained-pair count, 95% Wilson lower
bound, median absolute target-feature delta, and median WER.  A candidate is
eligible only when its lower retention bound is at least 0.90 and its median
absolute target delta reaches the family minimum.  It selects at most one
eligible arm per family by largest lower bound, then largest target delta, then
smallest WER, then canonical arm ID.  Controls are reported by the Q1 table
but cannot be selected.

The selector refuses incomplete runs, score-authorized summaries, mismatched
arm sets, duplicate pair IDs, unknown response-like table columns, and output
overwrites.  It has no detector/model import and produces no causal or EER
quantity.

```bash
PYTHONPATH=. python3 scripts/select_h2b_q1_quality_families.py \
  --quality-table /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2b_quality_calibration/h2b_q1_deepvoice_001/quality_pairs.parquet \
  --quality-summary /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2b_quality_calibration/h2b_q1_deepvoice_001/h2b_q1_quality_summary.json \
  --output-dir experiments/future_directions/results/h2b_q1_deepvoice_selection
```
