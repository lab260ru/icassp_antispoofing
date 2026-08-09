# H2 score-free input-CSV builder

`scripts/build_h2_input_csv.py` materializes the full label-derived candidate
pool for one explicit Arena dataset before seeded H2 panel selection. It reads
only the dataset entry in `data/arena-index.yaml` and the pinned labels table
through `src.arena_io.load_labels`; it does not open audio shards, H1 feature
tables, score files, ASR, or detector models.

Example (use a new run directory; this does not select the final 500-per-label
panel by itself):

```bash
PYTHONPATH=. python3 scripts/build_h2_input_csv.py \
  --dataset ASVspoof2019_LA \
  --output-csv /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/<run_id>/asv2019_label_pool.csv \
  --output-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/<run_id>/asv2019_label_pool.provenance.json
```

The CSV contains `dataset`, `sample_id`, `source_id`, `label`, and the exact
dataset revision; it preserves `speaker_id` only when the label loader exposes
one. Its JSON records the Arena-index hash, label-file hash, expected pinned
label hash, label counts, and explicit proof that no audio/features/scores/ASR/
detector were accessed. The builder rejects unknown dataset names, missing or
mismatched label hashes, response-like columns, and output overwrites.

Pass this CSV to `scripts/freeze_h2_pre_score_manifest.py` with the configured
seed and `--per-label 500` to create the score-independent H2 panel and the
pre-registered arm ledger. Do not run a detector before the later pair-quality
manifest has frozen the retained rows.
