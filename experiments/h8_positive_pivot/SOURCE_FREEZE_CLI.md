# H8-SF source-only input freeze

Run this stage before downloading target score artifacts or opening a target
label table:

```bash
PYTHONPATH=. python3 scripts/freeze_h8_source_inputs.py \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h8_score_fusion/h8sf_source_freeze_001
```

The command reads only the three source label tables and their 24 pinned raw
score files. It rejects incomplete score/label joins and a model whose raw-score
polarity changes across source corpora. It writes an immutable balanced source
manifest (up to 10,000 IDs per source × label), source score/label byte hashes,
and one source-learned orientation multiplier for every frozen system.

It refuses an existing output directory. The target datasets, target labels,
target scores, and target metrics are absent from both the CLI arguments and
the emitted provenance contract.

The H8 v2 normalizer removes only a terminal audio suffix, rather than using a
generic basename stem. This preserves literal dotted CVoiceFake identifiers
such as `multi_band_melgan.v2_generated...` and prevents accidental ID
collisions during target joining.
