# H9-PCR source-only training harness

`src/h9_pcr_training.py` is the only H9 source trainer. It starts the pinned
Res2TCNGuard `_net.py` architecture from a fresh seed; it does not load the
public pretrained checkpoint and has no target input option.

## Required frozen inputs

Every training invocation requires the canonical paired-eligible source
manifest and its byte-pinned P pair CSV. B2 additionally requires the
byte-pinned random-pair CSV. The CLI refuses to run without the expected
SHA-256 values recorded by the source freeze.

```bash
PYTHONPATH=. python3 scripts/train_h9_pcr.py \
  --source-manifest SOURCE_MANIFEST.csv \
  --p-pairs h9_odss_source_pairs.csv \
  --p-pairs-sha256 P_PAIR_SHA256 \
  --res2-bundle RES2_ARCHITECTURE_DIRECTORY \
  --output-dir HDD_OUTPUT_DIRECTORY \
  --method P --seed 9101 --batch-size 24 --num-workers 4 \
  --max-epochs 6 --learning-rate 1e-4 --weight-decay 1e-2 \
  --lambda-rank 0.10
```

Every condition supplies `--b2-pairs h9_odss_b2_random_pairs.csv` and
`--b2-pairs-sha256 B2_PAIR_SHA256` so its sidecar binds the complete shared
source-artifact trio. Only B2 decodes the table for rank scheduling; B1/P
validate its bytes but do not use it to schedule a loss. The frozen
GPU assignment is fixed by seed: 9101/9102/9103/9104 map to CUDA devices
0/1/2/3 respectively.

## Pair-file firewall

The P input schema is exactly:

`pair_id, split, group_key, voice_key, content_key, source_corpus, language,
bona_utterance_id, bona_relative_path, spoof_utterance_id,
spoof_relative_path, spoof_generator`.

The B2 input schema is exactly:

`pair_id, split, group_key, voice_key, content_key, source_corpus, language,
spoof_generator, random_bona_utterance_id, random_bona_relative_path,
random_bona_content_key, random_bona_voice_key, random_bona_split,
random_bona_language, random_bona_source_corpus, spoof_utterance_id,
spoof_relative_path`.

The trainer verifies every P edge against the canonical source manifest. B2
must contain exactly the P spoof edges and a bona-fide partner in the same
split/language/source-corpus/generator stratum; its partner must have a
different frozen content key. It emits both pair-file hashes and the
stratum/split distribution audit in each source-run ledger. Pair lists may be
cycled only to fill a common batch schedule; no B2 partner is ever regenerated
or remapped at a later epoch.

## Source-only selection

Run P for every locked lambda/seed combination, then use
`scripts/select_h9_pcr_lambda.py` on the twelve source-development ledgers.
It selects the lower mean source-dev EER across the four seeds (lower lambda
on ties). Supply the resulting lambda unchanged to final P and B2 fits.
Checkpoints use lower source-dev EER and then lower epoch; target data cannot
enter this process.
