# H9-PCR ODSS source materialization contract

`scripts/materialize_h9_odss_source.py` is the source-only adapter from the
hash-sealed ODSS pairing freeze to the waveform manifest accepted by
`src/h9_pcr_training.py`. It imports no target loader and has no target CLI
argument. Writing this contract did not invoke it or decode any source audio.

## Fixed input and fail-closed validation

The command accepts only the completed `h9odss_source_pairing_001` directory
and raw ODSS Parquet shards below revision
`1968e6d0ef141c4572073695bdc1d17a8706177f`. Before it opens a shard, it
requires the exact source-freeze JSON/CSV SHA-256 values in
`results/H9_SOURCE_FREEZE_001.md`, checks the repository revision and metadata
hash, validates all source-pool IDs, and reconstructs the frozen P and B2 edge
invariants. A changed schema, ID, label, pool, pair, or edge is a hard stop.
It does not derive a replacement pairing or remap B2.

On an authorized source run it copies exactly the 23,883 frozen RIFF/WAVE byte
payloads—one per paired-only source trial—and no unmatched or extra shard row.
The output must be a new directory under the project HDD root. An interrupted
or failed invocation removes only its fresh staging directory; existing output
is never overwritten.

```bash
PYTHONPATH=. python3 scripts/materialize_h9_odss_source.py \
  --freeze-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9odss_source_pairing_001 \
  --raw-shard-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/odss/1968e6d0ef141c4572073695bdc1d17a8706177f/raw/data \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9odss_source_materialization_001
```

## Output contract

`h9_odss_source_manifest.csv` contains precisely the H9 trainer columns, in
this order:

`sample_id,audio_path,label,split,pair_id,group_id,language,canonical_fingerprint,source_corpus,speaker_id,spoof_generator`

`sample_id` is the frozen ODSS ID, `pair_id` is the frozen content key, and
`group_id` is exactly `source_corpus|speaker_id`, preserving voice-disjoint
selection. The adapter reruns the emitted CSV through the H9 source-manifest
validator before publishing output. It preserves the sealed P/B2 CSV byte
hashes in its provenance rather than copying, modifying, or remapping edges.

The adjacent source-audio audit records input shard, raw payload SHA-256 and
bytes, copied payload SHA-256 and bytes, canonical fingerprint, and canonical
PCM size for every row. The JSON additionally binds each raw Parquet shard by
SHA-256 and Arrow schema hash.

## Canonical audio fingerprint

Copied WAVs remain byte-for-byte unchanged. Separately, the fingerprint decodes
WAV bytes through `soundfile` as float32 with channels retained; computes a
float64 arithmetic channel mean; resamples to 16 kHz using
`scipy.signal.resample_poly` with reduced integer factors,
`window=('kaiser', 5.0)`, and `padtype='constant'`; then rejects empty or
non-finite output. Samples are clipped to `[-1, 1]`, rounded from
`sample * 32767` into little-endian signed 16-bit PCM, and SHA-256 hashed with
a versioned policy tag and uint64 frame count. That deterministic normalized
representation is reserved for the later source--target collision audit; it
does not alter training audio.
