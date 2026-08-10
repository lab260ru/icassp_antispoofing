# H9-PCR source materialization 001

**Status:** complete source-only waveform materialization. No target corpus,
target label, detector score, or H9 model fit was accessed at this stage.

## Immutable output

HDD directory:
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9odss_source_materialization_001/`

| File | SHA-256 |
| --- | --- |
| `h9_odss_source_manifest.csv` | `91403d85acc249dcb66d9f197563113fad91077150b6bb4082a500ef0552101d` |
| `h9_odss_source_audio_audit.csv` | `c76493acad68eed905e1792e4589cba4b6f14bdd5b8e6fbb76187934298807a8` |
| `h9_odss_source_materialization.json` | `848c926dd6946a08132076fb75223ebf56a2bd1ffa78268dbc1cf194f8c64384` |

The materializer revalidated the completed `h9odss_source_pairing_001` freeze,
including the byte identities of the trial, matched-P, and random-B2 tables.
It copied exactly the frozen paired-only ODSS WAV payloads and wrote the
trainer's canonical eleven-column manifest. The P/B2 pair tables remain the
sealed upstream files; they were neither copied nor remapped.

## Outcome

- 23,883 source WAV payloads were copied below `waveforms/`, one per frozen
  paired-only source trial; no unmatched source row was admitted.
- Every manifest row has an existing audio path and a unique canonical 16-kHz
  PCM fingerprint under the versioned policy in `SOURCE_MATERIALIZATION.md`.
- The class/split counts remain exactly 6,051 bona fide and 12,102 spoof in
  train, and 1,910 bona fide and 3,820 spoof in development.
- The JSON provenance binds the raw ODSS shard hashes and schemas as well as
  the upstream freeze hashes. It explicitly records source-audio access and
  no target, score, model, or training access.

This establishes the only permitted training input for H9. The next stage is
the already-committed fresh-initialized BF16 source-only lambda selection; its
source-development outputs may choose only the predeclared P/B2 lambda.
