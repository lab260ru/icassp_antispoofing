# H9-PCR data and provenance contract

## Exact dataset roles

| Role | Dataset | Revision | Permitted role |
| --- | --- | --- | --- |
| Source development and training | `SpeechAntiSpoofingBenchmarks/ODSS` | `1968e6d0ef141c4572073695bdc1d17a8706177f` | Matched-pair construction, group-disjoint train/dev split, hyperparameter selection, and final source fit. |
| Blind external primary target 1 | `SpeechAntiSpoofingBenchmarks/SONAR` | `eca7c72ebdf0f7936a644605a56735ac8564dbd9` | Final prediction only; target labels are withheld from H9 configuration and training. |
| Blind external primary target 2 | `SpeechAntiSpoofingBenchmarks/ArAD` | `350184966eeb5b46ff2acdabd8f4d12e41e582da` | Final prediction only; target labels are withheld from H9 configuration and training. |

The source revisions and approximate trial counts were obtained from the
hash-sealed `arena-manifest` metadata snapshot under the HDD root. Those
metadata counts are not H9 observations. No target values may be read before
the committed source-selection artifact exists.

## Source eligibility hard stops

ODSS must supply, from its pinned revision:

1. a deterministic **content/utterance key** that maps at least one bona-fide
   clip and at least one spoof clip to the same linguistic item;
2. a source-corpus-plus-speaker/voice identifier that permits **voice-disjoint**
   train/dev allocation, with every language present in both splits;
3. audio decodable to mono PCM and a binary label; and
4. at least 1,000 complete matched groups after the split and source hash audit.

If any condition fails, H9-PCR records `SOURCE_PAIRING_UNAVAILABLE` and stops;
it does not infer pairs from audio similarity, use a random pairing as the
primary method, or substitute a new source dataset post hoc.

All unmatched source items are excluded from **every** H9 method, including
the BCE baseline. This makes pair eligibility a source-pool property rather
than a hidden data-volume advantage for PCR.

## Leakage and target integrity

Before any target metric is calculated, retain a canonical audio fingerprint
for every source and target trial (16-kHz mono PCM SHA-256 after the fixed
resampling/channel conversion) and hard-stop on an exact source--target
collision. This check detects identical standardized waveform content; it does
not claim to prove absence of upstream speaker or text lineage.

Target files may be acquired and byte-hashed for integrity, but their labels
must not be loaded by training, source-dev selection, throughput probing, or
prediction. The final evaluation command is the first allowed target-label
load and must evaluate both targets and every predeclared method together.

For source and target, record repository revision, file paths, SHA-256, bytes,
schema, sample IDs, label counts (only in the final target ledger), duration
statistics, canonical audio fingerprint, and split/pair assignment hashes. The
source ledger also reports any available content-key overlap across its
voice-disjoint train/dev split; no claim of text-disjoint development is made
without a transcript-level key.
