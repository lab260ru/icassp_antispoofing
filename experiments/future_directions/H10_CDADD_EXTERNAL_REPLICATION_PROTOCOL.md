# H10 — CD-ADD external-target replication protocol

**Status:** locked prospective protocol. Commit this file, its H10-namespaced
synthetic-only target adapter, and its tests before accessing a CD-ADD row,
audio payload, label, score, prediction, or metric.

## Question

Do the already sealed H9 same-item rank-ranking models (P) retain an advantage
over both same-pool BCE (B1) and equal-edge random-pair ranking (B2) on one
new, independently selected external target?

This is a source-frozen, **single-target external replication**, not a new
training experiment. It neither changes H9's two-target result nor tests an
architecture-general mechanism.

## Frozen roles

| Role | Frozen item |
| --- | --- |
| Training source and model panel | H9's immutable 12-checkpoint ledger, SHA-256 `3e0530e2b850cbb08329d14a8982a7448a319a4a30e75ee2d51dc4ecb7f2325d`; B1/B2/P × seeds 9101–9104, fresh Res2TCNGuard, ODSS-only source selection |
| New target | `SpeechAntiSpoofingBenchmarks/CD-ADD` revision `b03c6cf3463d67c1525ba20738c495d120675876` |
| Target construction | Every audio-bearing Parquet trial at that revision; no subset, generator stratum, or path-based exclusion. |
| Input policy | H9's deterministic mono-16-kHz waveform policy and source-model score orientation. |
| Inference | CUDA BF16, all twelve frozen checkpoints; prediction table is complete and written before a label table is joined. |
| Metric | All-trial EER (lower is better) and AUROC; four-seed probability ensemble. |
| Uncertainty | 2,000 shared sample-ID, label-stratified bootstrap draws with seed 2909; contrasts are `P - B1` and `P - B2` in percentage points. |

The source ledger must replay unchanged against the original H9 `PLAN.md` and
`DATA.md`; H10 has no parameter for a replacement source plan.

## Target firewall and materialization

The H10 materializer is allowed only after the H9 source ledger validates. It
has two non-overlapping raw-column passes.

1. **Record pass:** read only `path,audio`; byte-copy declared audio into a
   new HDD run directory, derive opaque sample IDs, calculate H9 canonical
   mono-16-kHz PCM fingerprints, and write a label-free table with exactly
   `sample_id,audio_path,audio_sha256,audio_bytes,canonical_fingerprint`.
2. **Label pass:** after the records artifact is written and hashed, read only
   `path,label`, validate one binary label per record, and write a separate
   `sample_id,label` artifact.

Before score calculation, the evaluator must byte-validate every copied
waveform, validate every raw shard/hash/schema recorded by the materializer,
and hard-stop if any canonical source–target fingerprint collides. It then
writes immutable raw predictions for all method/seed/trial combinations before
opening the sealed label artifact. No target label, score, or metric is
available to source fitting, target materialization's record pass, orientation,
or checkpoint selection.

CD-ADD's declared licence and public-card provenance must be carried into the
materialization provenance. The run must stop if a card/metadata audit shows
a known ODSS recording-source, speaker-list, or released-item overlap that
cannot be excluded; a zero waveform collision alone is insufficient to claim
speaker/text disjointness.

## Decision rule

H10 passes only if every condition holds:

1. The four-seed P ensemble has at least 10% relative lower EER than **each**
   B1 and B2 on the all-trial CD-ADD panel.
2. Both 95% shared-bootstrap intervals for `P-B1` and `P-B2` are wholly below
   zero.
3. The source-ledger replay, target input/reconstruction audit, label-free
   prediction order, complete 12-checkpoint matrix, and exact collision audit
   pass.

Any failure is terminal. It does not authorize a new target, a smaller subset,
a modified loss, another seed, a reweighted ensemble, a changed bootstrap, or
a source rerun. H10's bootstrap is conditional on the fixed four-model
ensemble; per-seed results remain mandatory.

## Interpretation

A pass would strengthen the H9 result from two to three separately pinned
external targets under the same source-trained compact architecture. It would
**not** establish target-family, source-family, architecture, speaker, or
semantic-content generality. A failure leaves H9's sealed two-target claim
unchanged and must be preserved as a no-retune result.

## Delivery and storage

All target shards, copied audio, manifests, predictions, and metrics stay
under `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h10_cdadd/` or the
corresponding HDD dataset cache. Commit only source code, protocol, tests,
compact provenance/hashes, and concise interpretation notes. Do not modify
the paper before the terminal decision and a new claim review. Do not stage
the user-provided template ZIP or `.env`.
