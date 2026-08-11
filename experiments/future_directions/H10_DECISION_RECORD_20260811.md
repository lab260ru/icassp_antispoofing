# H10 decision record — 2026-08-11

**Status:** prospective decision, before any H10 target row, audio, label,
score, prediction, or metric is accessed.

## Decision

Use a single fresh external target, `SpeechAntiSpoofingBenchmarks/CD-ADD` at
revision `b03c6cf3463d67c1525ba20738c495d120675876`, for a bounded **H10
external-target replication** of the already sealed H9 Res2TCNGuard
checkpoint panel. This is not a new source fit, architecture comparison, or
replacement of H9. It is a third, independently declared target for the fixed
B1/B2/P models.

The terminal panel, metric, and pass rule are fixed in
`H10_CDADD_EXTERNAL_REPLICATION_PROTOCOL.md` before acquisition. A result is
eligible to extend the H9 manuscript only if that new terminal gate passes;
otherwise the H9 paper remains scoped to SONAR and ArAD and H10 is retained as
a terminal follow-up without retuning.

## Reconciliation of the readiness audits

`H10_TARGET_REPLICATION_READINESS_20260811.md` recommends LibriSeVoc as the
smallest engineering target. `H10_METHOD_REPLICATION_AUDIT_20260811.md`,
however, correctly records that LibriSeVoc had already entered the earlier H8
score-fusion target panel. It is therefore not a clean independent H10 target,
regardless of the label firewall in a new evaluator. This record rejects it
for H10 rather than relabelling it as fresh.

The method audit's two-new-architecture/two-new-target AASIST replication
(HABLA and CD-ADD) remains the preferred *future* architecture replication,
but cannot be completed credibly inside this bounded extension: it requires a
new trainable backbone adapter, a new source-only training ledger, 24 fits,
and 15.65 GB of target acquisition. H10 here deliberately tests only the
fresh-target dimension with the pre-existing source-only checkpoint handoff.

## Why CD-ADD

The audit inspected only public revision-pinned card/API metadata. CD-ADD is
not part of the H8 or H9 terminal panels, is public under CC BY 4.0, packages
about 3.89 GB at the pinned revision, and represents modern zero-shot TTS
generation. Its source lineage must still be audited and exact canonical
waveform collisions with ODSS are a hard stop; neither a public card nor a
zero collision proves speaker/text independence.

## Boundaries

- No H9 source split, checkpoint, seed, method, loss, lambda, crop, batch,
  target metric, or bootstrap configuration may change.
- H10 is a source-frozen target replication, not an architecture replication.
  It cannot support a model-independent claim.
- No result can justify another target swap, a subset, a threshold change, or
  a second H10 loss/model search.
- The required two-pass target materializer and terminal evaluator are new
  H10-namespaced code with synthetic tests. They must be committed before a
  CD-ADD path or row is opened.

## Follow-up

1. Commit the protocol and synthetic-only adapter/evaluator implementation.
2. Acquire only the exact pinned CD-ADD shards below the HDD project root.
3. Materialize score-free records, seal labels separately, score all twelve
   existing H9 checkpoints, and run one terminal metric call.
4. Preserve the result and provenance regardless of direction. Amend
   `paper/main.tex` only after the terminal gate and a fresh internal review.
