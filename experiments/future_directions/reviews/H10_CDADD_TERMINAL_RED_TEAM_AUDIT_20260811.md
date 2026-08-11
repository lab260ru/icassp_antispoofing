# H10 CD-ADD terminal red-team audit — 2026-08-11

**Reviewer role:** read-only audit after the terminal evaluation. This review
did not rerun inference, alter data/code, or open a new target. It checks
whether the saved H10 record supports the stipulated **terminal no-retune**
interpretation; it is not an additional result.

## Verdict

**Pass for the no-retune interpretation.** I found no issue that invalidates
the literal H10 decision: the extension gate is false because P's 8.2983%
relative EER reduction against B1 is below the preregistered 10% requirement.
The fact that both paired bootstrap intervals exclude zero cannot replace that
failed practical-effect condition. H10 therefore neither extends H9's paper
claim nor authorizes a target/model/loss/source rerun.

## Evidence checked

### Freeze, source handoff, and provenance

- Git history places the H10 protocol/decision freeze in `1dbf4af`
  (07:48:13 UTC), the adapter in `988a3c8` (08:00:25), and the protocol-binding
  tests in `318888e` (08:03:20). The recorded CD-ADD metadata audit,
  materialization, raw predictions, and decision were then created at 08:04,
  08:08, 08:30, and 08:30 UTC, respectively. This is consistent with the
  intended pre-target implementation/freeze order.
- The H10 terminal provenance and an independent call to
  `_validate_h9_source_ledger` bind the same exact H9 checkpoint ledger:
  `3e0530e2b850cbb08329d14a8982a7448a319a4a30e75ee2d51dc4ecb7f2325d`.
  The loader validated all 12 B1/B2/P × seeds 9101--9104 entries, their
  checkpoint hashes/sidecars, H9 plan/data hashes, source artifacts, fresh
  initialization provenance, and source-only selection record.
- The H9 source freeze remains the paired-only ODSS pool of 23,883 trials,
  15,922 P edges and 15,922 B2 edges; H10 did not provide a code path for
  changing those source artifacts. Exact canonical fingerprints between that
  source manifest and all CD-ADD records have zero intersection.
- The target manifest/provenance lock the stated CD-ADD repository revision,
  8 audio-bearing Parquet shards, CC-BY-4.0 card provenance, and hashes for
  records, labels, materialization provenance, and each raw shard. Reloading
  the H10 materialization validator passed. A direct byte-size/SHA-256 audit
  also passed for all 20,786 copied waveforms.

### Firewall and all-trial scope

- The implementation replays the H9 ledger before opening the target
  manifest. The materializer's record pass requests only `path,audio`, writes
  and hashes a label-free records CSV, and its distinct later pass requests
  only `path,label`. The saved manifest declares `target_labels_read: false`
  and `target_metrics_read: false` at this handoff.
- The terminal evaluator writes and hashes the complete label-free raw
  prediction Parquet before its label loader is called. Production invocation
  rejects a non-CUDA device and a synthetic prediction runner; the injection
  seams are confined to synthetic tests. The five adapter contract tests pass
  (`5 passed`).
- The terminal and materialization provenance explicitly declare every
  audio-bearing trial, with `target_subset`, `generator_stratum`, and
  `path_based_exclusions` all null. The recorded all-trial panel is 20,786
  unique IDs (3,661 bona-fide; 17,125 spoof), and the raw matrix contains the
  expected 249,432 rows = 20,786 × 3 methods × 4 seeds, with no duplicated
  method/seed/trial keys.

### Metrics and gate replay

All compact output hashes listed in the terminal note match their files.
From the saved raw predictions and sealed labels, I independently recomputed
the ensemble metrics, all 12 per-seed metrics, the fixed 2,000-draw shared-ID
label-stratified bootstrap (seed 2909), and the decision object. Each replay
matched exactly.

| Method | All-trial EER | AUROC |
| --- | ---: | ---: |
| B1 | 45.4247% | 0.5848 |
| B2 | 48.9211% | 0.5280 |
| P | **41.6553%** | **0.6147** |

The verified contrasts are P--B1 = -3.7708 pp (95% CI [-4.6982, -2.7741])
and P--B2 = -7.2411 pp ([-8.3934, -6.1184]). P's relative reductions are
8.2983% versus B1 and 14.8520% versus B2. Thus the B1 relative-reduction rule
is the sole failed rule; all provenance, collision, prediction-order,
reconstruction, and bootstrap rules are true. The P per-seed EERs range from
42.3741% to 51.8755%, so the predeclared four-seed probability ensemble—not a
best seed—is appropriately the reported quantity.

## Residual limitations (not invalidations)

- Artifact timestamps and the code's write-before-read control corroborate
  sequencing, but they are not an external, independently timestamped
  attestation. The claim should remain the narrower reproducible-protocol
  claim, not a claim of cryptographic blind evaluation.
- The public-card overlap audit and zero waveform collision rule do not prove
  speaker, text, or recording-source disjointness. Both the protocol and the
  terminal note already avoid that stronger assertion.
- H10 is one source-frozen Res2TCNGuard target replication. Its failed
  extension threshold is not evidence against H9's sealed two-target result,
  nor evidence of architecture- or target-family generality.

## Required disposition

Keep `H10_DECISION_RECORD_20260811.md` as the historical prospective choice
record and `results/H10_CDADD_TERMINAL_EVALUATION_001.md` as the terminal
outcome. Do not re-open H10 by changing its threshold, subset, target,
checkpoints, seeds, ensemble, bootstrap, training source, or loss. Do not add
H10 to the H9 paper's supported-result table.
