# H9-PCR fairness and firewall audit — 2026-08-10

## Scope and method

This is a static, source- and target-data-free audit. It inspected only the
locked H9 protocol documents, implementation, synthetic tests, and compact
artifact contracts. It did **not** open ODSS rows/audio, SONAR/ArAD rows/audio
or metadata, source result values, target labels, predictions, checkpoints, or
metrics.

Reviewed components: `PLAN.md`, `SOURCE_PAIRING_FREEZE.md`,
`SOURCE_MATERIALIZATION.md`, `TRAINING_HARNESS.md`,
`CHECKPOINT_LEDGER.md`, `TERMINAL_EVALUATION.md`,
`TARGET_MATERIALIZATION.md`, `src/h9_pcr_training.py`,
`src/h9_pcr_checkpoint_ledger.py`, `src/h9_pcr_evaluation.py`, and the
synthetic H9 tests.

## Findings

| Contract | Static finding | Status |
| --- | --- | --- |
| Shared source pool | The trainer accepts one canonical paired-eligible source manifest. Its validator requires every complete group, and every method binds the identical manifest hash. B1 cannot add unmatched rows. | Pass |
| B1/B2/P BCE schedule | All conditions call the same deterministic class/language-balanced BCE sampler with the fixed seed, epoch, batch size, worker count, waveform policy, optimizer, and six-epoch source checkpoint rule. B1 has only BCE; B2/P add their declared rank term. | Pass |
| B2 fairness | B2 must contain exactly the frozen P spoof edges once, retain split/language/corpus/generator strata, use a different content key, and has the same rank-term count and lambda as P. | Pass |
| Static control | Rank loaders consume only parsed P or B2 CSV edges. They may cycle the already frozen list to fill the common number of BCE steps; no B2 partner is regenerated/remapped in any epoch. | Pass |
| P-lambda wave usability | The selection artifact now has to bind exactly 12 hash-pinned P sidecars (3 lambdas × 4 seeds). The ledger replays each sidecar/checkpoint and checks its EER, source-artifact triple, CUDA BF16 marker, fresh initialization, seed/device, P grid identity, and fixed training configuration before accepting the selected value. | Pass after this hardening; actual source values were not inspected |
| Fresh initialization / BF16 | The source builder seeds before architecture construction, has no checkpoint-loading path, and records `fresh_seeded` plus architecture hash. Production training requires CUDA BF16 autocast and rejects CPU fallback; the checkpoint ledger rechecks it. | Pass |
| Target firewall | Trainer, selector, and checkpoint-ledger builder have no target input. The terminal evaluator validates the complete frozen source matrix, writes complete raw label-free predictions, and only then joins the separate label artifacts. | Pass |
| Target handoff | A strict target materializer now exists. It validates the frozen source handoff before opening a target path, writes score-free `path,audio` records before a separate `path,label` artifact pass, and makes an immutable auditable manifest. | Pass in synthetic tests; no real target run performed |

## Important qualification

The rank objectives necessarily add rank waveform forwards to B2/P. The
fairness claim is therefore *equal BCE schedule and equal B2/P ranking budget*,
not equal wall-clock compute between BCE-only B1 and ranked methods. This is
consistent with the locked method definition and is explicit in the protocol.

## Required release condition

Before target materialization or terminal evaluation, run the source checkpoint
ledger builder with the existing frozen selection artifact. The new replay
checks are the release check for the first P lambda wave; a stale/malformed
selection sidecar must fail closed. No target command is authorized by this
audit itself.
