# Research provenance index

This ARA record is a compact, repository-local epilogue for the 2026-08-11
positive-result pivot. It complements, rather than replaces, the authoritative
hash ledgers in `experiments/`, `research-state.yaml`, `research-log.md`, and
`AGENTS.md`.

## Layers

- `logic/`: problem, claims, experiment plans, and boundaries.
- `trace/`: the evidence-preserving pivot sequence.
- `evidence/`: pointers to the authoritative result artifacts; large files stay
  on the HDD and are not copied here.

Credentials, raw audio, model weights, target predictions, and target labels
are intentionally excluded.
