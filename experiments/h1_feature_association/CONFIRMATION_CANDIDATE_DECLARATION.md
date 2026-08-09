# H1 held-out candidate declaration

`declare_h1_confirmation_candidates.py` is a narrow bridge between the
committed discovery freeze and H1's held-out clustered-bootstrap stage. It
creates an analyzer-compatible candidate manifest for one or both registered
confirmation datasets without opening their features, scores, audio, H1
screens, or bootstrap outputs.

The command accepts only the complete three-corpus discovery freeze. Each
unique frozen identity `(model, view, feature, class_label)` must occur exactly
once for ASVspoof 2019 LA, ASVspoof 2021 LA, and ASVspoof 2021 DF with
identical original freeze provenance. It then copies that identity -- not its
discovery dataset name -- to every explicitly requested held-out dataset:
`InTheWild` and/or `ASVspoof5`. It cannot rank, filter, add, or replace
candidates. The output CSV has exactly the schema consumed by
`analyze_associations.py`; its `selection_*` fields and `frozen_at_utc` remain
those of the discovery freeze rather than being rewritten at declaration time.

The adjacent JSON report records (1) an explicit declaration timestamp, (2)
the SHA-256 of the source discovery manifest, (3) hashes of locally available
selection-basis files, (4) the source discovery keys for every projected
identity, and (5) the output-manifest SHA-256. Both artifacts refuse to
overwrite existing paths.

Run this before interpreting a held-out H1 screen or bootstrap result:

```bash
rtk python3 scripts/declare_h1_confirmation_candidates.py \
  --discovery-manifest experiments/h1_feature_association/results/frozen_discovery_to_h2_20260809T201711Z/frozen_candidates.csv \
  --target-dataset InTheWild \
  --declared-at-utc 2026-08-09T21:00:00Z \
  --output-manifest experiments/h1_feature_association/results/declared_confirmation_candidates_20260809T210000Z/InTheWild_candidates.csv \
  --provenance-report experiments/h1_feature_association/results/declared_confirmation_candidates_20260809T210000Z/InTheWild_declaration.json
```

The new manifest is only a declaration of the already frozen discovery
identity. It is not confirmation evidence, a portable-association result, or
a causal result. After the declaration has been committed, pass it to the
H1 analyzer's `--bootstrap-candidates` option alongside the matching target
dataset. See `BOOTSTRAP_CLI.md` for the estimation procedure.
