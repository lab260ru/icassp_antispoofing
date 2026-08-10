# H9-PCR source-pool and voice-disjoint guard

This implementation note is authoritative for the frozen H9 source manifest
builder and clarifies the source-pairing execution contract without changing
the H9 question, target firewall, model, loss grid, or decision rule.

## Voice-disjoint source selection

The source split unit is `voice_key = source_corpus/speaker`, not an individual
content pair. `scripts/freeze_h9_odss_pairs.py` assigns every complete matched
group belonging to one voice to the same source split. Assignment uses
SHA-256 over `2909:voice_key`, stratified by ODSS's documented language. It
hard-stops unless every observed language occurs in both train and development;
it also requires at least 1,000 complete content groups in each split.

This is stricter than content-pair disjointness: neither the natural recording
nor either TTS rendering from an ODSS voice can appear in both source train and
source development.

## One matched-only source pool for all methods

The builder emits the complete natural+spoof content groups as the *only*
eligible trial table for B1, B2, and P. Every unmatched ODSS row is written to
`h9_odss_excluded_unmatched.csv`, including its identifier, reconstructed
path, generator, content/voice key, exclusion reason, and a byte-bound hash in
the provenance record. Thus the excess VITS-only IDs are not available to B1
as extra label data, to B2 as random partners, or to P as synthetic evidence.

The provenance includes hashes of eligible IDs, excluded IDs, complete trial
rows, P pairs, B2 pairs, and each emitted CSV. A downstream trainer must accept
only `h9_odss_source_trials.csv` as its B1 sample universe and reject a
checkpoint/run configuration whose input trial hash differs from this freeze.

## P versus B2 edge audit

P uses one edge for each matched `(pair_id, spoof_generator)` record, where
`pair_id = source_corpus/speaker/stem`. The deliberate non-uniqueness of
`pair_id` lets the VITS and FastPitch/HiFi-GAN renderings of the same natural
item remain separately auditable.

B2 retains exactly the P spoof edge for every `(pair_id, spoof_generator)`;
only its bona-fide partner changes. The new bona-fide partner must be from the
same source split, language, source corpus, and spoof-generator stratum, while
having a different content key. The deterministic B2 table makes the pair
count and synthetic-generator distribution directly comparable with P before
any model fitting.
