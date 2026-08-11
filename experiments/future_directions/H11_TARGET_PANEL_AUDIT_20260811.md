# H11 target-panel metadata audit — 2026-08-11

**Status:** prospective, metadata-only readiness audit.  This record does not
create an H11 protocol, download an artifact, access a target Parquet row,
decode audio, load a label, inspect a published score/prediction, or calculate
a target metric.  It exists to make a later fresh-architecture replication
resumable without silently reusing an earlier target panel.

## Question and selection boundary

Can a future H11 fresh-architecture replication use a compact, legally usable,
two-corpus external panel that is disjoint from the H8 fusion targets, H9's
SONAR/ArAD terminal targets, and H10's completed CD-ADD target?

The initial card screen suggested **HABLA + EmoFake_test**, both under CC BY
4.0, but EmoFake's missing recording-corpus provenance makes it conditional.
The addendum below supersedes that provisional panel with a firm scientific
recommendation. Neither public card/source record considered here identifies
VCTK, HiFiTTS, HUI German, or OpenSLR Spanish as an upstream recording corpus—
the four known ODSS lineages. That is a useful screen, not a proof of source,
speaker, text, or item disjointness. A later H11 protocol must preserve the
caveats and pass the explicit lineage gate below before it calls either target
external.

## What was and was not inspected

On 2026-08-11 this audit used only the public Hugging Face dataset metadata and
the revision-pinned `README.md` cards:

- `hf datasets info ... --revision <commit>` for repository visibility,
  resolved revision, tags, and file-name inventory;
- the public revision-qualified Hub tree API for **file names, LFS object
  metadata, and byte sizes**; and
- the public revision-pinned dataset cards for declared schema, licence, trial
  counts, languages, and source descriptions.

No Dataset Viewer row endpoint, Parquet URL, local cache, `datasets` loader,
audio payload, `labels.parquet` content, `notes`, Arena submission, target
prediction, score, or metric was opened.  Counts below are card-declared
population summaries, not locally recomputed labels.

## Initial card screen — superseded by the addendum

| Role | Exact public dataset revision | Card-declared population and source provenance | Packaging / licence | H11 terminal work if it retains 3 methods x 4 seeds | Status |
| --- | --- | --- | --- | --- | --- |
| Target A | `SpeechAntiSpoofingBenchmarks/HABLA` @ `764c00726cc4327ff2a270b8347375e95a7034db` | 80,816 trials: 22,816 bona-fide and 58,000 spoof. Spanish, covering Argentina, Chile, Colombia, Peru, and Venezuela. The card describes natural Latin-American Spanish and CycleGAN, Diff, StarGAN, TTS, TTS-Diff, and TTS-StarGAN attacks; it cites Tamayo Flórez et al., Interspeech 2023, DOI `10.21437/Interspeech.2023-2272`. | Public, ungated, enabled, CC BY 4.0. One `test` config, 40 audio-bearing `data/test-*.parquet` shards totaling **11,757,231,268 B** (11.76 GB decimal); declared 16-kHz mono audio plus `path`, binary `label`, and `notes`. A separate `data/labels.parquet` file is **1,034,759 B** in the tree. | 969,792 trial-checkpoint predictions total; a one-seed-per-GPU schedule gives 242,448 clips/GPU across B1/B2/P. | **GO, conditional on provenance gate.** |
| Target B | `SpeechAntiSpoofingBenchmarks/EmoFake_test` @ `a49bd1c2110e199421055fd2f2561c8eb9e43c13` | 17,500 trials: 3,500 bona-fide and 14,000 spoof. English emotional speech; the card describes emotion-conversion spoofing and cites arXiv `2211.05363`. It records `utterance_id`, `speaker_id`, `emotion`, `method`, and `subset` as `notes` fields, but does not name the upstream recording corpus in the card. | Public, ungated, enabled, CC BY 4.0. One `test` config, 3 audio-bearing `data/test-*.parquet` shards totaling **1,498,701,710 B** (1.50 GB decimal); declared 16-kHz mono audio plus `path`, binary `label`, and `notes`. | 210,000 trial-checkpoint predictions total; 52,500 clips/GPU in the same scheduling model. | **Conditional GO.** Primary-paper/upstream-lineage audit is mandatory before materialization. |

The packaged audio-only panel is **98,316 trials**, **43 Parquet shards**, and
**13,255,932,978 B** (13.26 GB decimal).  With the H9-style B1/B2/P x four-seed
matrix, this is **1,179,792** trial-checkpoint passes.  If an H11 architecture
retains four independently initialized seeds, the natural fixed schedule is
one seed per 48-GB GPU; each device evaluates 294,948 clips over all three
methods.  This calculation is a resource plan, not a measured throughput
claim.

## Feasibility decision

This is feasible as a staged H11 experiment, not as an opportunistic target
swap:

1. Before ODSS or target audio is decoded, pin the new trainable architecture,
   write a synthetic-only BF16 forward/backward qualification, and measure the
   fastest safe batch size and worker count on the four RTX 6000 Ada GPUs.
   The synthetic benchmark—not this card audit—must set the target inference
   batch and source-training recipe.
2. Commit the H11 source-only protocol, target panel above, all four new seeds,
   source-dev selection rule, B1/B2/P edge contracts, epoch budget, and
   two-target all-trial decision/bootstrap rule.  Do this before either target
   path, row, audio, label, score, or target prediction is visible.
3. Implement a new generic H11 materializer and test it only on synthetic
   Parquet fixtures.  Its record pass may project `path,audio` and create a
   label-free record/fingerprint manifest; only after that manifest is hashed
   may its label pass project `path,label` into a separately sealed label
   artifact.  HABLA's separate labels file must not bypass the same firewall.
4. Materialize exactly the addendum's 41 declared shards under
   `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`, record every LFS SHA-256,
   file size, schema, and opaque record count, then run all 12 frozen H11
   checkpoints before one terminal label join.  Do not use Arena submission
   files for orientation, selection, or comparison.

The principal practical risk is HABLA's 11.76-GB, 80,816-trial payload. The
addendum panel is nevertheless smaller than the 15.65-GB HABLA+CD-ADD panel
scoped by the earlier prospective AASIST audit, while replacing the now-terminal
CD-ADD target with a 170.92-MB target. A future protocol must still hard-stop if its
measured synthetic throughput or HDD headroom cannot support the complete
all-trial panel; it may not prune HABLA, remove/downsample J-SPAW, or choose a
target based on a preliminary score.

## Independence and leakage audit

### Required hard checks before target acquisition

- **H8/H9/H10 exclusions:** do not use CFAD, CVoiceFake_small, DECRO,
  LibriSeVoc, XMAD (H8 primary targets), SONAR/ArAD (H9), or CD-ADD (completed
  H10).  Neither recommended revision is an H8/H9/H10 terminal target.
- **Known ODSS lineage exclusion:** reject DFADD because its card explicitly
  says it is VCTK-derived.  Never infer that zero waveform collisions prove
  speaker/text disjointness from ODSS's VCTK, HiFiTTS, HUI German, or OpenSLR
  Spanish source components.
- **Canonical waveform collision:** after source and target manifests are
  independently materialized, require zero canonical mono-16-kHz PCM
  fingerprints against the exact H11 ODSS source manifest.  A collision is a
  hard stop for the affected target.
- **Documented source lineage:** before a target download, audit the upstream
  HABLA and J-SPAW documentation for recording-corpus names, speaker lists,
  utterance IDs, and released-subset overlap with ODSS. J-SPAW's card identifies
  newly recorded Japanese source audio, but that does not substitute for the
  required audit. If a
  known conflict cannot be excluded by predeclared metadata, stop that target;
  do not substitute another dataset after looking at H11 source or target
  outcomes.
- **Historical model evidence:** the repositories expose Arena submission
  files, including established model names.  H11 must neither read those files
  nor load pre-trained/Arena checkpoints.  The architecture must be
  fresh-initialized and all source choices must be made on the H11 ODSS
  development split only.

### Historical project exposure to disclose

HABLA card metadata was named in the pre-result
`H10_METHOD_REPLICATION_AUDIT_20260811.md` as a prospective architecture
target, but it was never an H8/H9/H10 terminal target and this audit did not
open any HABLA content.  This is card-level provenance awareness, not target
label/score exposure. J-SPAW_LA has no prior project match in the repository
text scan performed for this audit. The H11 protocol must disclose both facts
and keep the target panel fixed even if H10's CD-ADD outcome is known.

## Alternatives deliberately excluded

| Dataset | Exclusion reason |
| --- | --- |
| CFAD, CVoiceFake_small, DECRO, LibriSeVoc, XMAD | H8 primary target panel; target outcomes/scores are historically exposed. |
| SONAR, ArAD | H9 terminal target panel. |
| CD-ADD | H10's complete terminal target; its failed practical-effect gate cannot motivate a post-result target replacement. |
| DFADD | Public card identifies VCTK lineage, which overlaps an ODSS upstream source. |
| DeepVoice, InTheWild, ASVspoof2019/2021, ASVspoof5 | Already used elsewhere in this project as source, target, or descriptive/quality material; they would not provide a clean new target firewall. |
| ADD22_eval_31, ADD2023_track12_test_r1, PyAra, XMAD | Labels-only, inadequate source provenance, or restrictive licences; XMAD is additionally an H8 target. J-SPAW_LA is reconsidered in the addendum because its card has concrete source provenance. |
| EmoSpoofTTS | Spoof-only package, so it cannot provide the predeclared binary all-trial endpoint without constructing a new composite corpus. |

## Evidence links

- [HABLA card at the pinned revision](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/HABLA/blob/764c00726cc4327ff2a270b8347375e95a7034db/README.md)
- [J-SPAW_LA card at the pinned revision](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/J-SPAW_LA/blob/4f3929838e94f16f5e4a60aeb6c8a5f0bc089049/README.md)
- [EmoFake_test card at the pinned revision](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/EmoFake_test/blob/a49bd1c2110e199421055fd2f2561c8eb9e43c13/README.md)
- Earlier fresh-architecture feasibility reasoning, which must not be treated
  as an H11 protocol: `H10_METHOD_REPLICATION_AUDIT_20260811.md`.
- H9 source/target/firewall authority:
  `experiments/h9_paired_counterfactual/PLAN.md`, `DATA.md`, and
  `TARGET_MATERIALIZATION.md`.

## Go / no-go

**Conditional GO:** subject to the addendum's non-commercial licence boundary,
use the exact HABLA + J-SPAW_LA revisions only in a newly committed H11
protocol with a fresh architecture, synthetic BF16 preflight, source-only model
selection, complete panel materialization, and the two-pass label firewall.

**NO-GO today:** do not download, open, run, subset, compare, or put either
target in `paper/main.tex` from this readiness audit. Do not replace the
addendum panel after a later source or target result. If a provenance audit
finds an ODSS conflict, record H11 as blocked/negative under its precommitted
rule rather than target-shopping.

## Addendum — replacement for the conditional EmoFake target

**Supersedes the provisional `HABLA + EmoFake_test` recommendation above.**
EmoFake's Hub card does not name its upstream recording corpus, so it cannot
clear an H11 source-lineage gate on card evidence alone. A second metadata-only
search of every remaining audio-bearing organization package finds one better
provenanced candidate:

| Role | Exact public dataset revision | Card-declared population and source provenance | Packaging / licence | H11 terminal work, 3 methods x 4 seeds |
| --- | --- | --- | --- | --- |
| Replacement Target B | `SpeechAntiSpoofingBenchmarks/J-SPAW_LA` @ `4f3929838e94f16f5e4a60aeb6c8a5f0bc089049` | 2,397 trials: 800 bona-fide and 1,597 spoof. Japanese LA evaluation data from the J-SPAW (*Japanese Speaker-verification and sPoofing-attacks recorded in-the-Wild*) `ver1` release; natural speech was recorded with an iPhone 8 M2 microphone across four environments, and attacks are L1/L2. The card cites Shiota et al., Interspeech 2025, DOI `10.21437/Interspeech.2025-352`, the upstream repository, and `metadata_LA.txt`. This card names a newly recorded Japanese collection—not VCTK, HiFiTTS, HUI German, or OpenSLR Spanish—and the project text scan finds no prior H8/H9/H10 use. | Public, ungated, enabled, one `test` config, one 16-kHz-mono audio-bearing Parquet shard of **170,919,812 B** (170.92 MB). Its terms are explicitly **non-commercial research only** (`license: other`), not CC BY. | 28,764 trial-checkpoint predictions total; 7,191 clips/GPU in the fixed one-seed-per-GPU schedule. |

The **firm scientific recommendation** for an H11 architecture replication is
therefore **HABLA + J-SPAW_LA**, at exactly the two commits shown above. It
contains 83,213 trials, 41 audio shards, **11,928,151,080 B** (11.93 GB
decimal), and **998,556** trial-checkpoint passes for B1/B2/P x four seeds
(249,639 clips/GPU under the fixed seed schedule). J-SPAW's small but fully
declared panel is intentionally retained as a distinct Japanese in-the-wild
condition; the all-trial endpoint and a label-stratified shared-ID bootstrap
must be fixed before any data access, rather than replacing it with a larger
target after seeing an H11 outcome.

### Licence hard boundary

J-SPAW is usable only if H11 is conducted and distributed as **non-commercial
academic research** under its upstream terms. The named industry affiliation
does not by itself establish that condition. Before committing an H11 protocol,
obtain and record the project owner's confirmation that this exact research use
is non-commercial (or written permission from the rightsholder). If that
confirmation is unavailable, **declare no qualifying second target**: the
remaining organization candidates are either labels-only (ADD 2022/2023),
historically exposed, explicitly ODSS-lineage-conflicted (DFADD), spoof-only,
lacking adequate source provenance (PyAra/EmoFake), or have similarly
restrictive/non-commercial terms without J-SPAW's concrete provenance.

This addendum used only the J-SPAW revision-pinned Hub card, repository/tree
metadata, and a repository text-name scan. It did not access a row, audio,
label, score, prediction, or dataset download. EmoFake remains an unselected
fallback only if a pre-result upstream documentation audit later proves its
lineage and a **new** protocol chooses it before any H11 source/target outcome;
it may not be swapped in after J-SPAW results.
