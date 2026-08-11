# H10 target-replication readiness audit — 2026-08-11

**Status:** read-only readiness audit. This is not an H10 protocol, target
freeze, acquisition, materialization, prediction, metric, or result. It
selects no hyperparameter and reports no target metric.

## Question and hard boundary

H9-PCR's terminal SONAR/ArAD result is sealed. A future H10 must be a new,
independently declared replication: it must not use SONAR or ArAD labels,
audio, predictions, metrics, target subsets, or H9 outcome to select its
source fit, method, target, or stopping rule. This note merely identifies
public candidates whose *metadata and cards* make that possible.

During this audit, no candidate Parquet row, candidate label, candidate audio
payload, published score file, prediction, or target metric was opened. The
only external requests were public Hugging Face repository metadata/tree APIs
and the revision-pinned `README.md` dataset cards. Local checks used directory
metadata only. All byte counts below are from the immutable Hub tree metadata,
not a local download measurement.

## Candidate inventory

All three repositories are public, ungated, and enabled according to the
revision-qualified Hugging Face dataset API queried on 2026-08-11. Their cards
declare one public `test` configuration with `data/test-*.parquet`, binary
`label` (0 bona-fide, 1 spoof), `path`, and 16-kHz mono FLAC `audio` fields.
Thus, after an H10 freeze, each can use the existing two-pass contract:

1. hash/copy/fingerprint only `path,audio` into a label-free record manifest;
2. only after record-manifest hashing, map `path,label` into a separate sealed
   label artifact; and
3. write all predictions before one terminal label join.

The H9 materializer is deliberately hard-coded to SONAR/ArAD and therefore
**must not be repurposed as-is**. An H10-specific generic adapter must be
unit-tested on synthetic fixtures, committed, and protocol-bound *before* it
opens an H10 target path. It should accept only the exact pinned revision,
declared shards, FLAC payloads, integer binary labels, and an empty fresh HDD
output directory. Existing candidate snapshots expose `data/` rather than
H9's target-specific `<revision>/raw/data` layout, so the H10 adapter needs a
separate, explicitly hashed input-root contract; it must not infer a path.

| Priority | Candidate | Immutable Hub revision | Card population / language / generation scope | Public license | Exact packaged audio shards | H10 12-checkpoint prediction volume | Readiness |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | `SpeechAntiSpoofingBenchmarks/LibriSeVoc` | `13d69268a57afc7ebba2421d2a31730a390f1ff2` | 18,487 = 2,641 bona-fide + 15,846 spoof; English; official test split: one LibriTTS item re-synthesized by six vocoders (DiffWave, MelGAN, Parallel WaveGAN, WaveGrad, WaveNet, WaveRNN). | CC BY-SA 4.0 | 8 Parquet shards, **3,213,198,353 B** (3.21 GB decimal) | 221,844 clip-checkpoint passes | **GO, preferred single fresh target.** |
| 2 | `SpeechAntiSpoofingBenchmarks/DECRO` | `e54c29df5b81bac75e5cc294d1ab0d4db72a5c71` | 37,314 = 10,415 bona-fide + 26,899 spoof; English + Chinese evaluation subsets; TTS/VC systems. | CC BY 4.0 | 8 Parquet shards, **2,740,632,177 B** (2.74 GB decimal) | 447,768 clip-checkpoint passes | **Conditional GO** as a second H10 target only after the label-free throughput preflight below. |
| 3 | `SpeechAntiSpoofingBenchmarks/CFAD` | `53d7855c1c378524f7b7b1030bcb6b2caa327fe6` | 62,999 = 20,999 bona-fide + 42,000 spoof; Mandarin; clean `test_seen` and `test_unseen` partitions. | CC BY 4.0 | 14 Parquet shards, **4,186,887,968 B** (4.19 GB decimal) | 755,988 clip-checkpoint passes | **NO for a four-hour initial H10 terminal run**; retain as a preregistered later confirmation option. |

The full revision storage values exposed by the Hub API are 3,213,377,329 B
(LibriSeVoc), 2,740,938,114 B (DECRO), and 4,187,271,078 B (CFAD), respectively;
the small difference from the audio-shard totals is repository/card/license/
label/evaluation metadata. Labels-table byte sizes listed in the project's
prior Arena registry are 178,976 B, 305,937 B, and 383,110 B, respectively.
These are integrity-planning facts, not label reads in this audit.

## Why LibriSeVoc is the primary recommendation

LibriSeVoc is the smallest candidate, has explicit documented same-utterance
vocoder re-synthesis provenance, and differs from H9's two terminal corpora.
It gives an interpretable test of whether a source-trained same-item ranking
objective transfers to an independent English vocoder setting without using a
target pairing key in training or target selection. Its 2,641 base utterances
also make a future *predeclared descriptive* base-ID clustered uncertainty
analysis feasible, but that analysis must be written into H10 before any H10
target label is opened; it is not authorized by this readiness note.

At H9's frozen batch size 24, distributing one fixed seed per GPU means
55,461 inference passes per GPU for LibriSeVoc (three methods sequentially per
seed). This, plus about 3.21 GB of source shards and a 2,000-replicate terminal
bootstrap, is comfortably scoped for a four-hour initial run on the four
available RTX 6000 Ada GPUs, provided the preflight passes. DECRO would raise
this to 167,403 passes per GPU when paired with LibriSeVoc and is therefore
conditional rather than automatic. CFAD would exceed the bounded initial
budget once its 4.19 GB materialization and 755,988 total passes are included;
do not start it as an opportunistic third target.

## Required preflight and label firewall

The four-hour statement is a resource plan, not a claimed measured benchmark.
Before any candidate audio or labels are materialized, H10 must commit a
target-free protocol that fixes:

1. **Target panel:** LibriSeVoc alone for the bounded initial replication, or
   LibriSeVoc+DECRO only if an entirely label-free, metadata-only capacity
   check proves the fixed four-GPU schedule will finish inside four hours. It
   may not choose the panel from H9 result magnitude or from candidate model
   scores.
2. **Checkpoint identity:** reuse H9's already sealed 12 source checkpoints
   only if the new H10 protocol declares this as an *external replication of
   the frozen H9 method*. Otherwise it must repeat the complete source-only
   source/lambda/checkpoint ledger before target access. In neither case may
   H9 SONAR/ArAD enter any H10 decision.
3. **Device schedule:** three method evaluations per fixed seed on one assigned
   GPU, BF16, batch 24, with no batch, seed, or target-subset change after the
   preflight. The preflight is allowed to use only synthetic or source
   waveforms; it must record wall time and maximum memory. It must not decode
   candidate target audio.
4. **Materialization:** all declared Parquet filenames, byte sizes, SHA-256
   hashes, Arrow schemas, and record counts must be written from the pinned
   input root. The record phase opens `path,audio` only and creates opaque
   sample IDs/fingerprints. The label phase is separate, hash-sealed, and may
   open `path,label` only to create `sample_id,label`. `notes`, target strata,
   and Arena score artifacts are excluded from selection and the
   record/prediction stages.
5. **Terminal command:** after source/adapter ledgers are sealed, run every
   predeclared method/seed/target once, write a complete raw prediction table,
   check exact canonical source--target waveform collisions, then join labels
   once for all predeclared metrics and shared-ID/bootstrap contrasts. No
   target-specific rerun, threshold, orientation, or per-target rescue is
   permitted.

No actual candidate target rows should be opened until the H10 protocol and
synthetic adapter tests are committed. The existing directory presence under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/` does not waive this
firewall.

## Leakage and interpretation caveats

- **H9 target independence:** none of the candidates is SONAR or ArAD, but
  H10 still must avoid consulting any H9 target result while deciding whether
  to include LibriSeVoc versus DECRO. Freeze the panel and decision rule first.
- **Historical-checkpoint contamination:** the public Res2TCNGuard model card
  contains historical Arena evaluations for these candidate datasets. Its
  bundled ASVspoof2019-LA-trained checkpoint is prohibited, exactly as in H9.
  H10 must use only fresh-initialized architecture code or the already sealed
  H9 fresh-source checkpoints; it must never load the bundled checkpoint or
  use model-card outcomes to select a target/method.
- **Prior project exposure:** DeepVoice, the five H1 corpora, and the H9
  targets are not candidates for a clean new terminal firewall because this
  project has already used their metadata or labels in earlier loops. The
  recommended panel avoids them. The repository-wide historical download of
  other public snapshots is not evidence of a label firewall; H10 provenance
  must state exactly which processes previously had access.
- **DECRO overlap risk:** its card says the English bona-fide subset comes from
  ASVspoof2019 LA. The H9 source is ODSS, not ASVspoof2019 LA, but H10 must
  retain the exact canonical waveform-collision hard stop and disclose that
  the broader project previously audited ASVspoof2019 LA. This is why DECRO
  is secondary despite its useful bilingual scope.
- **LibriSeVoc lineage:** source and target share the broad vocoder-resynthesis
  paradigm, but exact source/target content, speaker, and upstream-corpus
  lineage are not ruled out by a PCM fingerprint. Do not call it an
  independent-speaker or independent-text replication without a separately
  frozen metadata audit.
- **CFAD partitioning:** its documented `test_seen` and `test_unseen` are
  structurally heterogeneous. Any partition breakdown must be declared before
  labels are opened and cannot replace its all-trials primary endpoint.

## Evidence records

- Project registry (revision, labels checksum, local destination):
  `data/arena-index.yaml`.
- Primary cards, pinned to the revisions above:
  `https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/LibriSeVoc/blob/13d69268a57afc7ebba2421d2a31730a390f1ff2/README.md`,
  `https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/DECRO/blob/e54c29df5b81bac75e5cc294d1ab0d4db72a5c71/README.md`,
  and
  `https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CFAD/blob/53d7855c1c378524f7b7b1030bcb6b2caa327fe6/README.md`.
- Revision-qualified repository metadata and recursive-tree APIs supplied the
  public/gated/disabled state, resolved commit, shard count, and byte sizes.
- Fresh-initialization/checkpoint restriction:
  `experiments/h9_paired_counterfactual/DATA.md` and the pinned
  `SpeechAntiSpoofingBenchmarks/Res2TCNGuard` model card at revision
  `4624265fa5e88c0abe425e37c278f3a9288aa914`.
- Current firewall implementation to adapt only under a new H10 freeze:
  `src/h9_pcr_target_materialize.py` and
  `experiments/h9_paired_counterfactual/TARGET_MATERIALIZATION.md`.

## Go / no-go

**GO:** declare a new H10 protocol for a single, independently pinned
LibriSeVoc terminal replication. Keep the H9 method/source decision frozen,
use a new H10 adapter/ledger, run only its all-trial terminal evaluation, and
report the result whether positive or negative.

**Conditional GO:** add DECRO only if a target-free four-GPU preflight is
committed and shows the *entire* LibriSeVoc+DECRO materialization/inference/
bootstrap budget fits four hours with adequate margin. Do not use any target
score, label, or preliminary prediction for that decision.

**NO:** do not include CFAD in the initial four-hour H10 run; do not run any
candidate before the new protocol, adapter tests, source ledger, and target
panel are frozen; and do not reopen H9 SONAR/ArAD for selection or comparison.
