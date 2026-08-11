# H10 methodology replication audit — 2026-08-11

**Status:** prospective, read-only decision record.  It neither changes the
sealed H9 protocol/results nor authorizes a fit, target download, or paper
claim.  Its purpose is to identify the smallest independently predeclared
extension that could turn H9's single-backbone/two-target finding into a more
credible ICASSP revision.

## Decision

The best next study is **H10-A: fresh-initialized AASIST replication with a new
external target package**.  It should ask whether the *same-item rank-edge
relation*, rather than the H9 Res2TCNGuard implementation, improves transfer
when the detector is an independently implemented spectro-temporal graph
attention network.

It must retain the H9 result as a completed first study.  H10-A is a new,
prospective replication—not an ablation, a re-run, or a replacement for H9.
The only combined paper claim allowed if H10 passes is that the effect was
observed in **two separately locked ODSS studies, two fresh architectures, and
four fixed external corpora**.  It must still report the individual H9 and H10
study estimates and must not pool them as though their seeds or targets were
exchangeable.

## Why this is the credible extension

H9 has a strong within-study control: B1, B2, and P share 23,883 paired-only
ODSS trials, 15,922 rank edges for both B2/P, selection logic, and four fresh
seeds.  Its limitation is external validity: one 172k-parameter
Res2TCNGuard architecture and SONAR/ArAD only.  The terminal red-team audit
explicitly identifies this as the principal limitation.

The official AASIST repository is a genuine, trainable PyTorch implementation,
not merely a published-score artifact.  It documents training AASIST plus
RawNet2/RawGAT-ST baselines, a model extension interface, and an MIT licence
([repository](https://github.com/clovaai/aasist), inspected 2026-08-11).
Its current `main` commit at audit time is
`a04c9863f63d44471dde8a6abcb3b082b07cd1d1`.  In contrast, this repository's
pinned Arena AASIST asset is only `aasist.onnx`; W2V2-AASIST and XLSR-SLS have
no local runnable source bundle.  Therefore an AASIST fresh-init replication
is feasible, while treating any Arena checkpoint/ONNX artifact as a trainable
backbone would violate H9's fresh-init logic.

The target package proposed below is deliberately new to this project’s H8/H9
terminal panels.  It improves on merely running AASIST over SONAR/ArAD after
their labels have already been disclosed.  Broad external evaluation is
important because the Speech DF Arena itself reports high out-of-domain EERs
for many systems ([Dowerah et al., 2026](https://doi.org/10.1109/OJSP.2026.3652496)),
and source/target language and generator differences materially affect speech
deepfake transfer ([Wang et al., 2025](https://aclanthology.org/2025.acl-long.493.pdf)).

## Recommended H10-A protocol

### Fixed data and architecture roles

| Role | Frozen item | Rationale and condition |
| --- | --- | --- |
| Source | The exact H9 ODSS source materialization, P edge CSV, B2 edge CSV, source voice split, and excluded-row ledger | Keeps the hypothesis fixed: a documented same-item partner versus a stratum-matched random partner.  Do **not** add unmatched VITS rows or alter B2. |
| New architecture | `clovaai/aasist` at `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`, initialised from random H10 seed weights | AASIST is a material architecture change with official source/training support.  Its supplied/pretrained weights, Arena ONNX, and any ASVspoof checkpoint are prohibited. |
| Target 1 | `SpeechAntiSpoofingBenchmarks/HABLA` at `764c00726cc4327ff2a270b8347375e95a7034db` | New Spanish/Latin-American-accent external corpus; public Hub metadata reports CC-BY-4.0 and 11.76 GB. |
| Target 2 | `SpeechAntiSpoofingBenchmarks/CD-ADD` at `b03c6cf3463d67c1525ba20738c495d120675876` | New English zero-shot-TTS external corpus; its primary paper describes five modern zero-shot systems and cross-domain evaluation ([Li et al., 2024](https://aclanthology.org/2024.emnlp-main.286.pdf)); Hub metadata reports CC-BY-4.0 and 3.89 GB. |

The two target choices use only public card/provenance metadata in this audit;
no target row, audio, label, published score, or target metric was accessed.
Before any H10 source fit, a new H10 plan must be committed and the two exact
target revisions must be recorded in an H10-only target table.  Do not replace
either target after seeing a source or target result.

### Methods, source selection, and evaluation

1. Implement an AASIST adapter in new H10-namespaced code.  It must consume
   the existing source manifest and the exact B1/B2/P sampler contracts, use
   the shared deterministic 64,600-sample first-window/tile policy, and expose
   a scalar **spoof** logit `z`.  Define `z` from the two AASIST class logits
   once in the plan and source-test its orientation; target labels may not
   change it.  H9 code and ledgers remain read-only.
2. Use BCE for B1; use the unchanged hinge
   `max(0, 1 - (z_spoof-z_bonafide))` for B2/P.  B2 must retain exactly every
   P spoof endpoint, the same 15,922 edge count, and its frozen H9 stratum
   constraints.  This preserves the attribution control rather than testing
   “a different loss on AASIST.”
3. Predeclare four new seeds, one physical GPU per seed.  Select the largest
   safe, fastest synthetic BF16 batch size and worker count before ODSS audio
   is decoded.  A source-only P lambda grid `{0.10, 0.30, 1.00}` with the same
   lower-mean-dev-EER/lower-lambda tie rule may be reused; its selected value
   applies unchanged to H10 P and B2.  All other model recipe choices need a
   finite, source-only table and tie rules in the protocol—not target tuning.
4. A maximum-epoch budget and source-dev checkpoint rule must be chosen before
   a target path is touched.  Prefer one finite, official-AASIST-compatible
   recipe plus early source-dev checkpointing over a wide architecture search.
   The study should record the complete 12 P source-selection fits and the 12
   final B1/B2/P fits, as H9 did; never select an individual seed from a target.
5. Only after a complete H10 checkpoint ledger is sealed, materialize both
   targets in two passes: `path,audio` to a score-free audio/record manifest,
   then `path,label` to a separate label artifact.  A generic H10 materializer
   is needed because the H9 materializer is deliberately hard-coded to
   SONAR/ArAD.  It must be tested on synthetic Parquet fixtures before it sees
   a target path.
6. Evaluate all 12 H10 checkpoints on both targets in one terminal call.  The
   evaluator must write complete target predictions before loading labels,
   calculate the four-seed probability ensemble, report EER/AUROC/per-seed
   values, and use 2,000 shared-ID, label-stratified bootstrap replicates.
   Apply the same target count weighting (unweighted two-target macro EER) to
   every method.

### Required integrity gates

- New source--target canonical 16-kHz PCM fingerprint audits must be zero for
  *each* H10 target.  In addition, check documented source-lineage conflict:
  ODSS contains VCTK, Hi-Fi TTS, HUI German, and OpenSLR Spanish; H10 cannot
  assume an audio fingerprint proves no upstream voice/text overlap.
- Before target data access, document why the HABLA and CD-ADD source corpora
  are not known to be the ODSS source recordings.  If either dataset card or
  upstream documentation establishes a common recording corpus, speaker list,
  or release-subset overlap that cannot be excluded from metadata, stop that
  target rather than calling it external.
- No target labels/scores may choose architecture revision, model width,
  crop, augmentation, batch size, optimizer, epoch, lambda, checkpoint,
  seed, or target inclusion.  Target-domain normalization, test-time
  adaptation, and pretrained SSL initialisation are out of scope.
- Pin every third-party file hash and commit; keep downloaded targets, cloned
  AASIST source, caches, checkpoints, and raw predictions under
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`.  Commit only manifests,
  code, hashes, compact metrics, and notes.

### Precommitted success and no-go rule

For a confirmatory **H10 architecture-and-target replication**, P must:

1. lower the H10 two-target ensemble macro EER by at least **10% relative**
   to each of B1 and B2;
2. have lower point-estimate EER than both controls on HABLA **and** CD-ADD;
3. have each paired bootstrap interval for `P-B1` and `P-B2` wholly below
   zero; and
4. pass every source/target/reconstruction/lineage gate above.

Failure of any condition is a complete H10 negative replication result.  It
does not permit a target swap, a smaller target subset, a new loss weight,
another architecture, a changed source pool, or a relaxation of the threshold.
H10's bootstrap remains conditional on its four trained seeds; report seed
heterogeneity instead of claiming a training-population confidence interval.

## Feasibility and staged compute plan

The only safe work before a protocol commit is dependency reconnaissance.  If
the protocol is approved and committed, use these gates in order:

| Stage | Target access | Expected cost | Stop condition |
| --- | --- | --- | --- |
| AASIST qualification | no ODSS/target audio; synthetic 64,600-sample inputs only | minutes on one GPU | BF16 forward/backward nonfinite, missing supported dependency, or no batch that fits safely |
| Throughput freeze | synthetic only | minutes | no reproducible fastest-safe batch/worker recipe |
| Source P grid | H9 ODSS source only | 12 fits, four GPUs in three waves | source ledger/reconstruction failure |
| Frozen final fits | H9 ODSS source only | 12 fits, four GPUs in three waves | any missing/failing seed or checkpoint ledger mismatch |
| Target materialization + terminal call | only after source ledger | at least 15.65 GB transfer plus materialization; full target evaluation | schema, license, lineage, collision, or prediction-reconstruction failure |

The official AASIST README reports about 16 GB for batch 24 on a V100, but it
also uses an old PyTorch 1.6/CUDA 10.1 environment.  Hence it is evidence of
plausible fit on the four 48-GB RTX 6000 Ada GPUs, **not** a substitute for a
current BF16 qualification measurement.  Do not import its random training
crop: the H10 deterministic crop is intentional so B1/B2/P differ only in the
edge relation.

## Alternatives considered and rejected

| Candidate | Decision | Reason |
| --- | --- | --- |
| Run AASIST on H9 SONAR/ArAD | Secondary robustness check only | Those labels are now disclosed.  Even with a fresh H10 model and a source-only fence, it is not an independently selected target replication and cannot establish genericity. |
| Use Arena AASIST/W2V2-AASIST/XLSR-SLS assets as the second backbone | Reject | Local AASIST is inference-only ONNX; W2V2-AASIST/XLSR-SLS directories are absent.  Reusing published/pretrained assets would confound fresh ODSS-only training. |
| Source replication with CVoiceFake-small or LibriSeVoc | Do not use as H10 primary | Both corpora were H8 terminal targets, so their labels/scores are historically exposed; CVoiceFake source pairing would also need a new verified counterpart builder.  This could only be a clearly retrospective future study. |
| DFADD as a new target | Reject | Its Hub card identifies it as VCTK-derived, while ODSS includes VCTK material.  Exact waveform collision testing cannot rule out voice/text lineage overlap, so it is too risky for the primary external claim. |
| Change PCR loss (contrastive, triplet, or learned pair mining) | Reject | The current literature already contains pairwise/contrastive speech anti-spoofing objectives.  Such a change would blur the H9 contribution and create post-result loss search instead of testing architecture/target robustness. |

## Novelty and reporting boundary

H10-A would strengthen the paper materially only if it passes the exact
replication gate.  The novelty is still the controlled comparison of a
documented same-item natural--synthetic edge against **equal-budget random
edges**, not AASIST itself or pairwise learning in general.  Related work
already includes Siamese/contrastive anti-spoofing and bonafide-pair learning;
the paper must retain that distinction.  If H10 fails, the correct paper is
the narrow H9 controlled-transfer result with an explicit failed replication,
not a claim of model-independent benefit.

## Sources and local evidence

- H9 authority: `experiments/h9_paired_counterfactual/PLAN.md`,
  `DATA.md`, `TRAINING_HARNESS.md`, and
  `results/H9_TERMINAL_EVALUATION_001.md`.
- Local asset audit: `data/arena-index.yaml`; model-directory listing on
  2026-08-11; `src/h9_pcr_training.py` and
  `src/h9_pcr_target_materialize.py`.  These establish the H9 trainer's
  Res2-only coupling and the hard-coded SONAR/ArAD materializer boundary.
- AASIST source/training interface: [official repository](https://github.com/clovaai/aasist)
  (commit recorded above); original architecture context:
  [Jung et al., AASIST](https://arxiv.org/abs/2110.01200).
- Cross-domain target motivation: [Li et al., 2024 CD-ADD](https://aclanthology.org/2024.emnlp-main.286.pdf),
  [Chen et al., 2025 SpeechFake](https://aclanthology.org/2025.acl-long.493.pdf),
  and [Dowerah et al., 2026 Speech DF Arena](https://doi.org/10.1109/OJSP.2026.3652496).
- Hub records inspected metadata-only on 2026-08-11:
  [HABLA](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/HABLA),
  [CD-ADD](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD),
  [DFADD](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/DFADD),
  [CVoiceFake-small](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CVoiceFake_small),
  and [LibriSeVoc](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/LibriSeVoc).
