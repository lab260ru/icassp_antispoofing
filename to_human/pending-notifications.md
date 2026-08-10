# Pending notifications

## 2026-08-09 — bootstrap

Research bootstrap is complete: the workspace, reproducible protocols, HDD
registry, and four-GPU plan are ready. Public Hugging Face access is being
tested next. Telegram delivery is queued because `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` are not currently present in the runtime environment.

## 2026-08-09 — first validated artifact

The Arena catalogue is pinned and public downloads work. ASVspoof 2019 LA is
downloaded at its Arena revision; all 71,237 published Spectra-AASIST scores
joined exactly to labels. The ongoing work is the audio-feature pilot and the
remaining model/dataset downloads. GitHub pushing remains queued until a valid
runtime token is available.

## 2026-08-09 — complete first score panel

All 72 pinned Arena score/result artifacts for the five core datasets are now
downloaded with byte-size validation. The first full analysis panel covers
eight detectors and 71,237 ASVspoof2019 LA trials per detector (569,896 score
rows); the CPU feature extraction is the remaining prerequisite for the first
association result. Delivery remains queued because the Telegram credentials
are not present in the runtime environment.

## 2026-08-09 — H1 discovery data ready

The locked ASVspoof2019 LA discovery subset is fully extracted: 10,000 balanced
utterances and three reproducible views (30,000 rows). Before creating any
association table, the analysis caught and stopped on a self-control edge case
for the loudness feature. The correction is under test; no provisional result
was kept from that failed invocation.

## 2026-08-09 — first H1 discovery result

The corrected ASVspoof2019 LA screen completed 1,344 within-class association
tests across eight published detectors, 28 signal descriptors, and three
waveform views. Crest factor is strongly negatively associated with spoof
evidence in Spectra-AASIST (and some other architectures), but it is not
directionally uniform across the panel. This is a promising heterogeneous-
sensitivity result, not yet a cross-dataset or causal shortcut finding. The
two remaining discovery corpora are now the priority before any candidate is
frozen.

## 2026-08-09 — initial paper checkpoint

The initial ICASSP draft is now compiled as a four-page IEEE-format PDF with
verified citations, the full first-discovery heatmap, and explicit limits on
what the current result can support. The H2 runner/quality-gate audit is also
preserved. The mandated three-reviewer launch was attempted but the local Claude
CLI is not authenticated, so all three subprocesses failed before reviewing;
this is recorded transparently and can be rerun after `claude /login`.

The worktree has committed checkpoints, but remote push and this Telegram
notification are still queued until valid runtime credentials are supplied.

## 2026-08-09 — second H1 discovery corpus

The unchanged H1 screen is complete on ASVspoof2021 LA. Spectra-AASIST's
spoof-class crest association remains negative but is much weaker than in
ASVspoof2019 LA (-0.099 vs. -0.454), and the multi-model pattern is
heterogeneous. I also corrected an output-path bug before it could obscure
provenance: each corpus now has immutable scoped tables and the aggregate tool
refuses to call the five-dataset portability criterion early.

## 2026-08-09 — H2 scorer parity gate

Two executable GPU scorers are now validated against their published Arena
ordering on a frozen 128-clip calibration set: AASIST uses raw waveform input
(rho 0.999994) and Spectra-AASIST uses external pre-emphasis 0.97 (rho
0.999971). This is only a baseline-parity gate, not an intervention result. H2
still waits for the H1 candidate freeze, ASR WER quality gate, and a complete
four-model scorer panel.

## 2026-08-09 — validated H2 inference settings

On the frozen parity set, the CUDA batch sweep selected initial model-specific
settings: Spectra-AASIST batch 8 (236.32 clips/s) and AASIST batch 2 (333.44
clips/s). These are stored as H2 inference engineering measurements; they do
not replace the separate BF16 batch/worker sweep required if H3 training starts.

## 2026-08-09 — literature and paper context checkpoint

The paper now incorporates source-verified ASVspoof 2021 and cross-domain
generalization context, with a compiled PDF and a paper-local citation ledger.
The framing remains intentionally conservative: cross-corpus differences
motivate this audit but do not validate any cue as causal. The ASVspoof2021-DF
audio download is continuing; remote push and Telegram delivery remain queued
because no runtime credentials are available.

## 2026-08-09 — third discovery corpus and H2 candidate freeze

ASVspoof2021-DF is complete despite a recoverable Hub decoder error on its
final protocol file. The locked 10,000-sample H1 screen adds 1,344 tests and
shows Spectra-AASIST's crest-factor association remains negative but attenuates
to -0.065, whereas AASIST reverses direction. The explicit five-dataset
portable rule remains unevaluable without held-out data. A disclosure-marked
exploratory freeze admits only the Spectra-AASIST crest-factor family for H2;
it is not a causal or portability conclusion. Telegram and GitHub push remain
queued pending runtime credentials.

## 2026-08-09 — H2 quality-gate runtime

The H2 ASR/WER component is now reproducibly pinned: OpenAI Whisper `small.en`
with a verified checkpoint hash, deterministic normalized WER, and FP16 GPU
mapping documented. This is only infrastructure; no audio has been transcribed
and no quality gate or causal conclusion has passed. Held-out audio processing
continues, while the multi-model transformed-audio panel is still incomplete.

## 2026-08-09 — Res2TCNGuard parity diagnosis

The downloadable Res2TCNGuard ONNX export failed the strict published-score
parity check, so it is blocked and its failure artifacts are retained. The
exact source PyTorch evaluator and checkpoint passed the same frozen 128-clip
check (rho 0.999994), which isolates an ONNX export/runtime divergence rather
than an input/window error. This adds a third parity-validated scorer but is
not transformed-audio or causal evidence.

## 2026-08-09 — H2 pre-score panel and held-out extraction

The first H2 cohort is frozen from ASVspoof2019 labels only: 500 bona fide and
500 spoof clips, with two crest-factor DRC arms and two registered controls.
No waveform or detector was used to choose it. Res2 source-PyTorch batch 2 is
the recorded fastest safe setting. Both held-out data sets are downloaded;
their no-reselection candidate declarations are committed, and their CPU
feature extractions are in progress. Telegram/GitHub delivery remains queued
until runtime credentials are available.

## 2026-08-09 — H2 quality-screen correction

The first detector-free H2 pair-quality run was intentionally stopped after
133 of 4,000 checkpoints: its clipping rule wrongly rejected recordings that
were already peak-clipped before a registered control transformation. No
detector was loaded or scored, and the preserved partial HDD artifacts are
explicitly diagnostic rather than results. The protocol is corrected and
committed (`eb6298c`); a distinct GPU-3 successor is running the pre-score
quality screen. No causal claim or detector-score analysis will begin until a
complete quality manifest is frozen.

## 2026-08-09 — H1 held-out confirmation and paper pivot

Both held-out H1 screens are complete and committed. The discovery-frozen
Spectra/crest candidate is adjusted-negative but tiny in InTheWild
(`rho=-0.032606`, `q=0.000523`) and adjusted-null in ASVspoof5
(`rho=0.001383`, `q=0.937089`), despite a negative raw ASVspoof5 association.
The explicit five-corpus aggregate is now evaluable: 19 of 168 registered
feature/view/class units meet the descriptive association rule, but crest does
not. The paper is pivoting toward a conservative negative crest-factor result
and association atlas; no post-hoc H2 feature is added. H2 quality screening
continues detector-free. Send this update when Telegram credentials are present.

## 2026-08-09 — H2 quality-gate outcome

The completed detector-free crest intervention panel is blocked before scoring:
DRC-3 retains 18.1%, DRC-6 0.6%, and the +0.1 dB control 64.6%, below the
locked 90% per-arm quality requirement; polarity alone retains 99.9%. The
formal freeze refused to create a score-eligible manifest, so no model saw a
transformed waveform and no causal claim is possible. We will preserve this as
a rigor-preserving negative result rather than tune thresholds after inspection.

## 2026-08-09 — audited four-page paper checkpoint

The ICASSP draft is now a compiled, readable four-page PDF backed by a
five-corpus H1 result table and vector association atlas. A claim-to-artifact
audit cross-checks the central reported quantities and makes an important
slice distinction explicit: the failed frozen H2 candidate is spoof-class
crest factor, while a separate bona-fide/full-waveform crest association is
descriptive only and is not reused post hoc. The external reviewer runtime is
still unauthenticated, so this is an internal evidence audit rather than peer
review. New local commits are `79ae170`, `f1b920b`, and `0c75633`; push remains
queued until a runtime GitHub token is available.

## 2026-08-09 — H2B clean-room calibration boundary

The next intervention loop is now independently reproducible without changing
the failed H2 panel. The public DeepVoice dataset was pinned and acquired on
the HDD; a 256-clip (128 per label) manifest was frozen from labels only,
with byte-bound CSV/provenance hashes. No waveform was decoded and no feature,
ASR, transform, detector, or score artifact was touched. The next task is to
commit a finite quality-only transform calibration before it can run. This
update and the new local commits remain queued for Telegram/GitHub credentials.

## 2026-08-09 — H2B Q1 negative feasibility result

The precommitted, detector-free DeepVoice Q1 calibration completed all 2,304
waveform/ASR quality pairs (256 frozen clips × 9 arms). It selected no
non-control transformation family: the closest, -0.5 dB/oct spectral tilt,
has a 95% Wilson lower retention bound of 0.89624, below the locked 0.90 gate;
endpoint zeroing retains 15/256 pairs. No detector was imported or run, no
scores/EER/causal effect were estimated, and Q2--Q4 will not begin from this
loop. The result, selection table, hashes, and resumability notes are committed
under `experiments/future_directions/results/`. Send this update when Telegram
credentials are present.

## 2026-08-09 — H4 score-free transportability result

A new precommitted, score-free five-corpus feature--label atlas is complete:
all 420 cells and 84 aggregations used only the frozen feature tables, not
Arena scores, models, audio, or ASR. Twenty-three units meet its explicitly
terminal descriptive rule. Crest factor itself again demonstrates corpus
dependence: full-waveform separation reverses sign in InTheWild and is near
null in ASVspoof5. This does not create a detector claim or a new intervention
candidate; the run note, full tables, hashes, and protocol are preserved. Send
this update when Telegram credentials are available.

## 2026-08-09 — final local initial-package handoff

The current working draft was rebuilt from clean TeX intermediates and is a
readable four-page US-letter PDF (Table I page 2; References begin page 3); all
87 local tests passed. The static preflight also verifies embedded fonts and
internal-draft anonymous PDF metadata. `to_human/FINAL_HANDOFF_20260809.md` and
`verification/FINAL_VERIFICATION_20260809.md` identify the exact evidence,
negative-result boundaries, and safe continuation conditions. The required
paper-review launcher was executed but remains unauthenticated, so no review
verdict is claimed. Send this update only after Telegram credentials are
configured in the environment.

## 2026-08-09 — official template access blocker

The ICASSP author page currently specifies the paper limit but does not expose
an ICASSP-2027-specific LaTeX archive. Its linked generic IEEE conference ZIP
was located, but a direct download from this runtime receives a CloudFront WAF
challenge (HTTP 202), so no template was downloaded or substituted silently.
Please provide an approved archive or access path if template reconciliation is
needed before submission; the exact URL and condition are in
`paper/submission-requirements.md`.

## 2026-08-09 — author-block requirement discovered

The official ICASSP 2027 editorial policy specifies **single-anonymous**
review: reviewers see author names. The present anonymous author block is an
internal draft placeholder and must not be uploaded. Please provide the
authorized author names, affiliations, and ordering; the exact replacement and
rebuild/preflight steps are in `paper/AUTHOR_BLOCK_REQUIRED.md`. This does not
alter any research result.

## 2026-08-09 — current paper-review runtime blocker

After correcting the author-policy status and rebuilding the PDF, the mandated
three-reviewer paper-review launcher was run again on the exact current draft.
All reviewers launched on schedule but exited before reviewing because the
local Claude CLI is not authenticated. The current bundle is preserved in
`paper/reviews/post_author_policy_20260809/`; no peer-review verdict is
claimed. Authenticate the local reviewer runtime to unblock a substantive
review.

## 2026-08-09 — portable repository backup

While GitHub authentication remains unavailable, a complete-history Git bundle
of `research/icassp-signal-audit` was created and verified on the HDD. Its
SHA-256, exact head, and restore command are in
`to_human/GIT_BUNDLE_HANDOFF_20260809.md`. This is a resilience backup only;
the branch still needs a normal authorized GitHub push.

## 2026-08-10 — named-author paper checkpoint and GitHub access needed

The ICASSP draft now uses the supplied ICASSP-2026 spconf/IEEEbib template,
includes the authorized Kirill Borodin dual affiliation, and has a rebuilt
four-technical-page plus references-only-fifth-page PDF. It passes 87 tests
and the local layout/font preflight; three fresh Codex-only internal reviews
are preserved and their actionable corrections are applied. The local commit
is `78a79e9`. A single normal GitHub push using the current local .env was
rejected as an invalid credential; please replace it with a repository-write
token for lab260ru/icassp_antispoofing. No token was exposed or retried.

## 2026-08-10 — GitHub push succeeded

The replacement local GitHub credential was verified and a normal non-force
push succeeded. The research/icassp-signal-audit branch is now on
lab260ru/icassp_antispoofing, including the named-author ICASSP template,
compiled paper PDF, Codex-only review bundle, reproducibility notes, and the
Telegram delivery receipt. No research gate was reopened.

## 2026-08-10 — H5 score-free view-invariance result

The independently frozen, label-free H5 audit is complete: all 420 planned
feature/view/corpus cells and all 84 aggregations were written. Seventeen
view-pair/feature units meet the strict descriptive stability rule. Six
InTheWild silence/clipping cells had no finite paired values and are retained
transparently rather than repaired. This helps delimit which feature
measurements are view-sensitive, but it does not claim detector reliance,
causality, or a preferred preprocessing method. The full hash-bound ledger is
in `experiments/h5_view_invariance/results/H5_ANALYSIS_001.md`.

## 2026-08-10 — H6 score-agreement atlas and H5 supplement

Two bounded follow-ups are complete. H6 used a newly frozen label-only panel
and the original published score artifacts: all 280 planned within-class
model-pair cells completed with 5,000 exact joins each, and their agreement
ranges from -0.629 to 0.869 across the fixed corpus/class registry. This is
descriptive architecture-diversity context, not a model ranking or causal
claim. I also rendered a hash-bound H5 supplementary heatmap showing all 84
predeclared feature/view units; it makes the 17 terminal descriptive stable
units and six unavailable cells explicit without selecting a cue. Both ledgers
are committed; the H6 result is in
`experiments/h6_score_agreement/results/H6_ANALYSIS_001.md`.

## 2026-08-10 — compiled paper audit checkpoint

The manuscript has completed a fresh three-perspective Codex-only audit.
Reviewers verified the central H1/H2 values and agreed to keep H5/H6
supplementary-only. I tightened the paper’s portability-rule, partial-control,
bootstrap, released-score, exploratory-crest, and conditional-H3 wording
without adding a causal claim. `paper/main.tex` was rebuilt immediately: the
tracked PDF has four technical US-letter pages plus a references-only fifth
page, and the local table/font/layout check passes. The paper and review
resolution are committed; remaining submission conditions are the official
ICASSP-2027 template/checker and an authorized archival artifact locator.

## 2026-08-10 — sealed H6 supplementary figure and handoff

I completed a display-only H6 supplementary heatmap after committing its exact
three-input hash/schema contract. It shows all 280 completed within-class
model-pair agreement cells in fixed registry order, with no model ranking or
new experiment. The vector PDF, 300-DPI PNG, metadata, and visual-inspection
record are on the HDD and their compact ledger is committed. I also refreshed
`AGENTS.md` and the future-work roadmap so a new agent cannot accidentally
reopen H1/H2/H2B; the branch is pushed through `0920e74`. The full repository
suite currently passes 115 tests. The compiled main paper remains unchanged
and readable.

## 2026-08-10 — reviewed main evidence-boundary paper revision

The compiled ICASSP working draft now foregrounds its central evidence boundary:
Figure 1 directly shows the discovery-frozen five-corpus crest association and
the separate one-corpus, detector-free H2 quality stop. A Codex-only
three-perspective review found only presentation risks; a separately
hash-pinned v2 layout now visibly separates discovery from held-out rows,
names the 1,000-clip ASVspoof2019 LA H2 scope, and explains that no
score-eligible retained-pair manifest was frozen. It adds no causal claim or
new experiment. The readable five-page PDF, both preserved figure iterations,
and the review/resolution bundle are committed and pushed at `03dc6fe`; all
134 repository tests and the local PDF preflight pass.

## 2026-08-10 — H1 adjustment disclosure completed

I completed a protocol-locked, score-free disclosure audit for the paper's H1
partial-adjustment design. It writes all 10 corpus-by-label metadata-coverage
rows from five byte-hashed feature cohorts and records the sealed held-out
bootstrap cluster rule/counts. It does not load score or feature values, rerun
a bootstrap, select a cue, or change the negative H1/H2 result. The full suite
now passes 139 tests; the artifact and resumability notes are committed and
pushed at `6a5b16e`.

## 2026-08-10 — paper reproducibility disclosure integrated

The compiled ICASSP draft now points directly to the sealed H1 corpus/class
covariate-coverage and held-out bootstrap-context material. This resolves a
methods-reporting concern without adding a new statistic, experiment, causal
claim, or supplementary-result dependency. `main.tex` was rebuilt immediately:
the PDF remains five US-letter pages, and the local preflight plus all 139 tests
pass. The refreshed source/PDF and verification ledger are committed and pushed
at `2d24c42`.

## 2026-08-10 — reproducibility release candidate prepared

I prepared owner-safe archival readiness for the ICASSP working draft:
`CITATION.cff`, a release checklist, corrected current scope metadata, and an
explicit exclusion list for raw audio, model weights, HDD products, and all
credentials. This is not a public release, DOI, or claim that an artifact
locator exists; those remain owner/license/template-reconciliation decisions.
The preparation is committed and pushed at `7465c87`.

## 2026-08-10 — ASVspoof5 paper citation verified and integrated

The paper now cites the canonical ASVspoof 5 dataset/challenge paper at its
held-out-corpus mention. The BibTeX was retrieved through the DOI endpoint and
checked against the ISCA Archive record; no quantitative or causal paper claim
changed. `main.tex` was rebuilt immediately and remains within five US-letter
pages; the local preflight and 139-test suite pass. The source, bibliography,
citation ledger, and compiled PDF are committed and pushed at `0f349d3`.
