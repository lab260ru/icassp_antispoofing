# Implementation Notes

## 2026-08-09 - ICASSP anti-spoofing autoresearch bootstrap

- Decision: Treat published SpeechAntiSpoofingBenchmarks Arena scores as authoritative inputs; analyze them but do not spend compute reproducing them.
- Decision: Use public core datasets first and reserve In-the-Wild and ASVspoof 5 as held-out confirmation data for feature selection made on the discovery datasets.
- Decision: Use one independent job per RTX 6000 Ada GPU; newly trained models use BF16 and per-backbone throughput measurement.
- Tradeoff: The current ICASSP 2027-specific kit is not yet available, so the initial manuscript will use the generic IEEE conference template and be migrated once the official kit is published.
- Constraint: Telegram, GitHub, and Hugging Face runtime credentials are not currently configured. Sensitive values are intentionally not retained here.
- Changed: The live Arena manifest pins dataset revisions that differ from the current heads of several dataset repositories. The catalogue therefore uses Arena-pinned revisions, not repository heads, for every audio/score join.
- Validation: Installed the public audio/analysis runtime; CUDA ONNX Runtime exposes TensorRT, CUDA, and CPU providers. The 28-feature synthetic-signal test passes after fixing the registry emission order.
- Validation: The first authoritative score artifact joined 100% of ASVspoof2019 LA trials by utterance ID; model-score polarity is normalized from labels and retained in the result metadata.
- Validation: The 200-utterance feature pilot produced three views per sample with no duplicate keys. Voice-quality measures are genuinely unavailable on some unvoiced examples, so downstream analysis preserves a missingness indicator.
- Validation: The score downloader now uses byte-validated `curl` range-resume transfers after the Hub client's compressed-transfer path failed. It completed all 72 score/result artifacts for the locked five-dataset, eight-model core panel. The ASVspoof2019_LA parquet panel has 569,896 rows and no duplicate `(model, sample_id)` keys.
- Validation: Full ASVspoof2019_LA extraction completed from the frozen 5,000-per-class manifest: 30,000 rows across three views, no duplicate `(sample_id, view)` keys. Dataset metadata includes 67 speakers but no attack identifier; its control column is deliberately represented as missing, not inferred.
- Guardrail: The first association invocation stopped before output when integrated loudness was both an outcome feature and a partial-correlation control, creating a duplicated column. The corrected estimator will exclude a tested variable from its own controls and receive a regression test before the same locked inputs are rerun.
- Validation: The corrected ASVspoof2019_LA H1 screen produced 1,344 exact-join tests (8 models × 2 classes × 3 views × 28 features). Spectra-AASIST spoof scores show large negative crest-factor partial correlations across the three registered views, while signs and sizes vary across architectures. The result is explicitly discovery-only.
- Guardrail: `partial_spearman` now removes the test/response variables from requested controls and deduplicates controls. A separate confirmation mode accepts only an explicit frozen candidate manifest, records its SHA-256, and uses deterministic label-stratified speaker/source-cluster bootstrap CIs (2,000 by default). Focused tests pass (`6 passed`).
- Paper: `paper/main.tex` is a compiled four-page IEEE-format initial draft. It embeds the complete first-discovery heatmap generated reproducibly from `association_summary.csv`, labels it as one-corpus discovery evidence, and cites only records in `paper/citation-verification.md`.
- Paper-review status: the required three-reviewer Sonnet launcher correctly spawned the three staggered processes but each returned `Not logged in · Please run /login`. `paper/reviews/initial_draft/REVIEW_STATUS.md` preserves this as an authentication failure. No peer-review verdict or meta-review is claimed; rerun only after the local Claude CLI is authenticated.
- H2 readiness: the audited document identifies executable initial paths for Spectra-AASIST, AASIST, and Res2TCNGuard, but transformed-waveform scoring must first pass a fixed 128-clip parity/orientation test. W2V2-AASIST ONNX and a fixed ASR WER runtime remain required for a confirmatory intervention panel.
- Validation: ASVspoof2021_LA full feature extraction and its 1,344-test H1 screen completed with the unchanged registry and estimator. The first generic output path would have overwritten the 2019 CSV; this was detected before interpretation, replaced by per-dataset output directories plus an explicit input-only aggregation utility, and both corpus artifacts were regenerated.
- Validation: H2 parity run_001 uses a frozen, score-independent 128-clip ASVspoof2019_LA calibration manifest. Its runner matches AASIST Arena raw-score ordering at 0.999994 with raw waveforms and Spectra ordering at 0.999971 only after external pre-emphasis 0.97. A bundled CUDA-13 library path was required for ONNX Runtime; the final recorded runs use CUDA devices 1 and 0 respectively with no CPU fallback.
- Validation: The parity-validated CUDA batch sweep measures Spectra-AASIST batch 8 as fastest over the frozen workload (236.32 clips/s) and AASIST batch 2 as fastest (333.44 clips/s). The sweep uses three repeats and candidate batches 1--32; it excludes model load/audio decode and does not claim peak VRAM or end-to-end H2 throughput.
- Constraint: This parity run establishes preprocessing and score orientation only. It does not justify an H2 causal arm, choose production batch sizes, or remove the fixed-ASR and fourth-scorer prerequisites.
- Validation: The pinned ASVspoof2021_DF corpus is complete (85 expected files). The Hub client raised a Brotli decoder error after transferring 84 files; a remote-tree audit identified the sole missing pinned protocol text, which was retrieved with identity encoding from the same immutable revision (`SHA-256 d65befecbac33a0b72101503b228119748f63a4e7f34f019f4e1dea55fb4961d`). All 81 audio/label Parquet files and all metadata/protocol files are present before H1 extraction begins.
- Validation: ASVspoof2021_DF extraction produced the locked 10,000-sample, 30,000-view-row feature artifact and 1,344-test H1 table; its 8-model score panel has 4,894,632 source rows and joins exactly to every selected feature view. The explicit three-discovery aggregate emits `not_evaluable` for the five-dataset portability criterion. A provenance-hashed, no-top-k freeze admits only Spectra-AASIST/crest-factor across the three discovery corpora; it must be committed before any bootstrap or H2 invocation.
- Follow-up: Download and analyze the held-out datasets only under their own corpus scopes; run manifest-selected bootstrap CIs and H2 only after the required source artifacts, ASR WER quality gate, and transformed-score runners are ready.
- H2 guardrail: `h2_quality_full_001` was deliberately stopped after 133/4,000 detector-free pair checkpoints because requiring both source and transformed clipping fractions to be zero incorrectly treated pre-existing source clipping as intervention harm. Preserve its HDD artifacts for diagnosis, never analyze them as quality or causal results, and use the corrected transform-induced clipping gate from commit `eb6298c`. The successor `h2_quality_full_002` runs on GPU 3 and remains strictly pre-score until its full quality manifest is frozen.

## 2026-08-09 - H2 quality-frozen paired scorer

- Decision: Require the committed retained-only quality-freeze CSV and its byte-hashed JSON report, then regenerate every selected source/transform pair and verify waveform hashes before constructing a detector.
- Decision: Keep the parity-validated panel at Spectra-AASIST (pre-emphasis 0.97, GPU 0, batch 8), AASIST (raw, GPU 1, batch 2), and source-PyTorch Res2TCNGuard (raw, GPU 2, batch 2). No fourth scorer is implied.
- Guardrail: The score runner refuses mutable quality-run tables, incomplete or below-90%-arm freeze reports, hash/identity drift, non-pinned GPU visibility, and any uncommitted protocol/freeze/parity artifact. It does not import or read Arena score outputs.
- Validation: Focused synthetic tests validate freeze-byte-hash rejection, regenerated-transform hash rejection before scorer construction, immutable resumability, pair delta materialization, and GPU assignment checks (`17 passed` across paired-scoring/quality-freeze/quality-runner/pre-score suites). No corpus or production detector scoring was run.

## 2026-08-09 - H1 held-out confirmation and outer-loop pivot

- Validation: InTheWild and ASVspoof5 each completed an explicit 1,344-cell H1 screen with exact eight-model joins, plus only their own predeclared Spectra/crest candidate bootstrap (2,000 valid clustered replicates each). The finite-value AUROC guard was committed and tested before the InTheWild rerun; source features were not changed.
- Decision: Preserve raw and registered adjusted estimates side by side. The ASVspoof5 raw negative association does not replace its null adjusted result; the InTheWild adjusted-negative result does not replace its null raw result.
- Decision: Apply the locked five-corpus partial-Spearman/BH aggregation once, with five explicit source tables and no candidate selection. It reports 19 qualifying registry units, but the operational H2 crest candidate fails the two-confirmation requirement.
- Tradeoff: Continue the already-frozen crest H2 execution only as an explicitly exploratory paired-sensitivity test. Do not expand its arms/features after the five-corpus screen, and do not begin H3 mitigation training from the failed crest-portability premise.
- Validation: The aggregation writes 1,344 model/unit and 168 feature/view/class rows, records `selection_or_freezing_performed: false`, and has a committed result note with all output hashes.

## 2026-08-09 - H2 full quality gate stops before detector scoring

- Validation: The corrected `h2_quality_full_002` run completed exactly 4,000 immutable checkpoints over the frozen 1,000-clip, four-arm input panel. The formal quality-freeze CLI verified completion/provenance and refused at `drc_cf3` (181/1,000 retained), before constructing a score-eligible manifest.
- Decision: Do not relax the STOI/WER/loudness/clipping/target-direction thresholds, or alter these arms, after observing the quality diagnostics. The H2 paired scorer remains unused.
- Tradeoff: The paper can report a stronger rigor result---the intervention could not clear its own content-preservation gate---but no score-delta or causal conclusion. Any reparameterized DRC or different feature requires an outer-loop protocol and fresh score-independent freeze.

## 2026-08-09 - Updated paper-review attempt

- Validation: Extracted 16,337 characters from the compiled four-page PDF and launched the mandatory three-reviewer `paper-review` panel with the requested stagger (0/10/20 seconds).
- Constraint: Every reviewer exited with `Not logged in · Please run /login`; the bundle contains only these authentication failures, no verdicts or review content. Do not represent the updated paper as peer reviewed.

## 2026-08-09 - Evidence-conservative paper audit

- Validation: An internal claim-to-artifact audit in `paper/reviews/five_corpus_h2_quality_20260809/INTERNAL_EVIDENCE_AUDIT.md` maps the paper's central matrix count, crest estimates, 19-unit registry count, and H2 quality-gate disposition to committed result files and their recorded hashes. It is not an external review or an ARA Seal assessment.
- Correction: The draft no longer says held-out confirmation is pending. It now distinguishes the frozen spoof/full-waveform crest candidate (not portable and blocked before H2 scoring) from the different bona-fide/full-waveform crest-factor unit that meets only the descriptive H1 rule. The latter must not be retroactively substituted into H2.
- Validation: Every `main.tex` revision is compiled with the repository-pinned template. The current `paper/build/main.pdf` is readable and four pages; Tectonic only reports known underfull-box and bibliography-rerun warnings.

## 2026-08-09 - Paper scope and layout correction

- Audit: A paper-review-criteria read-only audit found that the planned four-model/multi-corpus H2 specification could be read as completed, generic crest-factor language could conflate two distinct slices, and Table I had floated after the paper text. The external reviewer runtime remains unauthenticated; this is not a peer-review verdict.
- Correction: The paper now distinguishes planned confirmatory H2 from the completed 1,000-clip score-blind quality screen, writes DSP sensitivity as waveform-family `T_\theta(x)` rather than an isolated feature intervention, and names the failed frozen `Spectra-AASIST/full-waveform/spoof` candidate. It intentionally excludes the unrelated H2B Q1 quality-only result.
- Validation: After every `main.tex` edit, `tectonic --outdir build main.tex` was run. The current PDF is readable, four pages, and has Table I on page 2 before the References; no overfull-box or unresolved-citation warning remains.
- Paper-review: The required launcher was rerun against this exact post-scope/layout PDF. It correctly spawned Sonnet reviewers at the 0/10/20-second stagger, but all three returned the same unauthenticated Claude CLI error. `paper/reviews/post_scope_layout_20260809/REVIEW_STATUS.md` preserves the failed attempt; no review verdict is inferred.

## 2026-08-09 - H4 score-free label--cue transportability protocol

- Decision: After H2 and H2B quality feasibility failures, broaden only into a classifier-free descriptive question. H4 reads the five existing `v1_28` feature tables and computes feature--label AUROCs, never detector scores or a new H2 candidate.
- Guardrail: The pre-analysis freeze hashes the exact feature inputs, allows only identity/label/view/feature fields, caps deterministic source IDs by class, and rejects response-like paths and columns. The full 420-cell matrix and 84-unit aggregation are terminal; no post-result ranking or downstream selection is permitted.

## 2026-08-09 - H4 score-free input freeze

- Validation: `h4_input_freeze_001` validated the exact five HDD feature Parquets against the H4 firewall, preserving their SHA-256 hashes, schemas, binary labels, and three-view completeness. It then froze 5,000 source IDs per label in every corpus (50,000 clusters total) using seed 2609.
- Guardrail: The compact repository note contains the manifest/provenance hashes, but the selected source IDs remain in the HDD Parquet. The freeze authorizes only the analyzer that rechecks those hashes and has no detector/model/audio interface.

## 2026-08-09 - H4 complete score-free label--cue atlas

- Validation: The frozen-manifest-only analyzer rechecked input and selection hashes, then completed all 420 registered AUROC cells with 500 source-cluster bootstrap replicates and no failed cells. Its aggregation reports 23 of 84 terminal descriptive stable units.
- Result: Full-waveform crest label separation is negative in the three ASVspoof corpora, positive in InTheWild, and near null in ASVspoof5. This independent score-free instability supports cautious corpus-dependent framing, but cannot be connected back to detector-score associations.
- Guardrail: The H4 result is sealed. It neither adds a candidate nor alters the failed H1/H2/H2B decision boundaries; its output files are HDD-resident and hash-recorded in the run note.
- Visualization: Applying the publication-plotting guidance, a tested validator reads only the sealed matrix/aggregation and exports a colorblind-aware three-panel signed-AUROC heatmap as vector PDF plus 300-DPI PNG. Green square markers denote only terminal descriptive units; a visual inspection passed and the figure stays supplementary.

## 2026-08-09 - H2B quality-first outer loop

- Decision: Treat the H2 quality-gate failure as an immutable result boundary. `H2B_QUALITY_FIRST_OUTER_LOOP_PROTOCOL.md` is committed before execution and prohibits reading model scores or detector implementations during transform selection.
- Design: Q1 is a finite, mild transform-family quality calibration on a score-blinded 256-clip manifest. It uses a lower Wilson retention bound, measured target-feature change, and WER as a lexicographic selection rule; Q2 then confirms the chosen arm on an independently frozen multi-corpus panel.
- Guardrail: A future H2B causal score study still needs an independently declared H1 candidate and four parity-validated runners. Current H1 atlas units and the failed H2 crest candidate are background, never H2B selections.

## 2026-08-09 - H2B Q0 DeepVoice input freeze

- Validation: Downloaded the 550.5 MB public DeepVoice dataset at its declared Arena revision to the HDD and confirmed labels SHA-256 against `data/arena-index.yaml`. It has 5,053 label rows (628 label-0, 4,425 label-1) and no score-like file in the downloaded tree.
- Implementation: The H2 input-CSV writer now records a byte hash for its emitted CSV. `src/h2b_score_blind_manifest.py` requires that byte-bound, label-only provenance, rechecks the declared dataset/revision, uses stable per-label SHA-256 selection, and writes non-overwritable Q0 artifacts. Focused tests (`13 passed`) cover successful freeze, substitution and provenance rejection, and output byte binding.
- Result: `h2b_q0_deepvoice_001` froze exactly 128 DeepVoice IDs per label (`256` total) under seed 2609. It does not decode audio or authorize Q1/Q4 scoring. Its artifact note contains all compact hashes.

## 2026-08-09 - H2B Q1 detector-free quality calibration

- Protocol/implementation: The Q1 arm manifest was committed before execution: fixed-length endpoint zeroing, mild signed spectral tilts, mild all-pass cascades, and two controls. The runner validates Q0/Q1 hashes, has no detector callback, and writes immutable pair/transcript checkpoints on the HDD. A subsequent lock fix prevents any future duplicate Q1 writer.
- Validation: `h2b_q1_deepvoice_001` completed all 256 × 9 = 2,304 pairs on physical GPU 3 (logical Whisper `cuda:0`). The final detector-free table and summary are hash-sealed; the selector independently verifies its complete, unique pair IDs.
- Result: No non-control family reaches the predeclared 0.90 95%-Wilson lower retention bound and target-change condition. The closest arm is -0.5 dB/oct tilt (239/256 retained; lower bound 0.89624); the selector's `selected_arms` is empty. Keep Q2--Q4 blocked and preserve this negative feasibility result rather than tuning it.
- Operational note: A second resumptive process was briefly started during status diagnosis and terminated immediately. The original writer completed the run; its summary records 2,304 newly written checkpoints and the final selector validates 2,304 unique pair IDs. The subsequently committed exclusive lock prevents recurrence. No detector path was involved.

## 2026-08-09 - Final local build and reproducibility verification

- Rebuilt `paper/main.tex` after clearing only ignored TeX intermediates. The refreshed tracked `paper/build/main.pdf` has four pages; extracted text places Table I on page 2 and References on page 3. The generated bibliography contains seven entries.
- Validation: `PYTHONPATH=. python3 -m pytest -q` passed all 79 tests in 6.45 seconds. The durable command/output, PDF SHA-256, and delivery constraints are recorded in `verification/FINAL_VERIFICATION_20260809.md`.

## 2026-08-09 - Static anonymous-PDF readiness preflight

- Implementation: `src/paper_pdf_checks.py` and `scripts/check_paper_pdf.py` make the current working-draft handoff independently checkable without relying on absent `pdfinfo`/`pdffonts` binaries. The test suite covers passing, identity/metadata/font failures, missing files, and invalid page targets.
- Validation: The refreshed PDF passes with four pages, Table I on page 2, References on page 3, 16 embedded font resources, empty `/Author` metadata, and no selected identity-string hit. The full repository suite is now 82 passed in 6.47 seconds.
- Boundary: This preflight is not a substitute for the official ICASSP 2027 archive, submission portal, or IEEE PDF eXpress; those remain external final-submission steps.

## 2026-08-09 - Official template acquisition audit

- Discovery: The ICASSP author page is still not exposing a conference-specific source archive. The linked generic IEEE 2024 conference-template ZIP was resolved from an official IEEE conference-template page.
- Access outcome: A direct download received `HTTP/2 202` with `x-amzn-waf-action: challenge`. No archive was persisted under the HDD and no local template was overwritten. Keep the pinned IEEEtran working copy intact until a fetched official package can be hash-compared and reconciled.

## 2026-08-09 - Cross-loop artifact index

- Implementation: `ARTIFACT_INDEX.md` centralizes the authoritative protocol, result, stop rule, HDD ledger owner, paper status, and external handoff condition for H1, H2, H2B, H3, and H4. It deliberately links to existing ledgers rather than duplicating hashes or results.
- Rationale: A future agent can now establish the experiment boundary and exact evidence path before starting work, reducing the risk of duplicated jobs, accidental score access, or post-result threshold changes.

## 2026-08-09 - US-letter geometry check

- Implementation: The static paper preflight now reads each PDF MediaBox and fails if any page differs from 612 × 792 points (US Letter), complementing the existing count, landmark, font, metadata, and identity checks.
- Validation: Synthetic failure coverage includes an A4-sized page; the current working PDF passes all four US-letter page checks. The boundary remains unchanged: only the official ICASSP template and submission checker can certify conference compliance.

## 2026-08-09 - ICASSP author-visibility correction

- Discovery: The official ICASSP 2027 editorial policy specifies single-anonymous review: reviewers know author names. The previous anonymous label was an internal-draft convention, not the conference's review rule.
- Correction: `main.tex` now tells readers that author information is required before submission. `paper/AUTHOR_BLOCK_REQUIRED.md` provides a no-inference replacement handoff. The static PDF checker keeps strong identity checks only in `anonymous-working-draft` mode and supports `single-anonymous-submission` mode after authorized author insertion.
- Boundary: Do not invent author identity, affiliation, email, or order. The current PDF remains readable evidence for the research draft but is deliberately not upload-ready until that information is supplied.

## 2026-08-09 - Current-version paper-review attempt

- Validation: Extracted 16,288 characters from the compiled post-author-policy PDF and launched the required three-reviewer Sonnet workflow on the 0/10/20-second stagger.
- Constraint: Alfa, bravo, and charlie each exited with `Not logged in · Please run /login` before producing any review. `paper/reviews/post_author_policy_20260809/REVIEW_STATUS.md` preserves this authentication failure; it is not a review result or meta-review.

## 2026-08-09 - Safe queued Telegram delivery

- Implementation: `src/telegram_delivery.py` and `scripts/send_pending_telegram.py` parse only level-two queued Markdown sections. Listing is network-free; delivery requires an explicit latest or exact-heading action plus both environment variables, and sends exactly one bounded plain-text message.
- Validation: Pure parser/API tests cover ordering, exact selection, empty/oversized rejection, non-success API responses, and a redacted receipt. The command lists all 25 current queued milestones without credentials or a network call.
- Boundary: The utility does not read/store/print a token or chat ID, and it must not be used to send all historical messages at once. `to_human/TELEGRAM_DELIVERY.md` is the operational handoff.

## 2026-08-09 - Offline Git delivery backup

- Execution: Created `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/git-bundles/icassp_antispoofing_20260809T235303Z.bundle` from the local research branch. `git bundle verify` reports complete history at `55e50e8427a7bb845bfc8ecda0d2194878d76bd6`; SHA-256 is `80ce7fc0954c83202b4c5a1ac493c19e3894d68b180201686238328e8dbe9777`.
- Boundary: The bundle is stored on the designated HDD, not Git, and is only a portable recovery artifact. It neither contacts GitHub nor satisfies the requested remote push. The restoration/push steps are in `to_human/GIT_BUNDLE_HANDOFF_20260809.md`.

## 2026-08-10 - Named-author ICASSP template and Codex review checkpoint

- Template: The user-provided ICASSP-2026 archive was copied to the HDD and its
  spconf.sty/IEEEbib.bst inputs were pinned in Git with the archive SHA-256 in
  `paper/template/ICASSP2026/TEMPLATE_PROVENANCE.md`. This is a user-directed
  working format, not a claim of ICASSP-2027 approval.
- Paper: `main.tex` now contains authorized dual affiliations, a literal
  all-caps spconf title, narrowed H1/H2 wording, and a clearpage before the
  bibliography. The resulting PDF has four technical pages and a
  references-only fifth page, avoiding a float split through the reference list.
- Verification: A clean build confirmed the generated bibliography uses the
  vendored IEEEbib style; the local checker was repaired to require the actual
  Table 1/I label rather than a substring false positive. Full tests pass 87.
- Review: Three fresh read-only Codex-agent reviews were preserved and their
  actionable findings resolved in `paper/reviews/codex_post_template_20260810/`.
  Do not relabel this internal work as external peer review.
