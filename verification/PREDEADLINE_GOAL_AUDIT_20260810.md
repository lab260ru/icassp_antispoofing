# Pre-deadline goal audit — 2026-08-10

**Audit time:** 2026-08-10 15:30 UTC.
**Scope:** the active autonomous research objective through 20:00 UTC. This is
a live audit, not a claim that the time-bound objective has ended.

## Requirement-by-requirement evidence

| Objective requirement | Current authoritative evidence | Status at audit time |
| --- | --- | --- |
| Preserve research, notes, protocols, and results in the repository/HDD | `AGENTS.md`, `ARTIFACT_INDEX.md`, `research-state.yaml`, `research-log.md`, every experiment protocol/result note, plus hash-bound HDD path manifests | Satisfied for completed loops. |
| Test signal features across several models and datasets | H1 has the complete 5-corpus × 8-model × 2-class × 3-view × 28-feature matrix (6,720 cells); see the five-corpus aggregation record | Satisfied. |
| Establish whether the crest observation is portable | Discovery-frozen Spectra-AASIST/full-waveform/spoof crest fails the fixed five-corpus rule; Figure 1 and the sealed aggregate record exact estimates | Satisfied with a negative portability result. |
| Test causal waveform sensitivity carefully | H2 completed detector-free quality screening, but three arms fail the frozen 90% retained-pair gate; no detector received a transformed waveform | Satisfied as a quality-gated stop; no causal effect is claimed. |
| Consider feature-conditioned training with efficient BF16/GPU settings | H3 is explicitly not authorized because valid H2 causal sensitivity is absent. Parity/throughput artifacts exist, but no unsupported training run was started | Intentionally not run; this is required by the locked evidence boundary. |
| Strengthen the paper with review and additional evidence | Codex-only internal review bundles, H4/H5/H6 descriptive extensions, H1 covariate disclosure, canonical ASVspoof5 citation, and the evidence-boundary Figure 1 are preserved | Satisfied for an initial evidence-conservative draft. |
| Compile after each `main.tex` edit | The tracked current PDF hash and local five-page static preflight are in `verification/FINAL_VERIFICATION_20260810.md` | Satisfied for every recorded main-source revision. |
| Explore requirements and use a template | `paper/submission-requirements.md` records the official ICASSP 2027 policy/deadline; the supplied ICASSP-2026 template inputs are hash-pinned under `paper/template/ICASSP2026/` | Working draft satisfied; official 2027 kit remains an external final-submission prerequisite. |
| Push checkpoints and communicate material milestones | `origin/research/icassp-signal-audit` receives normal non-force pushes; redacted Telegram receipts are in `to_human/TELEGRAM_DELIVERY_LOG_20260810.md` | Satisfied through message 145 at audit time. |
| Make work resumable | `AGENTS.md` names the current sealed gates, latest remote/delivery state, artifact boundaries, and safe continuation rules | Satisfied. |

## Explicit non-completions that must not be misrepresented

- No transformed detector score, causal cue-reliance result, mitigation EER, or
  feature-conditioned training result exists. These are not missing runs: H2
  and H2B quality gates stopped before they were authorized.
- No official ICASSP-2027 template/PDF checker outcome, permanent public
  archive, DOI, or authorized supplementary submission package exists. The
  repository now contains a release-candidate checklist, not a release.
- The only intentional untracked repository-root artifact is the
  user-provided `ICASSP2026_Paper_Templates.zip`; no `.env` or large HDD asset
  is tracked.

## Final handoff checks to perform at or after 20:00 UTC

1. Verify remote head and `git status --short --branch` (only the template ZIP
   may be untracked).
2. Re-run the full repository test suite and, if `main.tex` changed since this
   audit, compile and run the local PDF preflight again.
3. Reconcile the exact PDF SHA-256 and test count in
   `verification/FINAL_VERIFICATION_20260810.md` and
   `to_human/FINAL_HANDOFF_20260810.md`.
4. Record a final redacted Telegram delivery only for a material new milestone;
   never bulk-send historical queue entries.
5. Mark the time-bound goal complete only if all completed-loop evidence,
   remote delivery, and final handoff artifacts still validate.
