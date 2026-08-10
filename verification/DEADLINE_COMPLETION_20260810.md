# Deadline-completion record — 2026-08-10

**Handoff time:** 2026-08-10 20:00:26 UTC.

This record closes the user-authorized research window. It distinguishes the
completed, reproducible evidence from the stages that were correctly stopped
by their locked gates.

## Final verification evidence

| Check | Result |
| --- | --- |
| Full repository suite | `PYTHONPATH=. python3 -m pytest -q` — **149 passed** in 27.46 seconds at 19:35 UTC |
| Paper PDF | SHA-256 `4395f1e131e74e926c475a43c7a965a51ee46dcc73387dbf4e9166120b7fdd4c` |
| Paper local preflight | Passed: five US-letter pages, Table 1 page 3, references page 5, ten embedded fonts. This is not the official ICASSP/PDF-eXpress checker. |
| Remote state | At final verification, local and `origin/research/icassp-signal-audit` both resolved to `088d2932e4c5378040799a915f757e62f01284cd`. |
| Worktree exception | Only the user-provided `ICASSP2026_Paper_Templates.zip` remains intentionally untracked; no credential is tracked. |
| Offline recovery | Complete-history bundle at `8892b7931344b79351b8b1e22d006be92b574456`: `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/icassp_antispoofing_research_icassp_signal_audit_20260810T193538Z.bundle`; SHA-256 `94635fada2433a7208973bbc2318371c788984e6550562ba6d22a40e53b7caec`; verified with `git bundle verify`. |
| Telegram | Final selected handoff sent successfully as message ID 147; see the redacted receipt in `to_human/TELEGRAM_DELIVERY_LOG_20260810.md`. |

## Completed evidence

- H1 completed all 6,720 registered five-corpus within-class association cells.
  The discovery-frozen Spectra-AASIST/full-waveform/spoof crest candidate fails
  the fixed portability criterion; this is the paper's principal negative
  result.
- H2 completed its detector-free quality panel, and H2B completed its separate
  quality calibration. Their locked stops occur before any transformed detector
  score, causal effect estimate, or H3 training result.
- H4--H7 are sealed descriptive extensions only. In particular, the H7
  all-feature cross-corpus transfer matrix and inspected supplementary display
  are score-free and cannot identify a detector cue or mitigation.
- The ICASSP working draft, H1/H2 main evidence-boundary figure, internal
  Codex-only review bundles, future roadmap, and all resumability material are
  present in the repository.

## Non-completions retained as boundaries

No causal feature-reliance result, transformed-score paired analysis,
feature-conditioned training outcome, official ICASSP-2027 template/PDF check,
authorized public archive/DOI, or approved submission supplementary package
exists. These are explicit next-stage requirements, not omitted claims.

For a subsequent research loop, start from
`experiments/future_directions/POST_20260810_RESEARCH_PLAN.md`; do not tune or
reuse the stopped H1/H2/H2B selection/gate decisions.
