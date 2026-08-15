# Peer Review — "Repetition, Not Length: Isolating the Counting Failure in Neural Text-to-Speech"

**Reviewer codename: charlie**

SEED: 7b3fa62c19e84d0f6a2c8e91b4d573af

LENS: Reproducibility (data/code availability, reporting completeness, hyperparameter disclosure)
STANCE: Skeptical-but-fair

## Summary

The paper asks whether autoregressive TTS models' well-known tendency to mis-render repeated phrases is caused by repetition per se, or is confounded with input length. Using a "repetition ladder" of 180 stimuli (a target word repeated k times, k ∈ {1,...,32}) paired with length-matched, non-repeating controls, the authors measure rendered-count error across six checkpoints spanning three architectures (Llasa 1B/3B/8B, Qwen3-TTS 0.6B/1.7B, XTTS-v2), plus non-autoregressive baselines (VITS, F5-TTS) and a held-out fourth architecture (CosyVoice 2). They report a large, robust repeated-vs-control gap (46.8 points overall, 75.4 at k≥6) that survives penalty sweeps, greedy decoding, exclusion-rule ablation, and re-scoring with four different speech recognisers, arguing this rules out the length account and judge-side artefacts. A second "period ladder" shows the deficit is graded with periodicity rather than a step function at verbatim repetition. Finally, the authors formalise a candidate mechanism (a Lean-4-verified contraction bound on per-repetition hidden-state dynamics) and empirically measure the premise, finding it false in every decoder tested (q ≫ 1), and conclude the mechanism they set out to test does not explain the phenomenon.

## Major concerns

1. **Issue** — The code, stimuli, Lean development, and both transcript sets are stated to become public only "on acceptance," meaning none of the paper's ~2,000 generations, the 420-specification robustness curve, or the formal proof can be independently verified at review time.
   **Where** — "Code, data, settings" paragraph, Section 2.
   **Why it matters** — Nearly every headline number in the paper (46.8, 75.4, 76.4-point mixed-effects estimate, q = 21.0–347.7, the Lean proof itself) is asserted rather than checkable. For a paper whose central rhetorical move is "we checked every alternative explanation and it survives," withholding the artifacts that would let a reviewer or reader check the checking is a serious gap, not a formality — especially since several of the paper's own escape hatches (e.g., "dated in S16" as evidence a test was pre-committed) are unverifiable claims about supplementary material we cannot read.
   **What would address it** — Provide an anonymized or embargoed-but-reviewer-accessible repository (a private GitHub link, an anonymous OSF/Zenodo mirror, or supplementary zip) rather than a hard acceptance gate. Conference reviewing norms generally expect artifacts available to reviewers, not just readers after the decision is made.

2. **Issue** — Sampling hyperparameters ("each system's shipped defaults") and the exact algorithm for extracting the rendered count ĉ from a transcript are described only in prose, deferring the actual values/pseudocode to an inaccessible supplement (S19, matching-rule description in Section 2).
   **Where** — Section 2, "Behavioural measurement" and "Code, data, settings" paragraphs.
   **Why it matters** — The count-matching rule ("skipping unrecognised ones rather than stopping at the first miss") is exactly the kind of scoring-logic detail where small implementation choices can materially shift exact-match rates, yet it's given only as prose intent, not as a specification or code excerpt. Likewise "shipped defaults" for temperature/top-p/repetition-penalty per system are asserted to matter (used to argue Qwen3-TTS's null result isn't due to conservative decoding) without stating what those defaults are.
   **What would address it** — Include the matching algorithm as pseudocode or a short code listing in the main text or a reviewer-visible appendix, and tabulate the actual decoding hyperparameters per system rather than referencing them by supplement number alone.

3. **Issue** — The k≥6 threshold used for the paper's second headline figure (75.4 points) is explicitly stated to have been chosen after inspecting the data, and the paper's mitigation ("we quote both") does not extend to most of the downstream analyses (Table 1, the mixed-effects model, the CosyVoice-2 extrapolation test), which are all conducted only at k≥6.
   **Where** — Section 3.1 ("The k≥6 threshold was chosen after seeing where the arms diverge...") and Table 1.
   **Why it matters** — This is a direct admission of a post-hoc analytic choice being used as the paper's featured effect size in the abstract and in most tables. The abstract does report the unconditioned 46.8, but the CosyVoice-2 "confirmatory" test — arguably the paper's strongest generalization claim, since it's framed as a held-out, pre-registered-style prediction — is evaluated against the k≥6-derived posterior (65.3 [5.9, 93.3]), which inherits the post-hoc threshold's bias.
   **What would address it** — Re-run the CosyVoice-2 held-out test (and ideally the mixed-effects model) at the unconditioned/whole-ladder definition, or explicitly justify why threshold selection doesn't compromise this specific downstream test given it was fixed before CosyVoice-2 was scored.

4. **Issue** — The filler-word control is not shown to be matched on phonological/durational/frequency properties to the repeated target word, and at k>8 the control itself becomes periodic (period-8) via filler-cycling, which the paper only partially addresses via a supplementary "never-cycled" re-analysis.
   **Where** — Section 2 ("Stimuli"), and Section 3.1 ("Above k=8 the cycled control is itself period-8...").
   **Why it matters** — If the eight fillers per template are more common/easier-to-articulate words than the target, the repeated-vs-control gap could be partly a difficulty confound rather than a pure repetition effect. The paper's own admission that "most of that 7.1 points is vocabulary, not periodicity" when comparing cycled vs. never-cycled controls concedes a non-trivial vocabulary confound exists, but the magnitude of that confound at k≤8 (where cycling isn't yet in play) is never separately estimated.
   **What would address it** — Report filler-vs-target matching statistics (frequency band, syllable/phoneme count, duration in a reference TTS or forced-alignment) and, ideally, a control condition using a single repeated-but-different-position filler to isolate vocabulary effects from periodicity effects directly rather than inferring it from a cycled/never-cycled contrast.

5. **Issue** — The paper generalizes across "three architecture families" using a mixed-effects model with architecture as a random effect, but explicitly acknowledges that three groups "barely identify a group-level scale" and that under a wide prior the extrapolated between-architecture gap "touches zero."
   **Where** — Section 3.1, paragraph beginning "Our six checkpoints are only three architecture families..."
   **Why it matters** — The paper's title and abstract-level framing ("across six checkpoints from three architectures") implies a level of architectural generality that a 3-cluster random-effects model cannot properly support; the authors' own sensitivity analysis shows the generalization claim is fragile to prior choice. The subsequent CosyVoice-2 "confirmation" (n=90 items) is used to paper over this, but one additional data point does not resolve a fundamentally underpowered clustering structure.
   **What would address it** — Temper the framing to reflect that the current design of the paper supports 3 architecture-level replicates, not a validated general law across "architectures," and/or add more architecturally distinct families (beyond CosyVoice-2) before making a between-architecture generalization claim.

6. **Issue** — The non-autoregressive baseline comparison (VITS "immune" vs. F5-TTS "not immune") rests on one model per condition, yet is used to motivate a specific causal explanation (duration modeling strategy) that is then "tested" via a single intervention (supplying F5-TTS the correct total duration) on one model.
   **Where** — Section 3.2, paragraph beginning "VITS shows no dissociation..." through "What plausibly separates the two..."
   **Why it matters** — The authors' own hedge ("suggestive rather than isolating," Abstract) is honest, but the discussion text still asserts a specific mechanistic explanation ("where the count must live") from an n=1-per-condition comparison, and the duration-supplying experiment, while a nice manipulation, only shows F5-TTS improves — it does not establish that duration-modeling-strategy is the operative variable versus some other VITS/F5-TTS difference (training data, loss, model scale: VITS run size isn't given).
   **What would address it** — Either soften the causal claim to match the n=1-per-condition evidence, or add a second duration-per-token vs. duration-as-total pair within a matched architecture (e.g., a duration-predictor ablation within one codebase) to isolate the proposed mechanism properly.

7. **Issue** — The decision to discard Whisper as the primary judge rests on a very small validation audit (n=12 trials over 6 donors for Whisper, n=24 for the CTC model), yet this audit result (0.19 vs. 1.00) is used as the load-bearing justification for the entire measurement pipeline.
   **Where** — Section 2, "Behavioural measurement, and why the obvious judge is wrong."
   **Why it matters** — This is a reasonable and well-motivated design choice, but n=12/n=24 against concatenative (not model-generated) audio is a thin evidentiary base for a methodological pivot this consequential; no confidence interval or variance estimate accompanies the 0.19/1.00 figures themselves (elsewhere the paper is careful to report intervals).
   **What would address it** — Expand the audit sample or report interval estimates around the 0.19 and 1.00 ratios so readers can judge how tightly the Whisper-disqualification result was pinned down.

## Minor concerns

- The prose style (elliptical, clause-dense, numbers embedded mid-sentence without consistent units restated) makes several key results difficult to parse on first read — e.g., the "0.547 on the strictest of five rules and 0.69–0.86 on the others" sentence in Section 3.1 would benefit from a table.
- No funding source, conflict-of-interest, or ethics statement is included anywhere in the manuscript.
- Figure 1 caption says panel (b) ran "on the four checkpoints the ladder ran on" without stating in the caption or main text why the other two checkpoints were excluded from the period ladder specifically (main text only implies resource/scope reasons).
- The claim "F5-TTS... so we withdraw the architectural claim an earlier draft made" (Section 3.2) references an "earlier draft" not otherwise described — this internal revision history is not reviewer-relevant and should be edited out or reframed as a straightforward negative result.
- Table 1 lists Par. (parameter count) for XTTS-v2 as 0.4B without a citation/source for that figure, unlike the Llasa/Qwen figures which are self-evident from the checkpoint names.
- The abstract's number density (nearly every clause carries a distinct statistic) makes it hard to extract the single headline claim; consider foregrounding one or two numbers and moving the rest to the body.
- Reference [15] (Venkatesh, arXiv:2605.09239, 2026) and reference [8] (companion paper, arXiv:2604.08591, 2026) both carry 2026 dates that are in the future relative to typical review timelines for an ICASSP-style submission — worth double-checking these arXiv IDs/dates are correct at camera-ready.

## Verdict

**Major revision** — the core manipulation (repetition vs. length-matched control) is a genuinely useful design and the robustness checks are extensive in spirit, but the paper's claims currently outrun what a reviewer can verify: code/data/Lean proof are withheld until acceptance, several headline statistics rest on post-hoc thresholds or small validation samples, and the architecture-generalization and non-AR-baseline causal claims are built on clustering too sparse (n=3, n=1-per-condition) to bear the weight the discussion places on them.
