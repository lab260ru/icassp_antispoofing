# Review by alfa

```
SEED: 9449cb2a263a40516c00430a1c0dae16
```

```
AXIS: Missing ablations (which component is doing the work?)
STANCE: Maximally adversarial (assume the result is artefactual until proven otherwise)
```

---

### Summary

The paper makes three claims: (1) TTS repetition-counting failures track periodicity of the input, not its length, established via a length-matched control ladder across six checkpoints/three architectures; (2) the standard judge (Whisper) is itself biased toward manufacturing this effect, so a CTC recognizer is used instead; (3) a Lean-4-verified "contraction map" account of why counting should fail is measured directly and refuted — the boundary-to-boundary Jacobian is expansive (q > 1) everywhere they check. The periodicity-vs-length design is the paper's real asset and is executed carefully, with an unusual amount of self-directed adversarial testing (specification curves, judge-artifact checks, exclusion-rule sweeps) that most TTS papers don't bother with. My problem is with what all that rigor is spent on: the paper builds an elaborate, formally verified refutation of a mechanism (linear contraction of a boundary map) that was never a strong candidate to begin with, while never testing against the mechanism that the interpretability literature actually points to for repetition failures — induction heads / repetition neurons — despite already possessing the attention and probing infrastructure to do so. The paper ends Section 5 declaring the causal question open; it was open in part because the authors didn't run the check that was available to them.

### Major concerns

1. **Issue** — The paper doesn't rule out codec/token-frequency artifacts as an alternative to "periodicity" as the causal variable. `[Evidence — alternative explanations]`
   **Where** — Section 5, "What the design does not separate."
   **Why it matters** — The paper controls for *text* likelihood under an independent LM (§5, "the control is the less probable text in 87% of pairs") but never checks *codec-token* n-gram frequency in the X-codec2 / vocoder training data. If periodic sequences are simply rarer at the acoustic-token level (independent of any abstract "periodicity" property), the periodicity-vs-length dissociation would still hold, but the causal story ("the map tracks periodicity") would be wrong — it would track token-sequence surprisal at the codec level, a purely statistical confound the authors' own framework (frequency vs. periodicity) was built to rule out but didn't fully.
   **What would address it** — Report codec-token n-gram frequency (or perplexity under a codec-only LM) for the repeated vs. control arms, the same way text-LM likelihood was checked.

2. **Issue** — The headline Whisper miscalibration ratio (0.19) quoted in the abstract rests on n=12. `[Statistical rigour]`
   **Where** — Abstract; §2 "Behavioural measurement" (`n=12 over 6 donors`).
   **Why it matters** — The paper is admirably careful about sample sizes and CIs everywhere else (e.g., bootstrapped intervals for the panel gap), which makes it conspicuous that the one number promoted to the abstract as a load-bearing methodological justification (why Whisper is disqualified as a judge) is stated to two significant figures off a 12-trial audit with no interval given. This is the number that licenses switching the entire study to a CTC judge — it deserves the same rigor as the panel-level claims.
   **What would address it** — Report a CI on the 0.19 figure, or bump the audit sample; if it's small by necessity (concatenative audio construction is labor-intensive), say why n couldn't be larger and hedge the abstract's precision accordingly.

3. **Issue** — The "duration budget" mechanistic explanation for why VITS is immune and F5-TTS is not generalizes from one model per class. `[Evidence — baselines / overgeneralization]`
   **Where** — §3.2, VITS/F5-TTS paragraph ("What plausibly separates the two is where the count must live…").
   **Why it matters** — This is an n=2 comparison (one per-token-duration NAR model, one global-duration NAR model) elevated to an architectural claim ("Absence of an autoregressive decoder confers no immunity"). Per-token vs. global duration is confounded with everything else that differs between VITS (flow + adversarial training) and F5-TTS (flow-matching, DiT backbone) — training objective, model scale, tokenizer, alignment mechanism. The duration-supply intervention on F5-TTS (§3.2, "supplying F5-TTS the duration a correct rendering takes") is a nice partial ablation, but it only manipulates one model, so it establishes that duration information helps F5-TTS, not that duration-budget-vs-per-token-duration is *the* variable separating the two architecture classes.
   **What would address it** — Add at least one more per-token-duration NAR system (e.g., a Matcha-TTS or StyleTTS2-class model) and one more global-duration one before asserting a class-level mechanism.

4. **Issue** — The Lean-4-verified contraction bound is an elaborate, formally-verified refutation of a mechanism that was a weak candidate from the start, and the paper concedes it contributes nothing to the actual results. `[Novelty — overclaimed rigor]`
   **Where** — Section 4, and the paper's own admission: "we keep the bound because it motivated the manipulation… but it explains nothing in Section 3."
   **Why it matters** — Linear contraction of a boundary-to-boundary state map under teacher forcing is an unusually strong, almost prior-implausible condition (it requires near-uniform attention weights, δ ≤ small, across every repeated span at every layer); nobody studying real transformer Jacobians should have expected q<1 to hold. Machine-checking the *implication* in Lean 4 with mathlib, an axiom audit, and three numerical validation gates signals a level of rigor that the actual epistemic content doesn't need — a back-of-envelope Lipschitz argument would have made the same point in a paragraph. This matters because "formally verified" is being used as a credibility marker for a claim (the bound) that turns out to be empirically vacuous; a reader skimming the abstract could easily read "Lean-4-verified bound" as evidence *for* the paper's mechanism rather than evidence *against a mechanism the authors themselves discarded*.
   **What would address it** — Reframe Section 4 explicitly as a negative-results appendix rather than a headline contribution (it currently gets equal billing with the periodicity result in the abstract and title-adjacent framing); the formal verification apparatus could move to supplementary material with the plain-language argument doing the work in the main text.

5. **Issue** — The paper never tests the mechanism the field's interpretability literature actually proposes for repetition-driven failures — induction-head toxicity — despite already having head-averaged attention traces and count probes in hand. `[Missing ablations]`
   **Where** — Section 4 (mechanism testing) and Section 5 ("The rival we could not decide").
   **Why it matters** — There is a directly relevant, mechanistically specific account already in the literature: Wang et al., "Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models" [2505.13514], and a companion line of work on repetition neurons [2507.07810]. Both propose that repetition failures arise from *specific attention circuits* (induction heads) whose behavior degrades under sustained repetition — a testable, falsifiable, and much better-motivated alternative to a generic global-contraction bound, and one that fits naturally with the periodicity-not-length framing the authors already established. The paper computes head-averaged attention and effective rank (§2, "State measurement") and even reports a global attention-flattening statistic (δ ≤ 1.23 nats, dose-response r=0.59, §4) — but it never decomposes this by head to ask whether specific heads (rather than the aggregate attention profile) carry the effect, nor does it cite or engage with this literature at all. Section 5 ends by saying the representation-vs-readout question "we could not build [a differentiating manipulation]" — but head-level ablation/patching, guided by the induction-head-toxicity account, is exactly the kind of manipulation that would differentiate it, and it is cheaper than the Lean-verified contraction analysis they did run. Rejecting a weak, self-generated hypothesis (contraction) while not testing the strong, externally-motivated one (induction-head toxicity) leaves the paper's central causal claim — "the count fails somewhere, we don't know where" — considerably less settled than the density of statistics in Sections 3–4 implies.
   **What would address it** — Run head-level ablation or attention-pattern classification (which heads attend backward to the previous occurrence of the repeated token/carrier) on the existing checkpoints, and correlate induction-head-specific metrics (not aggregate attention entropy) with the periodicity gap; cite and position the paper relative to [2505.13514] and [2507.07810].

### Minor concerns

- Figure 1 shows only 3 of the 6 checkpoints (Llasa-1B, XTTS-v2, Qwen-0.6B); the text never explains the selection, and it happens to include two of the panel's more dramatic gaps (per Table 1, Llasa-1B and XTTS-v2 both show sizeable rep/ctl splits) without showing, e.g., Qwen3-TTS-1.7B, whose median error is *zero* — arguably the most interesting curve to plot, and it's left out.
- "exact-match scores 30-of-32 and 3-of-32 alike" (§2) is opaque without more context on what property is being claimed equal.
- The self-referential "we withdraw the architectural claim an earlier draft made" (§3.2) is honest but unusual for a camera-ready-style ICASSP submission; if intentional it should be flagged as a revision note rather than embedded mid-paragraph.
- Every load-bearing detail (S1–S28) lives in an unincluded supplement; the main text alone is not self-contained for verification (e.g., the "not for want of competence" claim about VITS tuning in S13 can't be checked here).
- "period" and "verbatim identity" are used somewhat interchangeably in places (§3.1) before being formally separated later in the same section — a reader encountering the periodicity ladder for the first time may need to re-read to see the distinction is load-bearing, not just phrasing variation.
- Units/axes in Figure 1 caption don't state whether "count error" is signed or absolute in panel (a) versus the text's signed median figures — should be explicit.

### Belief update

I did update, modestly, on the proposition "counting failures in autoregressive TTS are driven by input periodicity rather than sequence length per se, and are not artifacts of the standard (Whisper-based) evaluation pipeline." The length-matched control design is genuinely good and the judge-artifact analysis (§3.2, the "which arm moves" argument) is a real distinguishing test, not just a consistency check. I did not update on "we now understand the mechanism" — Section 4 only rules out one weak candidate, and Section 5 correctly reports the field is left with an open question, but that openness is partly self-inflicted: the natural next mechanistic test (induction-head-level analysis, motivated by existing literature) wasn't run despite the tooling being in hand.

### Verdict

**Major revision** — the periodicity-vs-length result is solid enough to survive review, but the paper needs to either run the induction-head-level ablation that the existing interpretability literature (missed entirely here) motivates, or explicitly scope Section 5's "rival we could not decide" claim as narrower than currently framed, before the mechanistic story can be trusted as anything more than "we refuted the hypothesis we ourselves invented."
