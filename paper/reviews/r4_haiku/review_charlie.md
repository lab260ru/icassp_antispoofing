---

# Summary

This paper proposes that autoregressive TTS repetition failures stem from a fundamental structural limit: when a decoder's per-repetition state map contracts with factor q < 1, no Lipschitz readout can distinguish more than N* = log(µ/(2LC)) / log q repetitions. The theorem is formally verified in Lean 4. Empirically, six TTS checkpoints systematically undercount repetitions by median –8.3% at k≥6, but length-matched controls with distinct words remain exact (0.0%), isolating periodicity as the relevant variable. The effective rank of decoder trajectories saturates under repetition at 50% the control growth rate. The work is rigorously executed with excellent reproducibility artifacts (code, Lean proofs, both transcript judge outputs), but the evidence for the claimed mechanism rests on indirect proxy measures and the paper does not isolate which component (contraction strength, readout margin, or architectural feature) drives failure in any given model—a striking gap given the precision of the theorem.

---

# Major Concerns

1. **The theorem is proven but its premises are not** [Evidence — distinguishing test].
   **Where:** Section 5 ("What is proved, and what would refute it").
   **Why it matters:** Theorem 1 is a conditional result: "if F contracts with q < 1, then N* ≤ log(µ/(2LC))/log q." The paper proves this implication rigorously in Lean (excellent), but then claims to validate it empirically by measuring the *consequence* (state saturation, capacity loss), not the *antecedent* (contractivity q). The authors acknowledge this explicitly: "What we establish empirically is the theorem's consequence, not its antecedent." But this is a critical gap: the paper cannot rule out alternative mechanisms that also produce representational collapse at repetition boundaries without any contracting map. For instance, if the stop token head is simply undertrained on long sequences or trained with a distribution shift (seeing mostly short contexts), it would fail at high k regardless of q. The paper measures µ (readout margin) only implicitly via the observed counting horizon, not directly.
   **What would address it:** Direct measurement of q from boundary state distances, or an ablation showing that the specific contraction dynamics (not just any form of representational loss) is necessary. Alternatively, interventions that change q while controlling µ, or vice versa.

2. **Why does Qwen3-TTS-1.7B not exhibit the deficit at all?** [Evidence — alternative explanations].
   **Where:** Section 4.1 and Section 5.
   **Why it matters:** One model (Qwen-1.7B) shows 0.0% error at k≥6, while every other model (5/6) shows –8.3% to –16.7%. The paper reports this honestly as a regime where the theorem does not apply, but does not investigate why. Is it because Qwen-1.7B's per-repetition map does not contract? Because its readout margin µ is enormous? Because of architectural differences (the paper notes Qwen sums text/acoustic embeddings differently, preventing attention column analysis)? This is not a minor outlier—it is the largest model in the panel, and it breaks the pattern. Without understanding why, the claim that "periodicity-driven counting collapse" is a structural limit of "this class of AR decoders" (codec-language models with softmax attention) is overstated. The class seems to split into two regimes, and the paper does not explain the boundary.
   **What would address it:** Ablation studies on Qwen-1.7B (layer-wise interventions, attention masking, readout margin probing) to isolate whether it's contraction-free or just high-margin. Analysis of architectural differences that might explain why.

3. **The effective-rank measurement does not directly validate the theorem's object** [Evidence — measurement validity].
   **Where:** Section 4.3, discussion of effective rank as a proxy for state distinguishability.
   **Why it matters:** Theorem 1(ii) claims that states s_m converge geometrically: d(s_m, s_*) ≤ Cq^m. This would imply that the set {s_1, s_2, ..., s_k} becomes increasingly close as k grows, reducing the dimension of the "reachable manifold." The paper measures this via effective rank N_eff = exp(−Σ p_i log p_i) of the *generation trajectory* {h_t : t ∈ [1, T]}, where h_t is the decoder hidden state at time t. This is a temporal trajectory (all states visited during generation), not the orbit {s_m} (states at repetition boundaries). A monotonous output (correct repetition of the same word) will naturally produce a low-rank temporal trajectory because most tokens are identical, even if the boundary states {s_m} have not converged. The paper attempts to control for this by conditioning on output diversity and restricting to correctly-rendered items, which weakens but does not eliminate the circularity. A model generating "very very very very" has fewer acoustic degrees of freedom than one generating "cat dog bird fish"; a low temporal rank may reflect output sparsity, not state collapse.
   **What would address it:** Direct measurement of the orbit diameter d(s_m, s_n) for m, n ≥ N (boundary-state distances), or probe-based reconstruction of the count signal at early vs. late boundary states (already done in Section 4.4 as a negative result, but that attempt should be reported as an ablation against this concern).

4. **Missing ablation: which component of the bound sets N* — is it q, C, or µ?** [Evidence — missing ablations].
   **Where:** Entire empirical section; no direct measurement of any of q, C, or µ.
   **Why it matters:** The theorem states N* = log(µ/(2LC))/log q. The paper measures the consequence (that models undercount), but does not ablate which of the three factors (contraction q, orbit scale C, or readout margin µ) is actually driving the observed failure rate. This is essential for understanding the mechanism and for targeted mitigation. For example:
   - If q is close to 1 (weak contraction), then N* is large and the problem is mild; the paper should show q.
   - If C is huge (the orbit spans most of the state space), then models have little representation room; the paper should characterize C.
   - If µ is tiny (the stop head has only a narrow margin to separate counts), then the problem is really about the stop head, not the decoder.
   The paper measures N* implicitly (observed count error maps to an implied horizon) but never reports q, C, or µ directly. The three-factor bound hides which lever matters most.
   **What would address it:** Reporting or estimating q, C, and µ from the model's learned weights or probed trajectories. Even order-of-magnitude estimates would clarify the mechanism. Alternatively, controlled interventions (e.g., amplifying the readout margin via logit scaling) to test whether N* changes predictably.

5. **The repetition penalty ablation result contradicts the paper's theory, but is dismissed too quickly** [Narrative — theory-data conflict].
   **Where:** Section 5, paragraph beginning "Existing mitigations act late."
   **Why it matters:** The paper tests XTTS-v2's repetition penalty at α ∈ {2, 3, 5, 8, 1 (disabled)} and finds the deficit moves only from –12.5% to –16.1% across a fourfold penalty change. The paper concludes: "this lever does not act on the quantity that sets the horizon." But this is strange. If the horizon is set by q, C, and µ, then:
   - The penalty is a *decoding-time* intervention (it downweights repetitions in the next-token distribution).
   - Under the paper's theory, this cannot change q (the learned contraction), cannot change C (the orbit scale), and should not change µ (the readout margin is set before generation).
   - So the penalty *should* fail to remove the deficit if the bound is tight.
   However, this means the paper's theory predicts nothing falsifiable about decoding-time interventions—any attempt at a late-stage fix was doomed from the start. That's not science; it's unfalsifiability. The stronger claim would be: "the horizon is set at training time, so only training-time interventions (e.g., contraction regularization, readout margin training) can help." But the paper doesn't attempt or propose such interventions.
   **What would address it:** Either (a) propose a training-time mitigation motivated by the theory, test it, and show it works better than the decoding-time penalty, or (b) soften the conclusion to "this penalty does not affect the measured horizon, consistent with a pre-learned contraction," which is honest but less satisfying.

6. **No pre-registration of the effective-rank hypothesis or threshold choices** [Evidence — post-hoc hypothesis].
   **Where:** Section 4.3; no mention of when/how the effective-rank measurement was decided.
   **Why it matters:** The paper measures effective rank as a test of Theorem 1(ii). But was this measurement planned before seeing the data, or was it chosen post-hoc after the count-error and periodicity results already confirmed the effect? The paper notes: "Section 4.4 explains why we do not measure q by differencing boundary states" — so they tried one measurement (q), it failed, and they switched to a proxy (N_eff). This is honest reporting, but it raises the question: what other measurements were tried and failed? The capacity-gain result (0.50 ratio with disjoint intervals) is very clean, but without pre-registration we cannot know if it's a real effect or a lucky cherry-pick from multiple attempted measurements.
   **What would address it:** A pre-registration statement, or at minimum a clear listing of all attempted measurements and their outcomes (what was tried but failed, and why).

---

# Minor Concerns

- **Qwen3-TTS attention structure prevents Lemma 1 validation**: The paper notes Qwen sums embeddings per position, so "no attention column belongs to the text and the lemma cannot be checked there." But then the theorem's premise (Assumption 1, periodic conditioning through attention dilution) is not validated for the model that doesn't fail. Is Qwen exempt from the theorem for a different reason?

- **Median of exactly 0.0% for control error**: Reported as [0.0, 0.0] with n=639 generations. If every control generation was exactly correct, that's a striking result. But CTC recognizers can make errors. Is the report "0.0 to three significant figures" or truly zero? A mean+std alongside the median would clarify.

- **Notation D(s_m, s_n) vs d(·) inconsistency**: The theorem uses d(·) for metric distance; fine. But the text mixes "profile entropy," "attention weight," and "distance" without always being clear on units. Is δ in nats? Is q dimensionless? The Lean proof should disambiguate, but the paper-text exposition could be clearer.

- **Figure 1 x-axis**: Points are at k ∈ {1,2,3,4,6,8,12,16,24,32} but connected with lines, slightly overstating smoothness and interpolation. Consider plotting as discrete points with no connecting line, or add confidence bands instead.

- **"Three seeds" without stating values**: Section 3 says "Each item is generated with three seeds," but does not list them. Are they deterministic (seed=1,2,3) or random? Exact replication requires this.

- **Whisper bias audit results**: The paper shows Whisper scores 0.27–0.65 ratio on repetitions but 1.00 on controls. This is a strong finding on Whisper's own bias, and it's important because the TTS field may have been using Whisper as a judge. But the paper mentions this as a reason to use CTC instead, and does not dig into *why* Whisper is biased. Is it the language model prior? The decoder? This would strengthen the narrative.

---

# Belief Update

**Belief 1: "Repetition failures in AR TTS are a training-time exposure-bias artifact."**
- **Before**: 60% confidence this is the primary cause.
- **After**: 40% confidence. The paper's evidence for periodicity-vs.-length (Figure 1b) is genuinely strong and rules out simple length-based explanations. I now believe there is a real effect beyond training bias. However, the failure to measure the theorem's premises (q, C, µ) prevents me from full confidence in the *specific mechanism* (fixed-point contraction). The theory is plausible, but the evidence is indirect.

**Belief 2: "Representational collapse alone predicts repetition failures."**
- **Before**: 50% credence in this as sufficient.
- **After**: 60% credence. The paper's control (distinct words, same length) showing zero errors argues against length-based collapse and for periodicity-specific dynamics. But the Qwen-1.7B outlier (no deficit, still some capacity loss) suggests representational collapse is necessary but not sufficient, which the paper acknowledges honestly.

**Belief 3: "The limit is fundamentally structural and unchangeable via decoding rules."**
- **Before**: 40% credence (seemed possible but unproven).
- **After**: 55% credence. The theorem + control experiment make this more plausible. The repetition penalty ablation (no benefit across 4x change) supports it. But Qwen-1.7B's exception (no deficit despite similar capacity loss) shows the mechanism can be escaped, so the limit is not universal—it's conditional on the model's learned q. Whether q is learnable or fixed by architecture remains unclear.

---

# Verdict

**Major revision** — required because the paper proves a conditional theorem but does not empirically validate its premises (contraction factor q, orbit scale C, readout margin µ), only its consequence (state saturation). The control experiment (Figure 1b) is genuinely strong evidence for periodicity-driven failure, and the formal verification is exemplary, but without isolating which of q/C/µ drives the observed horizon in real models, the mechanism remains underspecified, and the paper cannot rule out simpler post-hoc explanations (undertrained stop head, stop-head margin collapse, distribution shift). The Qwen-1.7B counterexample must be analyzed to clarify the boundary of applicability. These gaps do not invalidate the core contribution but prevent it from being a confident advance on mechanism.
