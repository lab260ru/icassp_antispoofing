I'll proceed with the review from the paper text alone. Let me now write the full review:

---

## Summary

This paper proposes a formal dynamical-systems account of counting failures in autoregressive text-to-speech (TTS) models. The authors model repetition as a deterministic state transition F on the decoder's hidden state and prove (in Lean 4) that if F is a q-contraction (q < 1) under periodic conditioning, then the counting horizon is bounded by N* ≤ log(µ/2LC)/log q, where µ is the decision margin of any Lipschitz readout. Empirically, they measure this across six checkpoints (Llasa 1B/3B/8B, XTTS-v2, Qwen3-TTS 0.6B/1.7B), finding –8.3% median count error at k ≥ 6 repetitions while length-matched controls with distinct words show 0.0% error; they argue that the decoder stops acquiring distinguishable internal states at rate 0.50× the control rate, and that existing mitigations (repetition penalty) fail to remediate this structural limit.

---

## Major Concerns

1. **Theory-measurement gap: Premise is assumed, not verified.**
   - **Issue**: The formal theorem (Theorem 1) is conditional on Assumption 2 (F is q-Lipschitz with q < 1), but the authors explicitly state "we could not measure q itself" (Section 4.4). They measure only the *consequence* (capacity saturation), not the *premise* (contraction).
   - **Where**: Section 4.4, and throughout the framing of "what we establish empirically is the theorem's consequence, not its antecedent."
   - **Why it matters**: A conditional theorem that cannot verify its own antecedent is vulnerable to alternative explanations. Qwen3-TTS-1.7B exhibits no counting deficit yet shows a capacity gap, demonstrating that capacity loss is necessary but not sufficient (as the authors acknowledge). This raises the question: does contraction actually occur, or is the capacity phenomenon driven by something else entirely (e.g., attention geometry, residual-stream dynamics, output layer saturation)?
   - **What would address it**: Direct measurement of the Lipschitz constant q via a method robust to flat attention (e.g., neural network distance metrics, or a non-attention-based localisation of repetition boundaries); or an ablation that manipulates contraction directly (e.g., by adding skip connections or adjusting layer normalization) and measures both q and count error.

2. **The CTC judge is chosen without human ground truth.**
   - **Issue**: The paper disqualifies Whisper as "demonstrably the wrong one" because its decoder is autoregressive with a language model prior. Whisper does show bias (ratio 0.27–0.65 for k ≥ 4), but the paper offers no human-verified transcriptions of the generated audio to establish which judge is actually correct, or whether CTC's greater uniformity (1.00 at k ≥ 4) reflects accuracy or merely insensitivity to errors CTC-specific constraints preclude.
   - **Where**: Section 3, "Behavioral measurement, and why the obvious judge is wrong" and the audit against "concatenative audio with known ground truth."
   - **Why it matters**: The paper's headline claim (0.0% error for controls, –8.3% for repetitions) rests entirely on CTC outputs. If CTC systematically undercounts or overcounts in its own way—or if it simply cannot represent the acoustic phenomena at play—the entire empirical edifice shifts. The statement "no decoder, no language model, output length governed by the acoustics" sounds principled but assumes CTC's acoustic-only perspective is correct for measuring *intended* repetitions.
   - **What would address it**: (a) Hand-verification of a subset of CTC and Whisper transcriptions against the audio by native speakers; (b) a third judge (e.g., a fine-tuned wav2vec or a different CTC model); or (c) explicit tolerance bounds on CTC's own error rate under periodic conditions (is it truly 1.00, or 1.00 ± 2%?).

3. **Limited architectural and linguistic scope undermines generalization claims.**
   - **Issue**: All six checkpoints are codec-language models with softmax attention over text, English-only. The paper acknowledges this ("the claim concerns that class rather than AR TTS at large") but then presents results as an account of "autoregressive text-to-speech" failures, not specifically codec+softmax. The theorem itself is abstract and should generalize, yet the empirical proof is narrow.
   - **Where**: Section 5, "Implications and limits"; throughout the introduction and figures, which do not qualify claims by model family.
   - **Why it matters**: (a) Non-English phonologies and writing systems (e.g., tonal languages, right-to-left scripts, pitch accents) may exhibit different attention patterns and contraction behavior. (b) Future non-codec TTS models (e.g., diffusion, flow-matching, or alternative tokenizers) may not exhibit contraction or may exhibit it differently. (c) Attention mechanisms beyond softmax (e.g., sparse, linear, or hierarchical) may not satisfy Lemma 1 (attention dilution). The claim "periodicity, not length" is striking, but is it a property of AR TTS, or specifically of transformer TTS with softmax?
   - **What would address it**: (a) Test one non-English model (e.g., Mandarin, Russian, or Arabic) to verify the periodicity vs. length finding holds cross-linguistically; (b) test one non-softmax attention mechanism; or (c) explicitly frame the paper as "for codec-language TTS with softmax attention" rather than "for autoregressive TTS."

4. **The repetition-penalty ablation does not rule out alternative causes of its ineffectiveness.**
   - **Issue**: Sweeping XTTS-v2's repetition penalty over {2, 3, 5, 8} yields minimal benefit (–12.5% to –16.1%, change of –3.6 points over 4× penalty increase). The authors conclude: "this lever does not act on the quantity that sets the horizon." But penalty ineffectiveness could reflect poor penalty design, not proof of the horizon hypothesis.
   - **Where**: Section 5, "Implications and limits"; and the statement "Disabling the penalty entirely (β=1) breaks the control too, so no setting both preserves general quality and removes the deficit."
   - **Why it matters**: A well-designed penalty that explicitly targets contraction (e.g., one that softly constrains the state-space volume or enforces non-contraction via auxiliary loss) might work. The current penalty is multiplicative on logits, not geometrically targeted at the map's spectrum. The null result shows the existing penalty doesn't work, not that *no* penalty can work, and certainly not that contraction is the cause.
   - **What would address it**: (a) An auxiliary loss that explicitly penalizes contraction (e.g., by maximizing singular values of the Jacobian F' at each boundary); (b) evidence that models with experimentally reduced contraction (if measurable) also show reduced counting errors; or (c) a model trained with a contraction-penalizing loss and tested on the same repetition ladder.

5. **Effective rank is an indirect proxy for "distinguishable states"; count information itself is not null but degraded.**
   - **Issue**: The paper claims "the decoder stops acquiring states in which the count could be represented" (abstract) and "stops acquiring internal states at the rate non-repetitive text buys them" (Section 4.3). But the measurement (effective rank Neff of the generation trajectory) is an entropy aggregate, not a direct measure of count-predictive capacity. Section 4.4 concedes: "we could not claim the count is absent," finding instead that a probe retains the count at 0.97 of its early level on repeated text (vs. 1.11 on controls), i.e., *degraded* not *erased*.
   - **Where**: Section 4.3, "The states the decoder visits stop multiplying"; and Section 4.4, the discussion of what "stops acquiring" means.
   - **Why it matters**: The paper's framing ("the count is erased at a geometric rate," Theorem 1(iii)) is stronger than the data support. The effective-rank result shows the trajectory is low-rank, but rank deficiency does not prove information erasure—only that the information, if present, is compressed into a lower-dimensional manifold. The finding that count is *degraded* suggests the fixed-point attractor is not information-destroying; rather, the readout or a bottleneck downstream is failing to access the compressed signal. This is qualitatively different from Theorem 1's implication and suggests the bottleneck may not be in the hidden state but in the output path.
   - **What would address it**: (a) A direct probe for count information: train a linear classifier to predict the current count m from hidden states at each step, measure its accuracy as a function of k and position in the trajectory, and compare against a nonlinear variant to test information-theoretic limits; (b) ablate the output layer (stop head) structure to test whether a richer readout can recover the count even under low-rank state trajectories.

6. **The attention analysis for Lemma 1 is incomplete; two of six checkpoints are unverified.**
   - **Issue**: The authors verify Lemma 1 (attention dilution) for only four of six checkpoints. For Qwen3-TTS, which "sums text and acoustic embeddings per position," they state "no attention column belongs to the text and the lemma cannot be checked there." Qwen3-TTS-1.7B is the only model with zero count deficit—the absence of verification of Assumption 1 for this model is a gap.
   - **Where**: Section 4.3, final paragraph; Section 5, "What is proved, and what would refute it."
   - **Why it matters**: Qwen3-TTS-1.7B breaks the count-deficit prediction. If Assumption 1 (periodic conditioning via attention dilution) does not even hold for this model, then the theorem's silence on it is expected; but the paper should either (a) verify that Qwen3-TTS still satisfies Assumption 1 through an alternative mechanism, or (b) accept that for architectures without separable text attention, the theorem's applicability is unknown.
   - **What would address it**: An analysis of Qwen3-TTS attention or conditioning that demonstrates whether Assumption 1 holds by an alternative route (e.g., via the summed embedding geometry, or via attention over the acoustic output stream).

7. **Bootstrap confidence intervals for capacity gain are wide; claims of "disjoint" intervals need scrutiny.**
   - **Issue**: Figure 1(d) shows capacity gain with 95% bootstrap intervals. While the paper states "disjoint 95% intervals for all six panel checkpoints," visual inspection shows some intervals (particularly for controls) are quite wide. The repeated-vs.-control ratio (0.50) is claimed robustly, but the absolute gains are noisier, especially for smaller models (Llasa-1B, XTTS-v2).
   - **Where**: Section 4.3, Figure 1(d), and Table 1.
   - **Why it matters**: If the capacity-gain measurement is noisy, the quantitative link to the theorem's prediction (capacity gain ~ 0.50 × control rate) is loosened. A 0.50 ratio could arise from many mechanisms; the confidence that this reflects state saturation (vs. output monotony, or attention saturation) rests on the magnitude of the effect, and noise weakens that case.
   - **What would address it**: (a) Larger sample size (currently "bootstrapped over stimulus templates"; how many templates?); (b) a direct estimate of the ratio and its confidence interval, rather than separate estimates of repeated and control gains; or (c) a sensitivity analysis: how much does the ratio move if effective rank is computed with alternative definitions (e.g., rank-k approximation vs. entropy)?

---

## Minor Concerns

- **Notation and clarity**: The symbol C for orbit scale is defined as d(s₀, Fs₀)/(1−q) but is only interpreted intuitively; a brief remark on its meaning (scale of the attractor basin) would help.
- **Reference [1] is withdrawn**: Wang et al. (arXiv:2509.19852) is cited for alignment analysis but marked "Withdrawn 13 Feb 2026." This is unusual; has the citation been verified post-withdrawal, or is the withdrawal recent?
- **Table 1 label**: "capacity gain dNeﬀ/d log k" — the units and normalization (per what baseline?) are not explicit; does this control for model size?
- **Section 4.4 negative results**: The statement "we could not measure q itself" and "we could not claim the count is absent" are important but placed late (Section 4.4) and could be emphasized earlier, in the main results.
- **Generalization of Theorem 1 to non-Euclidean spaces**: The theorem works on "complete metric space S," but the application is to Rd. Do the results change for other geometries (e.g., hyperbolic, or learned manifolds)?
- **Assumed Lipschitz constant for the readout**: The theorem assumes g (the stop head) is L-Lipschitz. How is L estimated or chosen? Is L = 1 a reasonable default, or does the bound's tightness depend critically on this constant?

---

## Verdict

**Major revision**

The paper makes a bold and novel contribution—a formally verified impossibility result for repetition counting in AR decoders, with a creative experimental design (length-matched controls, CTC judge, effective-rank measurement). However, the gap between the conditional theorem and its empirical premise is substantial and unresolved: contraction is assumed, not verified, and Qwen3-TTS-1.7B's zero deficit with non-zero capacity gap shows the premise can fail while the predicted effect vanishes. The CTC judge, though better-motivated than Whisper, is not validated against human ground truth. The scope (codec-softmax-English) is narrow and acknowledged, yet claims remain broad. The repetition-penalty null result does not rule out alternative designs. And the state measurement (effective rank, count degradation not erasure) is indirect and compatible with bottlenecks downstream of the hidden state, not necessarily within it. These are not fatal—the core insight about periodicity vs. length is striking and robust across controls—but they require substantial evidence gathering and narrower claims. The formal verification is genuine and valuable, but it certifies an implication, not ground truth.
