# Gap Analysis: What Existing Work Does NOT Cover

**Paper**: "Counting Collapse: A Formally Verified Attractor Theory of Repetition Hallucination in Autoregressive TTS"

---

## FLAGGED: closest related paper found — read this first

**Gao, Yang, Chen, "Clustered Attractor Manifolds and Dynamical Condensation in Self-Attention," arXiv:2608.08922 (submitted 2026-08-09, two days before this search was run).**

This is the single closest piece of prior theoretical work found in the entire search, and it must be
addressed explicitly in the paper's Related Work, not just cited. It:

- Treats self-attention as a **feedback dynamical system** (token representations determine the
  attention matrix, which updates the representations, iterated).
- Proves the existence of a **"clustered fixed-point manifold"** — i.e., attracting fixed points of the
  self-attention map, exactly the kind of object our Theorem's τ-step fixed point is an instance of.
- Frames the onset of clustering as a **phase transition** ("attention condensation") controlled by an
  "overlap gap" order parameter and attention sharpness — structurally similar in spirit to our
  contraction-constant q < 1 threshold, and to the Spectral Sensitivity Theorem regime boundary in the
  companion Whisper paper (`viakhirev2026dispersion`).

**Why it is not a scoop, and how our contribution differs (this must be argued explicitly in the paper):**

1. **No text conditioning / no autoregressive generation.** Gao et al. study a self-attention layer (or
   stack) acting on an initially Gaussian token cloud in isolation — a statistical-mechanics idealization,
   not a conditional generative model. There is no notion of a text prompt, no cross-attention to a
   fixed conditioning sequence, and critically no *periodicity structure in the input* driving the
   dynamics. Our Theorem's hypothesis is specifically about τ-periodic text conditioning ("very very
   very ... dog") inducing a τ-step contraction; periodicity of the driving input, not just the presence
   of attention feedback, is the mechanism. Their model has no analogue of τ or of a text sequence at
   all.
2. **No downstream readout / task consequence.** Gao et al. characterize the geometry of the fixed-point
   manifold itself. They do not connect it to any impossibility result for a task performed on top of the
   representations (counting, stopping, decoding a token index). Our contribution is precisely that
   connection: convergence ⟹ no Lipschitz counting/stopping readout can succeed past a horizon N*. This
   is the load-bearing novel claim of the paper, and it has no counterpart in their work.
3. **No formal verification.** Their result is a physics-style statistical-mechanics argument (mean-field,
   thermodynamic limit), not a machine-checked theorem. Ours is proved in Lean 4 + mathlib. This is a
   difference in *epistemic status*, not just presentation, and should be stated as such.
4. **No empirical validation on real TTS models.** Their paper has no experiments on trained
   autoregressive TTS systems, no repetition-hallucination case studies, and no measured N* against
   XTTS/Qwen3-TTS/Llasa.
5. **Timing.** The paper appeared on arXiv 2026-08-09; if our paper submits after this date, we are
   obligated to cite and distinguish it (done above), but it does not anticipate our specific claim, so
   priority is not at risk — *provided* we do this differentiation explicitly rather than ignoring the
   paper. Silently omitting it would be the real risk: a reviewer who knows the self-attention-dynamics
   literature will find it in five minutes and ask why we didn't cite it.

**Action taken**: `gao2026clustered` is included in `refs.bib` and discussed in `survey.md` Area 5 with
this same differentiation. Recommend the paper's Related Work section cite it explicitly with language
close to points 1–4 above, ideally in a single dedicated paragraph rather than a passing mention, given
how close the framing is.

No other paper found in this search comes as close to the core claim. Everything else below is either a
mechanism precedent (rank collapse, DEQ, attention over-squashing) at one remove, or an empirical
precedent (repetition in AR-TTS, repetition in text LMs) without the theoretical machinery, or vice versa.

---

## The 2-3 strongest "this has been done before" objections a reviewer could raise

### Objection 1: "Barbero et al. (2024) already proved transformers can't count — what's new?"

`barbero2024glasses` ("Transformers need glasses!") proves that decoder-only Transformers can be driven
into "representational collapse" — arbitrarily close final-token representations for distinct long
inputs — and explicitly connects this to counting/copying errors. A reviewer who knows this paper will
reasonably ask why our Theorem is not simply a TTS-flavored restatement.

**Honest response**: It is closely related but mechanistically and quantitatively different, and we must
say so explicitly rather than relying on the reader to infer it.
- **Mechanism**: Barbero et al.'s collapse is an *over-squashing* argument — information about a long,
  generic input sequence gets compressed into a single final-token representation as sequence length
  grows, exacerbated by finite-precision arithmetic. Ours is a *contraction* argument specific to
  *periodic* repetition structure: we do not require the sequence to be long in an unstructured sense,
  we require it to contain a repeated period τ, and we show the τ-step *decoder update map itself* is a
  contraction (Lipschitz q < 1) under that specific structure, which is a much narrower and more
  falsifiable hypothesis than generic over-squashing.
- **Quantitative output**: Barbero et al. give an existence/impossibility result (representations can
  become arbitrarily close) but not a formula for *how many repetitions* it takes. We derive a closed-form
  detection horizon N* ≈ log(D/(ε(1−q)))/log(1/q) and validate it against measured behavior — a
  quantitative, falsifiable prediction their result does not make.
- **Domain**: theirs is stated for general decoder-only LMs on abstract counting/copying tasks; ours is
  stated and empirically validated for the specific AR-TTS decoding setting (discrete audio-codec tokens,
  cross-attention to text, real trained checkpoints).
- **What we owe them**: an explicit statement that our Theorem can be read as a *special case with an
  explicit convergence rate* of the general over-squashing phenomenon they identify, restricted to
  periodic inputs — this is honest framing, not a dismissal.

### Objection 2: "Geshkovski et al.'s clustering-in-self-attention-dynamics papers, and now Gao/Yang/Chen's attractor-manifold paper, already say attention converges to fixed points — this is not new mathematics."

`geshkovski2023emergence`, `geshkovski2023mathematical`, and especially `gao2026clustered` (see flagged
paper above) all establish, in various formalizations, that iterated self-attention dynamics have
attracting fixed points / clusters. Dong et al. (`dong2021attention`) show a related doubly-exponential
rank collapse with *depth*. A reviewer could argue "fixed points in transformers" is now a saturated
sub-literature and ask what is left to prove.

**Honest response**:
- The existing fixed-point/clustering results are indexed by **depth** (Dong et al., in the number of
  layers) or by an **abstract continuous "time"/temperature parameter** (Geshkovski et al., Gao et al.)
  in a model with no generation process. None of them are indexed by **autoregressive generation step**,
  and none of them connect the existence of a fixed point to an **impossibility result for a downstream
  task computed by a separate readout head** (counting). Convergence-of-representations and
  impossibility-of-counting are logically related but not the same claim, and the literature has
  (as of this search) only the former.
- We are not claiming to be the first to observe "attention converges to a fixed point." We are claiming
  a **specific new instance** of that broad phenomenon (τ-periodic text conditioning ⟹ τ-step decoder
  contraction), tied to a **specific new consequence** (no Lipschitz counting readout beyond N*), and
  **formally verified**, in a domain (AR-TTS repetition hallucination) where — to the best of this
  search — no one has made the connection at all, empirically or theoretically.
- We should be candid in the paper that we are *applying and specializing* a known dynamical-systems
  view of transformers (Geshkovski et al., DEQ, Gao et al.) to a new setting and a new consequence, rather
  than inventing the "transformers-as-contractions" idea from scratch. Overclaiming novelty of the
  general framing, rather than the specific TTS/counting instantiation, is the failure mode to avoid.

### Objection 3: "Fu et al. (2021) already gave a theoretical proof that repetition is inevitable in text generation — and Xu et al. (2022) already showed repetition is self-reinforcing. Isn't your Theorem just this, ported to TTS?"

`fu2021theoretical` proves repetition is provably favored under standard decoding because many contexts
predict the same high-probability next token ("high inflow"). `xu2022learning` empirically shows repeated
sentences become more probable the more times they repeat (self-reinforcement). Both predate our paper
by several years and are in the direct ancestry we cite.

**Honest response**:
- Fu et al.'s theorem is about the **decoding rule** (greedy/near-greedy token selection under a fixed
  next-token distribution) and a **single repeated token**; it is a *distributional* argument about
  language statistics (inflow), not a *representational/dynamical-systems* argument about the hidden
  state. It does not address hidden-state convergence, does not give a convergence *rate*, and does not
  say anything about what a downstream readout (e.g., a counting head) can or cannot recover from the
  hidden state.
- Xu et al.'s result is empirical, not a proof, and again is stated at the level of output-token
  probability, not hidden-state geometry.
- Our Theorem is at a different level of the causal chain: we prove that the **hidden state itself**
  becomes ε-indistinguishable across different repetition counts, *before* any decoding rule is applied,
  and derive consequences for *any* Lipschitz readout, not just next-token greedy selection. This is a
  strictly different (and we believe strictly stronger, in the sense of being decoding-rule-agnostic)
  claim, and the two results are complementary: Fu et al. explains why the *decoder* keeps re-emitting the
  same token; we explain why *no* downstream mechanism, however designed, could have told the decoder
  how many times it had already done so.
- We should still credit Fu et al. and Xu et al. as the direct theoretical/empirical ancestors of the
  general repetition-inevitability claim, and be precise that our novelty is (a) the state-space /
  dynamical-systems level of argument, (b) the AR-TTS setting and attention-dilution mechanism specifically,
  (c) the quantitative horizon N*, and (d) machine verification — not the base observation that repetition
  is a systematic, non-accidental failure of AR generation.

---

## What is genuinely NOT covered by existing work (our contribution, stated plainly)

Based on this search, no paper combines all of the following, and each individual piece has at most
partial precedent as detailed above:

1. **A quantitative, closed-form detection horizon N\*** for when repetition/counting hallucination
   becomes inevitable, derived from a Lipschitz-contraction argument on the AR decoder's τ-step state map
   — as opposed to qualitative existence claims (Barbero et al., Hahn) or asymptotic/depth-indexed results
   (Dong et al., Geshkovski et al., Gao et al.).
2. **The specific mechanism of τ-periodic *text* conditioning** (not depth, not an abstract temperature
   parameter, not arbitrary long sequences) as the trigger for contraction, tied to an explicit
   attention-dilution Lemma (softmax over k near-identical keys ⟹ weight ≈ 1/k, entropy ≈ log k) that
   explains *why* repeated text becomes effectively periodic conditioning. We found no paper stating this
   specific mechanism for AR-TTS.
3. **Application to autoregressive TTS specifically**, with the failure mode (repetition/looping/skipping
   under numeric or reduplicated text) named and worked on empirically (`wang2025eliminating`,
   `liu2026experience`, `song2024ellav`, `xin2024ralle`) but, as far as this search found, never given a
   dynamical-systems explanation connecting it to transformer counting limits.
4. **Formal (Lean 4 + mathlib) machine verification** of the resulting theorem. This is close to
   unprecedented for a result of this kind: the small number of Lean-formalized ML results found
   (`george2026torchlean`, `cipollina2025formalized`) verify architecture-level execution semantics or
   classical (Hopfield/Boltzmann) network convergence, not a transformer-generation dynamical-systems
   theorem with a stated empirical consequence for real trained models. **Formally verified theorems about
   LLM/TTS failure modes appear to be essentially absent from the literature as of this search** — this
   is a genuine and easily defensible selling point, not an overclaim.
5. **A direct empirical link back to the companion Whisper paper's Spectral Sensitivity Theorem**
   (`viakhirev2026dispersion`): treating dispersion-vs-attraction regime transitions in context Jacobians
   as the same family of phenomenon across ASR (Whisper) and TTS (XTTS/Qwen3-TTS/Llasa), which is a
   claim about *architecture-general* attractor dynamics across encoder-only and decoder-AR speech models
   that, to our knowledge, no other paper attempts.

---

## Honest caveats about coverage of this search

- **arXiv search, not exhaustive.** This survey relies on arXiv abstract pages, ACL/NeurIPS/ICML/ICLR
  proceedings pages, and targeted web search. It is not a systematic review; a paper that never appeared
  on arXiv or under an easily-searched title could exist and was not found. The Gao/Yang/Chen paper
  above was found only because of a deliberately broad search for "attractor" + "self-attention" +
  "fixed point" terms — the same search strategy could plausibly miss an equally close paper phrased with
  different vocabulary (e.g., "equilibrium," "invariant manifold," "saturation").
- **Qwen3-TTS and Llasa are very recent** (Jan/Feb 2026); the surrounding published-venue landscape for
  these specific systems is thin (mostly arXiv preprints, GitHub/HF releases), which is expected for a
  panel including bleeding-edge models but means some Area 2 citations are necessarily preprints rather
  than peer-reviewed papers.
- **`wang2025eliminating` (arXiv:2509.19852) was withdrawn** by its authors on 2026-02-13 pending
  institutional approval. We still cite it (the content remains on arXiv and the withdrawal reason is
  administrative, not scientific misconduct), but this must be flagged in-text wherever cited, and a
  reviewer may reasonably ask us to downweight or drop it. It is retained here as the most directly
  relevant "someone else tried to fix this exact failure mode" citation, but the paper text should not
  lean on it as a load-bearing empirical comparison point without noting its status.
- **No paper was found and excluded for fabrication risk** — every reference in `refs.bib` was confirmed
  against a live arXiv/DOI page during this search. No candidate reference was dropped for being
  unverifiable; all leads that were pursued resolved to a real, checkable source.
