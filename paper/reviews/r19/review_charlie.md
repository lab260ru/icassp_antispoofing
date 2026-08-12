SEED: 8f2e6c91a4d370b5e1f9c826b7a04d3e

AXIS: 7 mod 8 → **Reproducibility gap** (`int("8f",16)=143`, `143 mod 8 = 7`)
STANCE: `int("2e",16)=46`, `46 mod 3 = 1` → **Maximally adversarial** (assume the result is artefactual until proven otherwise)

*(Verification note: I checked the paper's own claims against its cited public artifact directly — the working directory I have access to **is** the repository the paper points readers to, at the tag it names. That is unusual and I used it. Findings below marked "verified against repo" are not inference from the text alone.)*

---

### Summary

The paper claims to isolate a TTS repetition-counting failure from a generic length effect via a matched-control ladder, to show the obvious evaluation instrument (Whisper) is itself biased in the direction of the effect, and to formally state then empirically refute a "contracting state map" mechanism for the failure via a Lean-verified bound and a measured Lipschitz constant. The writing is unusually self-critical for an ICASSP submission — it discloses a full post-hoc decision chronology, runs its own adversarial checks, and reports a null result (the theorem) as a positive contribution. That transparency is real and should be credited. But under a maximally adversarial read, the paper's most dramatic piece of evidence — an out-of-sample quantitative prediction (65.3 points) confirmed by later data (64.4 points) on a fourth architecture — is the one claim the paper's own reproducibility apparatus does not back up: the supplement section it cites for the prediction's timing contains no trace of it. Combined with n=3 architecture-cluster statistics whose confidence flips under a wider (undefended) prior, and a mechanism-refutation built on a single checkpoint, the paper's rhetorical confidence outruns what a skeptical reader can actually verify.

---

### Major concerns

1. **Issue** — The abstract's causal-sounding claim that the effect is "not the judge-side artifact" is weaker in the body than the framing suggests. `[Evidence — alternative explanations]`
   **Where** — Abstract ("ruling out the judge-side artifact"); §3.2, "Disagreements do concentrate on repeated items, but in the direction that under-states the deficit (0.27 against −1.0 required for the confound; S14)."
   **Why it matters** — 0.27 is not the "ruled out" number the abstract implies; it's a residual in the predicted direction, just smaller than needed to fully explain the gap. A reader who only reads the abstract takes away "not a judge artifact" when the body's own number shows the judge is contributing *something*, just not enough to be the whole story. That's a real distinction for a paper whose entire second contribution is "the judge choice is itself a result."
   **What would address it** — State the residual explicitly in the abstract, or quantify what fraction of the 75-point gap the 0.27 confound-direction effect could in principle account for.

2. **Issue** — The formal refutation of the contraction mechanism generalizes from a single checkpoint. `[Evidence — missing ablations / generalization]`
   **Where** — §4, "On Llasa-1B, over 29 repeated items and their length-matched controls, q = 38.1 [32.9, 51.4] against 31.8."
   **Why it matters** — The paper's language ("The mechanism is refuted; the phenomenon is not") is stated as a panel-wide conclusion, but the actual Jacobian measurement — the only quantitative test of the theorem's premise — was run on one 1B checkpoint out of six. Nothing establishes that Llasa-3B/8B, XTTS-v2, or either Qwen checkpoint would give q > 1 too. A contraction mechanism specific to the larger or differently-trained checkpoints is not excluded by this experiment.
   **What would address it** — Run the same power-iteration estimator on at least one more checkpoint from a different architecture family before calling the mechanism refuted across the panel, not just on Llasa-1B.

3. **Issue** — The control arm's own Jacobian is also expansive (q=31.8), which the paper doesn't engage with as undermining the premise more broadly, not just for the periodic case. `[Evidence — alternative explanations]`
   **Where** — §4, same passage.
   **Why it matters** — If a plain non-repeated control sequence already has q=31.8 ≫ 1, the "per-repetition state map contracts" premise was implausible on priors for *any* input to this decoder family, not specifically falsified by periodicity. That's a different and more deflationary conclusion than "repetition makes the map more expansive rather than less" (abstract) — it suggests the theorem was never a live mechanism for this architecture class, repeated or not, which is a much less interesting/precise finding than the paper presents it as.
   **What would address it** — Discuss what q>1 on the *control* implies for the premise's plausibility a priori, rather than framing q=38.1-vs-31.8 purely as a repeated>control contrast.

4. **Issue** — The n=3-family hierarchical model's headline confidence is fragile to a prior choice the authors themselves flag but do not resolve. `[Statistical rigour]`
   **Where** — §3.1, "under the widest prior on the group-level scales that interval does touch zero (−1.6, P=0.956)."
   **Why it matters** — With 3 clusters, a Bayesian hierarchical model's between-family variance is barely identified from data at all — the posterior is doing what the prior tells it to when data are this sparse at the top level. The paper is honest about this in-text, which is good practice, but then proceeds to hang a specific quantitative out-of-sample prediction (65.3 points, precise to one decimal) on this same fragile model (see #6). A number whose sign flips under a "less defensible" prior should not also be the number a "successful pre-registered prediction" headline is built from.
   **What would address it** — Either drop the illusion of point-precision on the predictive gap (report it as an order-of-magnitude claim: "plausibly 30–90 points"), or show the CosyVoice-2 confirmation is robust across the same prior sensitivity sweep already run for the interval.

5. **Issue** — Novelty relative to the closest prior work is understated in exactly the direction that matters. `[Novelty]`
   **Where** — §5, citing [13] (Venkatesh, arXiv:2605.09239) as "the rival the evidence favours."
   **Where verified** — I pulled [2605.09239v2] directly: its abstract states "Linear probes on the residual stream decode the correct count with near-perfect accuracy at every post-embedding layer and they do so even at the exact layers where the wrong answer crystallizes in the output" — i.e., representation-survives/policy-fails for repeated-token counting is *already established*, in text LLMs, by this cited paper.
   **Why it matters** — The paper's §5 causal patching result (rank-1 count-coordinate transplant that "transplants cleanly and the decoder ignores it") is presented as this paper's own discovery of the dissociation, with [13] cited merely as the "rival hypothesis" being adjudicated. But [13] already demonstrated the representation side of the dissociation directly via probing, in a closely related task. The genuinely novel piece here is the *causal* patch experiment and the *TTS-specific, periodicity-isolated* framing — that's real, but the abstract/discussion phrasing ("Deciding between them needs a manipulation... and we ran two") reads as though the dissociation itself, not just its causal confirmation, is this paper's contribution. That's a subtly overstated novelty framing given [13] predates it as cited.
   **What would address it** — Explicitly state in §5 that [13] already found the representational half of the dissociation via probing, and that this paper's contribution is (a) the causal intervention confirming the policy-side failure, and (b) showing the same pattern is periodicity-specific in a different modality (speech).

6. **Issue** — The single most load-bearing empirical claim in the paper — an out-of-sample quantitative prediction confirmed almost exactly (65.3 predicted vs. 64.4 observed on CosyVoice 2) — is explicitly attributed to a supplement section that, in the artifact the paper directs readers to, does not contain it. `[Reproducibility]`
   **Where** — Main text §3.1: *"a gap of 64.4 points against the 65.3 predicted, from a fit committed before the checkpoint had been run (S16 dates both)."*
   **Where verified** — I checked this directly against the repository the paper cites as its public artifact ("public at tag v1.1-icassp of https://github.com/lab260ru/icassp_antispoofing"). At that exact tag, `paper/supplementary/supp.tex` §S16 ("Chronology: what was decided when," `\label{sec:s16}`) is a 22-row commit-timestamped table of *other* post-hoc decisions (judge switch, ladder extension, probe retraction, etc.) — it contains **no row, no mention, and no timestamp for CosyVoice 2, the 65.3-point prediction, the 64.4-point observation, or any "fourth architecture" acquisition**. A full-text search of the supplement for "CosyVoice", "64.4", or "fourth architecture" (outside of one unrelated 64.4% figure in a different section about something else entirely) returns nothing beyond a single reference-list citation of the CosyVoice 2 paper.
   **Why it matters** — This is the paper's strongest rhetorical move: a genuinely falsifiable, pre-registered numeric prediction that came true almost to the point, which is exactly the kind of evidence that should be hardest to fake and easiest to trust. But as things stand, a reader cannot verify that the prediction actually preceded the CosyVoice-2 run — the one piece of documentation the paper points to for that claim isn't there. Everything else in this paper's reproducibility story (Lean proofs, stimuli, `build.sh`, the specification curve, the chronology table for every *other* forking decision) genuinely does exist in the cited repo and is unusually thorough by ICASSP standards — which makes this one gap more conspicuous, not less: the authors clearly know how to document a pre-registration claim, and did so everywhere except the one place carrying the most evidentiary weight.
   **What would address it** — Add the CosyVoice-2 prediction/observation pair to the S16 chronology table with its commit timestamp (the same standard applied to every other post-hoc decision in that table), or point to the specific commit hash where the prediction was recorded before the CosyVoice-2 checkpoint was run, in-text.

---

### Minor concerns

- Table 1's Qwen3-TTS-1.7B row shows 0.0% median count error but only 38.9% exactly-right — not a bug, but worth one clause in the caption explaining how a zero median error coexists with a majority of non-exact renderings, since a reader skimming the table could misread it as "solved."
- The label "S12" is reused for at least two distinct claims (probe-checkpoint splitting and the VITS/F5-TTS duration story) — makes the supplement harder to cross-reference from the main text alone.
- §3.2's "0.27 against −1.0 required for the confound" is stated with no units or scale definition in the main text — a reader can't sanity-check this number without the supplement in hand.
- The intro's "six hundred sixty-six thousand six hundred sixty-six" example is vivid but risks reading as a flourish rather than a stimulus actually in the ladder (the ladder's k range tops at 32/128, not million-scale) — a brief note that number-phrase stimuli are a separate, smaller category (mentioned only in passing under "Stimuli") would avoid the mismatch.
- n=90 for the CosyVoice-2 confirmation (single checkpoint, presumably 30 items × 3 seeds) is thin for the number the paper otherwise treats as decisive; an explicit CI on the 64.4% itself (not just the panel-level Bayesian gap) would help.
- Self-citation [7] (arXiv:2604.08591, same authors, dated 2026) is called a "companion" without noting it is contemporaneous, unreviewed work by the same group — fine to cite, but should be flagged as such rather than presented alongside the peer-reviewed prior art list without distinction.

---

### Belief update

Modest, and directional rather than confirmatory. I update toward "TTS repetition failures are at least partly about periodicity of the token sequence, not raw length" — the matched-control design is genuinely the right instrument for that question and the length-matched controls sitting near zero error is a clean, hard-to-fake pattern replicated across six checkpoints. I do **not** update toward the specific causal story the paper's discussion leans on (representation survives, readout fails) beyond what [13] already established for text LLMs — this paper adds a causal patch and a new modality, which is real but narrower than the framing suggests. And I explicitly do **not** update on the CosyVoice-2 "prediction confirmed" narrative, because I cannot currently verify it was a prediction at all rather than a post-hoc-fitted number reported as if pre-registered — the one document that would settle this doesn't contain it.

### Verdict

**Major revision** — the paper's most dramatic evidentiary claim (a pre-registered numeric prediction borne out on a held-out architecture) is unsupported by the very chronology document it cites as its proof, and that gap has to close before the claim can be taken at face value `[Reproducibility]`.
