SEED: 09fcccfc91d13b6ebd066e4e1d6fc4e6

AXIS: Cherry-picked qualitative examples
STANCE: Sharp but balanced

---

### Summary

The paper makes three claims: (1) autoregressive TTS models' failure to count repeated tokens is driven by repetition/periodicity, not sequence length — shown via a length-matched control ladder; (2) the standard instrument for measuring this (Whisper) is itself repetition-biased and must be replaced with a CTC recognizer; (3) a Lean4-verified "state-contraction" explanation for the failure is formally sound but empirically false — measured contraction factors never drop below 1 on any checkpoint. This is an unusually self-aware paper: it flags its own post-hoc threshold ($k \geq 6$), runs a 420-specification robustness curve, and explicitly kills its own leading mechanistic hypothesis in Section 4. That epistemic hygiene is real and should be credited. But the paper's flagship visual evidence (Fig. 1a) silently drops two of its six checkpoints with no stated rule, the mechanism-refutation (Section 4) is scoped to only five of eight systems in the panel without explanation, and three named stimulus families from the Methods section (tongue-twister, sentence-repetition, number-phrase) never resurface with results anywhere in the text. Combined with the fact that essentially every number a skeptical reader would want to audit — validation gates, exclusion criteria, per-condition rates — lives in an unseen supplement (S1–S28), the paper is currently unverifiable from its own text on exactly the points where a reader would most want to check for curation.

### Major concerns

1. **Issue** — The title claims to be "isolating" the counting failure, but the paper's own central moderator (whether periodic *ordering* vs. token *composition* drives the deficit) is reported as unresolved and flips sign between architectures. `[Narrative — overclaiming]`
   **Where** — Section 3.1, "Does the ordering carry it?": Qwen checkpoints gain 17.8 points from shuffling, XTTS-v2 loses 30.0, "intervals excluding zero both ways."
   **Why it matters** — A reader taking only the abstract/title away believes the mechanism has been pinned down to periodicity. In fact the one experiment capable of separating ordering from composition returns contradictory answers per architecture, meaning the paper isolates *that* length isn't the cause but does not isolate *what about* repetition is.
   **What would address it** — Soften "isolating" in the title/abstract to reflect that the length-vs-repetition axis is resolved but the periodicity-vs-composition axis is not, or run the $p=4$ shuffle test on more than two architectures before treating the disagreement as informative.

2. **Issue** — Three stimulus families introduced in Methods (sentence-repetition, tongue-twister, number-phrase, alongside the k-ladder and period-ladder) never appear again with quantitative results in Section 3 or the Discussion. `[Evidence — quality vs. quantity / missing results]`
   **Where** — Section 2, "Stimuli" vs. Section 3 (absent).
   **Why it matters** — If these families were run and showed a weaker or null effect, silently dropping them from the narrative is selective reporting of exactly the kind the paper's own robustness-curve ethos claims to avoid. If they were never analyzed, the "180 stimuli" figure in the abstract overstates what actually informs the headline numbers.
   **What would address it** — Report per-family results (even null ones) in the main text or explicitly state in Limitations that these families are supplement-only and why.

3. **Issue** — The claim that the counting failure is "not peculiar to autoregressive decoding" rests on one non-AR model per strategy (VITS, F5-TTS), and the authors themselves call this "suggestive rather than isolating." `[Evidence — baselines/ablations]`
   **Where** — Section 3.2, VITS/F5-TTS paragraphs; abstract, "of two non-autoregressive baselines one is immune, one is not."
   **Why it matters** — $n=1$ per architecture family is not a replicated ablation; the VITS/F5-TTS contrast could equally be explained by duration-modeling strategy, model scale, training data, or checkpoint quality rather than autoregressive vs. non-autoregressive decoding per se. The paper's own proposed explanation (per-token vs. total duration estimation) is itself untested beyond this single pair.
   **What would address it** — Add a second flow-matching and a second duration-per-token non-AR system before generalizing about the architecture class.

4. **Issue** — Precision-heavy headline numbers rest on very small validation audits, and a confirmatory test is framed as strong evidence despite a prediction interval wide enough to be barely falsifiable. `[Statistical rigor]`
   **Where** — Section 2 ("0.19 of the true count, n=12 over 6 donors" — repeated verbatim in the abstract and Table framing); Section 3.1 (CosyVoice 2 test against a partial-pooling posterior interval of [5.9, 93.3]).
   **Why it matters** — A ratio computed from 12 trials is quoted to two significant figures and propagated through the abstract as if it were a stable estimate; no CI is given for it (unlike almost everywhere else in the paper, which is otherwise interval-disciplined). Separately, an interval spanning 5.9 to 93.3 points will contain almost any positive gap — landing "nearer its mean than a wide interval entitles us to expect" is a much weaker confirmation than the prose implies.
   **What would address it** — Report a CI (even a crude Wilson interval) on the $n=12$ Whisper audit, and describe the CosyVoice2 result plainly as "consistent with, but not a tight test of" the partial-pooling model given the interval width.

5. **Issue** — Nearly every number a reader would need to audit for curation — exclusion rules, validation gates, per-condition rates, which five (not six) checkpoints and "two families" (unspecified) went into the contraction-factor measurement — is deferred to an unseen supplement (S1–S28), making the main text unverifiable on its own terms. `[Reproducibility]`
   **Where** — Throughout Section 3–4; explicitly Section 4, "Across five checkpoints and two families, q runs 21.0–347.7" (panel is six checkpoints, three families, elsewhere).
   **Why it matters** — The paper's single strongest claim — that the proposed mechanism is flatly false everywhere — is scoped to a subset of the panel with no stated reason for the exclusion. Without the supplement in hand, a reviewer cannot tell whether the omitted checkpoint(s)/family were dropped for a good instrumentation reason or because they were less clean.
   **What would address it** — State explicitly, in the main text, which checkpoints/families were excluded from the mechanism test and why, even in one sentence.

6. **Issue** — The paper's flagship figure (Fig. 1a) plots only 4 of the paper's 6 checkpoints, with no stated selection rule, and the entire manuscript contains zero example transcripts, spectrograms, or audio descriptions despite explicitly relying on a curated "audit by ear" sample for validity. `[Evidence — cherry-picked qualitative examples]`
   **Where** — Figure 1(a) caption ("one pair per checkpoint" — legend shows Llasa-1B, XTTS-v2, Qwen-0.6B, Qwen-1.7B only; Llasa-3B and Llasa-8B are absent); Section 2, "a stratified 165-clip sample... ships with each clip's transcript and scoring decision" (never shown or excerpted).
   **Why it matters** — This is the exact failure mode the paper otherwise guards against everywhere else (specification curves, mixed-effects models, pre-registered thresholds): the one place a reader visually absorbs "the curves diverge," two of six checkpoints are missing with no explanation, and there is no way to tell from the text whether they were dropped for legibility or because they looked less clean. Combined with the total absence of any in-text qualitative example — no transcript excerpt, no description of what a "count error" actually sounds/reads like on real (not concatenative) audio — the reader has to take the automated $\hat c$-extraction pipeline's behavior entirely on faith for the generative (non-synthetic) condition it's actually applied to; the only validation shown (n=12–24, concatenative audio with known ground truth) doesn't stress-test the messier transcription errors (homophones, filler insertion, partial words) that real model failures would produce.
   **Why it matters (cont.)** — The counting rule itself has documented edge cases the authors acknowledge shaping design decisions post hoc (e.g., "skipping unrecognised ones rather than stopping at the first miss, which would void credit for every later filler in a control") — exactly the kind of rule where seeing one or two real transcripts would let a reader sanity-check it, and none are shown.
   **What would address it** — State the selection rule for Fig. 1a's four curves (or plot all six, faceted if needed), and include at least 2–3 real (non-concatenative) example transcripts with their scoring decisions in the main text or a figure, not just a link to a 165-clip supplementary sample.

### Minor concerns

- The vivid intro example ("six hundred sixty-six thousand six hundred sixty-six") is never revisited quantitatively — it's rhetorical color with no corresponding number-phrase result in Section 3 (ties to Major concern 2).
- Fig. 1(b) caption is not self-contained: "hollow markers are the $p=k$ arm, matched to neither in vocabulary nor length" needs unpacking for a reader who hasn't already internalized the period-ladder design from the body text.
- The rhetorical tic "checkable rather than asserted (commit 6f34945)" and similar phrases recur several times; a bare commit hash isn't actually checkable without already having the repo open, so the phrase reads more as a plausibility cue than a verification path.
- Abstract sentence "(67.6 if the control's fillers are never cycled, and we quote the smaller)" is grammatically awkward and hard to parse on first read — could be split into two sentences.
- No error bars/dispersion are shown directly on the Fig. 1(a) curves themselves (only medians), even though CIs are reported liberally in the prose.
- Repo name "icassp_antispoofing" hosting this TTS-counting study, flagged as "(name predates this study)," is a minor discoverability nit for anyone trying to find the code from the paper alone.

### Belief update

I came away moderately more confident that autoregressive TTS counting failures track repetition/periodicity rather than raw input length — the length-matched control design (Section 3.1) is the paper's genuinely strong result and a real methodological contribution to how this class of failure should be studied. I also updated (moderately, given the 5-of-8-systems scoping issue in Major concern 5) against the specific Lean4 contraction-map hypothesis as an explanation — the measured $q$ values are so far from 1 that this looks like a solid negative result even if I can't fully audit its scope. I did not update on *what does* explain the failure: the ordering-vs-composition question is explicitly left open and inconsistent across architectures, so the paper's title promise of "isolating" the failure is only half delivered.

### Verdict

**Major revision** — the aggregate evidence for repetition-over-length is credible, but the flagship figure quietly drops two of six checkpoints with no stated rule and the paper shows zero qualitative examples despite leaning on a curated audit sample, so a reader cannot currently rule out that the cleanest-looking subset of checkpoints was what made it into the one figure meant to visually carry the paper's main claim.
