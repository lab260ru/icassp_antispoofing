# Meta-review — round 5

Four independent reviewers (alfa, bravo, charlie, delta; delta on the
alignment-forum rubric). **All four returned Major revision.** There is no split
verdict to adjudicate, and the agreement is not diffuse: three of the four
independently identified the *same* sentence in the abstract as the paper's
weakest point.

## The consensus finding

> The abstract leads with the number that the paper's own evidence supports
> least.

The 4.1-fold horizon ratio ($\hat K_{\mathrm{rep}}=23$ vs $\hat K_{\mathrm{ctl}}=94$)
came from a post-hoc extension covering two of six checkpoints, with per-cell
sample sizes the paper described only as "few items per cell", and a curve that
declines rather than plateaus. Every qualification lived two pages later in the
Results. delta put it most directly: *"a reader currently cannot check it."*

This is a fair hit and it has been taken. The abstract now states, inline: post
hoc, two checkpoints, 3–6 items per cell, and "fitted saturation scale" rather
than the theorem's horizon.

## Concerns ranked by how much they should change the paper

**1. Symbol conflation between the theorem's $N^\ast$ and the curve fit (bravo).**
The sharpest technical catch of the round. Equation 1 defines $N^\ast$ as a
function of $q$, $L$, $C$, $\mu$ — none of which the extension measures. The
extension fits $\hat c(k)=N^\ast(1-e^{-k/N^\ast})$ to raw counts and calls the
free parameter $N^\ast$ too. Sharing the symbol silently upgraded a descriptive
fit into a confirmation of the bound. **Fixed**: the fitted quantity is $\hat K$
throughout, with an explicit sentence saying it is not the theorem's $N^\ast$.

**2. The extension is post hoc and was not labelled as such (delta, charlie).**
The $k\le32$ ladder failed to show saturation; we then extended the range and
found it. That is legitimate exploratory work and illegitimate to present as
though the range had been planned. **Fixed**: labelled post hoc at the point of
introduction.

**3. Exclusions least controlled exactly where the claim lives (bravo, charlie).**
Budget-truncated generations are dropped "throughout", including at $k=128$ where
output-length saturation *is* the phenomenon. The reviewers were right that this
needed checking rather than asserting. **Checked**: cap-hit items are censored
*downward* (the model was still generating when our budget stopped it), so
including them would deepen the apparent saturation. The exclusion is therefore
conservative against our own claim. **Fixed**: counts and direction both stated.

**4. The circularity control is not a like-for-like subset (charlie).**
Restricting the capacity comparison to "correctly rendered" items keeps 94.3% of
controls and 18.2% of repeated items. Conditioning on an outcome with a fivefold
different survival rate is a selection effect, not a control. **Fixed**:
disclosed at the point of use. Not resolved — it would need a matched-difficulty
subsample, which the item count does not support.

**5. Vote-counting as an evidentiary standard (alfa, bravo).**
"5 of 6", "4 of 6", "2 of 2", "80% of pairs" appear with different denominators
and no common test or multiplicity correction, while pooled CIs are computed at
the generation level. This is a real methodological weakness and is **not fixed**
— a checkpoint-level random-effects estimate is the right answer and does not fit
in four pages. It belongs in the supplement, and is recorded as outstanding.

**6. Reproducibility of the load-bearing experiment (delta).**
Which checkpoints, what $n$, what fitting procedure, and 27.5 GB of primary data
"available on request". **Partly fixed**: checkpoints named, $n$ and exclusions
given, fit described. The data-hosting question is a real limitation we cannot
close from inside the paper.

**7. Panel narrowness and Lemma 1 inapplicability (alfa).**
Six English-only codec-language checkpoints, with the dilution lemma uncheckable
on the Qwen family whose attention is not separable. Already scoped in Limits;
the reviewers judged the scoping adequate but the title and abstract framing
broader than the evidence. Partially addressed by the abstract edits.

**8. Lemma 1 may suffice without Assumption 2 (delta).**
If attention dilution alone predicts the observed behaviour, the contraction
premise — twice unmeasured — is doing no work. This is the most interesting
unaddressed criticism in the round and deserves a designed experiment, not a
sentence. Recorded as future work.

## What the reviewers did not dispute

No reviewer challenged the Lean artifact, the periodicity-vs-length dissociation,
the decision to abandon Whisper, or the honesty of reporting two failed attempts
at the contraction premise. delta explicitly named the last of these a strength.
The disagreement is entirely about how much the extension-ladder result can be
asked to carry.

## Standing weaknesses after this round

* Checkpoint-level inference is still vote-counting (concern 5).
* The extension covers two checkpoints; Qwen was generating during this round and
  will raise it to four.
* $\hat K_{\mathrm{rep}}$ declines rather than plateaus — a horizon predicts flat.
* Assumption 2 remains unestablished after two independent attempts.

## Verdicts

```
alfa     — Major revision
bravo    — Major revision
charlie  — Major revision
delta    — Major revision
---
Consensus: Major revision
```
