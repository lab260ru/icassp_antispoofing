# Reviewer Bravo — H9-PCR ICASSP draft

## Summary

This is a compact controlled study of whether pairing a bona-fide utterance
with a documented natural--synthetic counterpart improves external transfer
beyond both same-pool BCE (B1) and an equal-edge, stratum-matched random-pair
ranking control (B2).  The experimental discipline is unusually good for a
small anti-spoofing training study: all methods use the same matched-only ODSS
pool; the final 12 source checkpoints are selected and hash-sealed before
target materialization; B2 directly addresses the otherwise obvious
``any-ranking-loss'' alternative; and the two-target ensemble endpoint and
shared-ID bootstrap were fixed in advance.  The reported ensemble result is
substantial relative to the controls (45.02% macro EER for P versus
51.78%/52.01% for B1/B2), and the paper appropriately avoids a SOTA or
deployment claim.

The present draft nevertheless overstates what its pairing key identifies and
does not yet make its novelty and uncertainty story sufficiently clear for an
ICASSP reader.  The result is best framed as a controlled *content-key-aligned
pairing* effect in one ODSS-to-(SONAR, ArAD) configuration, not evidence that
linguistic/semantic content itself is the operative cause.  With that boundary
made central, a stronger related-work comparison, and transparent AUROC and
conditional-uncertainty reporting, this could be a defensible focused ICASSP
paper despite weak absolute EER.

## Major concerns

### 1. The central construct is currently stronger than the evidence — mandatory

The title, abstract, and introduction use “content-aligned” and, in the
abstract, “linguistic content.”  However, the H9 audit establishes only a
documented ODSS `corpus/speaker/stem` content key, not transcript-level
semantic equivalence.  Changing a B2 partner also changes every
same-item-correlated property: speaker/recording identity, utterance duration,
source provenance, and potentially details of the synthesis conditioning.  It
therefore cannot isolate linguistic content as the mechanism.

The draft acknowledges this only late in the limitations section.  This is not
enough because the causal-looking construct is the title-level contribution.
Use “content-key-aligned,” “documented paired-item,” or “same-group” pairing
throughout (including the title and abstract), and state early that the
experiment isolates the *choice of documented pairing key versus a
different-group control*.  Reserve semantic/linguistic explanations as
hypotheses for a transcript-validated follow-up.  This change preserves the
positive controlled result while making its claim accurate.

### 2. Novelty and related-work positioning are too thin — mandatory

Five references do not establish that this is the first or a distinct answer
to the stated question.  The two cited anti-spoofing pair/contrastive papers
show that pairwise objectives exist, but the manuscript does not compare its
hinge-ranking formulation, same-item resynthesis setup, or random-pair
counterfactual to prior metric-learning, contrastive, Siamese, and
cross-corpus anti-spoofing work.  As written, a reviewer cannot tell whether
the contribution is a new objective, a more careful control for an existing
objective, or merely a new split of a known setup.

Conduct a focused citation-verified literature audit and add a short paragraph
or table that explicitly contrasts: (i) prior pair/contrastive supervision,
(ii) whether its pairs are matched natural--synthetic versions of the same
documented item, (iii) whether it controls an equal number of random pair
edges, and (iv) whether selection is source-only with held-out cross-corpus
targets.  If no direct comparator is found, say “to our knowledge” and scope
the novelty to the *controlled B2 comparison and terminal protocol*, rather
than claiming novelty for ranking itself.  Do not add unverified citations.

### 3. The result presentation omits crucial diagnostic metrics and conditions — mandatory

The protocol calls for EER and AUROC, while the manuscript reports only EER.
The audited P AUROCs are 0.557 on SONAR and 0.615 on ArAD; omitting them makes
the very weak absolute discrimination less visible than it should be.  Replace
or augment the main EER-only presentation with a compact central table that
contains every method/target EER, macro EER, P AUROC at minimum, the two
relative reductions, and the two bootstrap intervals.  If space permits,
report AUROC for every arm rather than only P.

The bootstrap intervals are valid for shared target-trial resampling of the
*already frozen four-seed probability ensemble*.  They do not quantify
variation over source voices, training seeds, another random pairing, or a
new source/target dataset.  Table 2 usefully exposes macro per-seed
heterogeneity, but its caption and the figure/result prose should explicitly
call the intervals “paired trial-bootstrap CIs conditional on the frozen
ensemble.”  The draft must not imply per-target significance or independent
training-run replication.  Point readers to the released per-target,
per-seed artifact if all 24 values cannot fit.

### 4. Fair-control and loss descriptions need precision — mandatory

The key fairness statement should not be weakened by imprecise wording.
Equation (1) defines only the unweighted rank loss; write the actual objective
for B2/P, e.g., \(\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}+\lambda
\mathcal{L}_{\mathrm{rank}}\), and make clear that \(\lambda=1.0\) is
selected on P source development then imposed on B2 as an equal-budget
control.  This does **not** establish B2 at its own optimum, which should be
stated as a control-design limitation.

Likewise, “same final budget” and “equal-budget” may be misread as equal
wall-clock compute among B1, B2, and P.  Ranked arms necessarily perform rank
forwards in addition to BCE.  The supported fairness claim is the same source
pool, BCE schedule, training envelope, and equal B2/P edge and rank-term
budget—not equal total compute for B1 versus ranked methods.  Correct this
explicitly.

### 5. The narrative needs one neutral protocol view — important revision

Figure 1 is readable and its values/CI encoding are effective, but “Content-
aligned advantage” presupposes the interpretation.  Use a neutral right-panel
title such as “P minus control macro EER,” retain the explicit lower-is-better
cue, and label its CIs as conditional paired trial bootstraps.  A small
pipeline graphic or schematic should then show: matched ODSS source pool;
B1/B2/P with B2/P’s matched edge count; source-only lambda/checkpoint freeze;
one terminal SONAR+ArAD evaluation.  That graphic would make the real paper
contribution—the counterfactual control and temporal firewall—legible faster
than the current prose.

## Minor concerns

- The introduction says pairwise objectives are “common” on the basis of two
  citations.  Either substantiate the breadth of that statement or soften it.
- Give a one-sentence description of what SONAR and ArAD contribute to the
  distribution shift (and cite their source documentation), rather than only
  trial counts and their common Arena ecosystem.  Readers otherwise cannot
  assess the scope of “cross-corpus.”
- Define the score orientation and EER computation once, and distinguish an
  unweighted two-target macro average from trial-pooled EER.  This avoids a
  common reproducibility ambiguity.
- “Complete terminal comparison” is inaccurate while AUROCs and full
  per-target seed cells are absent from the paper.  Call it the complete
  *predeclared primary EER comparison*, or include the missing diagnostics.
- The historical visibility of rows for the excluded public checkpoint is
  responsibly disclosed.  Keep it, but move it into a compact “non-blind
  historical context” limitation so readers do not mistake it for target-driven
  selection of H9.
- Make the artifact-release pointer concrete (repository URL or anonymous
  supplementary link at submission time) and list the source revision, target
  revisions, and key ledgers there.  Hash language is valuable, but not a
  substitute for a discoverable artifact.

## Limitations that must be disclosed, not repaired post hoc

The following are genuine limits of the evidence, but the locked H9 protocol
should **not** be retuned to address them after seeing its terminal results:

1. P remains a weak absolute detector (47.07%/42.98% EER; macro 45.02%), so it
   is neither a deployment result nor an Arena/SOTA comparison.
2. The four-seed ensemble is the predeclared endpoint, yet individual seeds
   are heterogeneous; the result is not “every seed improves.”
3. One source corpus, one 172k-parameter backbone, and two fixed targets do
   not establish general benefit across architectures, corpora, languages,
   codecs, or attacks.
4. The source--target test rules out exact canonical waveform duplicates, not
   speaker, text, generator, or broader lineage overlap.
5. The terminal evaluation is target-held-out for H9 selection but not
   literally blind, given the documented historical public-checkpoint context.

An independently preregistered replication on another source and backbone
would materially strengthen a future paper, but it must be a new study rather
than an H9 post-result selection exercise.

## Verdict

**Major revision; promising but currently a weak accept / weak reject border.**

The controlled B2 counterfactual, frozen source-only selection chain, and
two-target ensemble result are credible and potentially publishable as a
focused ICASSP contribution.  Acceptance should depend on replacing the
overstrong linguistic-content framing, establishing the precise novelty
against related pairwise anti-spoofing work, and making the weak absolute
performance plus conditional uncertainty impossible to miss.  The limitations
above should narrow the paper, not trigger post-hoc target tuning.
