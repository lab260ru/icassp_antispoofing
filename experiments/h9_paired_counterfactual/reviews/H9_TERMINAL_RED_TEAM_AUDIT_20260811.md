# H9-PCR terminal red-team audit — 2026-08-11

**Auditor:** independent Codex read-only audit.  This is an internal claim and
methods review, not external peer review.

## Verdict

**Pass, with a deliberately narrow claim.**  The terminal H9 gate is genuinely
passed under the locked protocol: PCR has lower frozen four-seed-ensemble EER
than both controls on both primary targets, exceeds both 10% relative
reductions, and has two fixed shared-ID bootstrap intervals wholly below zero.
The source-only handoff, fresh initialization, source--target canonical-audio
collision test, and output reconstruction all replay successfully.

The result is suitable for an ICASSP paper only as a controlled
**cross-corpus transfer comparison for this compact ODSS-trained model**.  It
is not evidence of high absolute anti-spoofing performance, state of the art,
or a general causal mechanism inside a detector.

## Evidence checked

- `PLAN.md`, `DATA.md`, `TARGET_MATERIALIZATION.md`, and
  `TERMINAL_EVALUATION.md` before inspecting any H9 output.
- Source freeze/materialization and lambda-selection reports: 23,883
  paired-eligible ODSS trials, 7,961 documented groups, 15,922 P edges and
  15,922 frozen B2 edges; 3,071 unmatched rows are excluded from *all* arms.
- The frozen 12-checkpoint ledger, whose source-only selection is
  `lambda_rank=1.00`; the ledger replayed with all 12 expected method x seed
  records and its current SHA-256 matched the terminal provenance.
- Terminal decision, target and seed metric tables, bootstrap artifact, and
  provenance.  All compact output hashes named by provenance matched on this
  audit.  Target labels were not reopened.
- Trainer/evaluator code and the focused unit tests.  The evaluator writes
  complete label-free predictions before reading the sealed labels, verifies
  target audio byte hashes, enforces all four seeds, uses the declared
  probability ensemble, and calculates the predeclared 2,000 shared,
  label-stratified bootstrap.  `tests/test_h9_pcr_evaluation.py` and
  `tests/test_h9_checkpoint_ledger.py` passed (7 tests).

## Terminal result to report

All values below are the predeclared mean-spoof-probability ensemble over the
four frozen seeds.  Lower EER is better.

| Target | B1 BCE EER | B2 random-pair EER | P content-aligned EER | P AUROC |
| --- | ---: | ---: | ---: | ---: |
| SONAR (n=3,948) | 58.88% | 56.09% | **47.07%** | 0.557 |
| ArAD (n=3,570) | 44.69% | 47.93% | **42.98%** | 0.615 |
| Unweighted two-target macro | 51.78% | 52.01% | **45.02%** | — |

PCR reduces macro EER by 13.05% relative to B1 and 13.44% relative to B2.
The fixed 95% bootstrap intervals for macro `P - comparator` EER are
`[-8.25, -5.25]` percentage points versus B1 and `[-8.54, -5.30]` points
versus B2.  Both target point estimates favour P; the protocol does **not**
provide separate per-target confidence intervals, so the paper must not claim
per-target statistical significance.

## Exact supported claim

> With a fresh 172k-parameter Res2TCNGuard architecture trained only on the
> hash-frozen paired ODSS pool, content-key-aligned natural/synthetic
> pair-ranking supervision lowered the predeclared two-target external macro
> EER relative to same-pool BCE and a fixed equal-count, stratum-matched random
> pairing control.

The qualified phrase “content-key-aligned” is important.  ODSS pairing uses
the documented `corpus/speaker/stem` key; it is strong source metadata but not
a transcript-level proof of text-disjoint source development.  The paper may
say the source development split is voice-disjoint, but must not upgrade that
to text-disjoint, speaker-independent target evaluation, or a causal account
of an internal representation.

## Material strengths

1. The comparison is unusually well controlled for a small training study:
   all arms use the same paired-eligible pool, BCE schedule, compact fresh
   architecture, six-epoch source checkpoint rule, four seeds, BF16 recipe,
   and fixed source-only selection.
2. B2 matters.  It retains every P spoof edge and the same split/language/
   corpus/generator strata and margin-term count, so P beating B2 distinguishes
   the matched partner from merely adding a ranking loss.
3. Lambda selection occurred before target access and applies identically to
   B2 and P; the source checkpoint matrix was sealed before target prediction.
4. Both fixed targets favour P and the gate has paired trial-level uncertainty,
   not only a raw point-estimate comparison.  The evaluator records zero exact
   source--target canonical waveform collisions and a complete reconstruction.

## Claim risks and mandatory caveats

### 1. Absolute performance is weak

P still has 45.02% macro EER, including 47.07% on SONAR.  This cannot be
written as a strong detector, a practical deployment result, or an improvement
over published/SOTA systems.  Its contribution is the controlled *relative*
effect, not raw accuracy.  Report AUROC alongside EER so the near-chance
operating regime is visible.

### 2. The bootstrap is conditional on the four frozen trained models

The bootstrap resamples target trials and correctly shares sample IDs across
methods, but it does not resample training seeds, source voices, or a new
pairing realization.  Individual seeds are heterogeneous: P is not lower than
B1 and B2 on every seed-target cell (for example, on SONAR seed 9101 P EER is
60.17%, versus 50.00% for B1).  Do not call the bootstrap a training-run
replication interval or claim every seed improves.  Include seed-level EERs
in a compact appendix/supplement or report their range in the main text.

### 3. One source and one compact backbone limit external generality

The evidence is ODSS-to-SONAR/ArAD only, with three source generators and one
fresh Res2TCNGuard architecture.  It cannot establish that alignment helps
other backbones, source corpora, attacks, codecs, languages, or future target
sets.  It is especially not a comparison against Spectra-AASIST or the Arena
leaders.

### 4. “Blind evaluation” is not defensible

The protocol explicitly records that a public, excluded checkpoint exposed
historical benchmark rows.  No H9 checkpoint loaded it and no target metric
selected H9 configuration, but the paper should say “fixed terminal external
evaluation” or “target-held-out selection,” not “blind benchmark.”

### 5. B2 is a fair fixed control, not an independently optimized method

B2 deliberately inherits P's source-selected `lambda=1.0`; that is right for
an equal-budget control but does not show random pairing at its own optimum.
Also, B2 matches P in edge count and declared strata, not necessarily in every
individual natural-utterance multiplicity.  State the actual control rather
than “all marginals are identical.”

## Minimum four-page paper additions

1. Lead with the bounded question and use a title such as *Content-Key-Aligned
   Pair Ranking for Controlled Cross-Corpus Anti-Spoofing Transfer*; avoid
   “robust,” “general,” “SOTA,” or a causal-mechanism title.
2. Make one central results table with the six target/method EERs, P AUROCs,
   macro EER, relative reductions, and the two macro bootstrap intervals.
   A caption must say that the intervals are paired, trial-resampling, and
   conditional on the frozen four-seed ensemble.
3. Use one method/pipeline figure: source-only ODSS pairing and lambda/epoch
   selection -> freeze 12 checkpoints -> one exhaustive SONAR+ArAD terminal
   call.  Explicitly show B1/B2/P and the same paired-eligible pool.
4. Allocate a short paragraph to the B2 rationale: it controls the added hinge
   loss, edge count, source strata, and source pool while removing content-key
   alignment.  This is the paper's main methodological value.
5. Include a plainly labelled limitations paragraph containing the weak
   absolute EER, conditional seed uncertainty, single-backbone/single-source
   scope, filename-derived content key, and non-blind historical context.
6. Put source hashes, target revisions, code, source-only lambda table, and
   all 24 seed-level target metrics in a reproducibility appendix or released
   artifact index.  Do not run a new target ablation or select a stronger
   model after this terminal result.

## Bottom line

The locked H9 gate is a legitimate positive result.  It becomes reviewable
rather than fragile if the manuscript foregrounds the B2-controlled relative
effect and makes its limitations impossible to miss.  A claim that PCR is a
generally stronger anti-spoofing detector would not survive review.
