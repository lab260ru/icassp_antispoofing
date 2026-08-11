# H9-PCR manuscript review — alfa

## Summary

This is a clear, unusually well-audited controlled transfer study.  The
manuscript reports the locked ODSS-to-SONAR/ArAD comparison faithfully: the
four-model probability ensemble has 45.02% macro EER for P versus 51.78% (B1)
and 52.01% (B2), and the two specified shared-ID, 2,000-replicate bootstrap
intervals are below zero.  I independently checked the manuscript/PDF against
`PLAN.md`, `H9_TERMINAL_EVALUATION_001.md`, the terminal red-team audit, the
source-pairing contract, and the compact terminal metric tables.  The reported
numbers and stated firewall/ledger boundaries agree with those artifacts; the
focused evaluator and ledger tests pass (7 tests).

The result is publishable in principle as a *bounded controlled comparison*,
but the current wording over-identifies the manipulated factor as linguistic
content and under-reports the conditional nature and weak absolute performance
of the endpoint.  These are correctable manuscript issues, not reasons to
rerun or tune the sealed terminal study.

## Major concerns

1. **The P--B2 contrast does not isolate linguistic content as claimed.**
   The source contract defines the key as `corpus/speaker/stem`; it is
   documented pairing metadata, not a transcript or semantic annotation.
   B2 replaces the natural item with a different-group natural item while
   matching split/language/source-corpus/generator.  Thus it may also alter
   speaker, recording/prosodic correspondence, duration/phonetic overlap, and
   other same-item properties.  The paper acknowledges this late (lines
   240--244), but the title, abstract (21--33), introduction (51--61), and
   repeated “only” language (114--115, 211--212) are stronger than the
   design supports.  Rename the intervention and central claim to
   **content-key-aligned (documented same-item natural/synthetic) pairing**;
   say that it tests the bundled same-item relation, not that it establishes a
   linguistic-content mechanism.  “Only intended difference” is acceptable
   only if immediately qualified with this limitation.

2. **The uncertainty interval is conditional on the frozen four trained
   models, and seed instability must be more prominent.**  The shared-ID
   bootstrap correctly quantifies target-trial sampling uncertainty for the
   fixed probability ensemble; it does not resample training seeds, source
   voices, or an alternative random-pair draw.  The disclosed macro table
   already shows substantial heterogeneity: for seed 9101 P is 53.30%, worse
   than B1 (50.00%) and B2 (53.05%), whereas seed 9104 is 41.10%.  Revise the
   Results claim to say explicitly that the interval is *conditional on the
   predeclared four-seed ensemble*, not a training-run replication interval.
   A compact supplement/repository table should give all 24 target × method ×
   seed EERs; do not infer per-target statistical significance, since none was
   predeclared or calculated.

3. **The main result reporting does not yet meet the locked reporting
   contract or make the near-chance regime fully inspectable.**  `PLAN.md`
   calls for EER, AUROC, retained counts, orientation provenance, and all seed
   predictions for every target/method.  The paper gives EERs and macro seed
   EERs but omits AUROCs and orientation information.  The compact terminal
   table has, for example, SONAR AUROC 0.411/0.428/0.557 for B1/B2/P and ArAD
   AUROC 0.586/0.519/0.615.  These values are important context for a result
   whose EER remains 43--59%; they should be placed in a central table or an
   explicit supplementary table, with target class counts (SONAR 2274/1674;
   ArAD 484/3086 bona-fide/spoof) and score orientation.  Also replace the
   generic “repository preserves” statement (229--235) with a resolvable
   release locator: code commit, immutable compact artifact/ledger bundle,
   and conditions for accessing non-redistributable data.  Hashes on a local
   HDD are auditable internally but not by themselves a reproducibility route
   for a conference reader.

4. **“Equal budget” needs a precise scope.**  B2 and P have equal frozen
   edge count and rank weight, which is the right control for the principal
   comparison.  BCE-only B1, however, does not execute the additional ranking
   waveform forwards, so it is not wall-clock or total-compute matched.  In
   addition, B2 intentionally inherits P's source-selected lambda rather than
   receiving its own optimum.  State both facts in the method/limitations
   section.  They do not invalidate P--B2, but they limit the stronger reading
   that generic random ranking has been exhaustively tested or that all three
   arms used identical compute.

## Minor concerns

- Replace “content alignment improves transfer” in the abstract and conclusion
  with “content-key-aligned ranking lowered the predeclared two-target macro
  EER under this protocol.”  This retains the positive result without making a
  mechanism claim.
- State “fixed terminal external evaluation” or “target-held-out selection,”
  not “blind evaluation.”  The manuscript correctly discloses the historical
  public-checkpoint exposure (135--137); retain that disclosure prominently.
- The EER/CI figure is legible and numerically consistent with the terminal
  decision.  Its caption should add that the intervals are trial-resampling
  intervals conditional on the frozen ensemble, not seed-level intervals.
- The related-work paragraph is sparse for the strength of the novelty framing.
  Expand it with verified pairwise/ranking anti-spoofing work and distinguish
  the present fixed B2 control from supervised-contrastive and Siamese
  objectives; do not claim the first use of pairwise learning.
- The abstract's slash-form control numbers are correct but difficult to parse.
  Label them explicitly as SONAR/ArAD/macro for B1 and B2, as is done for P.

## Verdict

**Major revision.**  The terminal comparison is real and the current numeric
claims agree with the locked artifacts.  I would support submission after the
claim is narrowed to documented content-key alignment, conditional seed/trial
uncertainty and AUROC/orientation are reported transparently, and the release
path plus B2/compute qualifications are made concrete.  No new target
experiment or post-terminal retuning should be performed in response to this
review.
