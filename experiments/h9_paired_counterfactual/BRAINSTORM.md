# H9 structured idea pass — positive-result search

**Date:** 2026-08-10
**Decision status:** H9-PCR selected; no H9 waveform, label, score, or target
result has been read by this decision record.

## Constraint and opportunity map

The completed H1--H7 loops are sealed descriptive or stopped studies. H8-SF
was a separately frozen score-fusion test and failed its mandatory
dominant-single-system comparison. It cannot be tuned or reframed. The next
idea therefore needs (i) a trainable, transparent model, (ii) a mechanism that
is not score fusion or a cue intervention, (iii) a same-budget control, and
(iv) multiple previously unused-for-H9 evaluation corpora.

Available assets are four 48-GB GPUs, a local, runnable public
Res2TCNGuard PyTorch checkpoint, and a compact public source candidate,
ODSS. The public Arena manifest gives only the following *metadata-level*
information for the intended H9 inputs: ODSS revision
`1968e6d0ef141c4572073695bdc1d17a8706177f` (26,954 trials), SONAR revision
`eca7c72ebdf0f7936a644605a56735ac8564dbd9` (3,948 trials), and ArAD revision
`350184966eeb5b46ff2acdabd8f4d12e41e582da` (3,570 trials). No target samples,
labels, model outputs, or metrics informed the choice.

## Divergent candidates

The idea pass used inversion (identify the shortcut ordinary label training
would exploit), dimension shifts (supervision, provenance, test chronology,
and deployment channel), and falsifiable hypothesis templates.

| ID | Candidate | Mechanism and prediction | Feasibility / reason not selected |
| --- | --- | --- | --- |
| A | **Content-aligned pair ranking (PCR)** | Match natural and synthetic renderings of the same text, then require the spoof logit to rank the synthetic rendering higher. This tests whether content-aligned supervision helps more than an equally sized random pairing. | **Selected.** ODSS is compact and documents matched material; transparent Res2TCNGuard can be trained in BF16. Same-data BCE and random-pair controls isolate the matching mechanism. |
| B | DeepVoice identity-disjoint GroupDRO | Hold out every source/target voice incident edge in RVC conversion data; GroupDRO should reduce worst identity-fold error. | Useful ablation/future work, but one 5,053-trial corpus cannot sustain a broad generalization claim and a subset was previously used for H2B waveform-quality work. |
| C | Prospective post-release generator panel | Freeze systems before generating a factorial modern-generator/channel panel. A prospective gap could be an important benchmark contribution. | Data collection, licensing, generator access, and enough balanced voices/channels exceed the remaining deadline. A small panel would be less convincing than a proper source-trained test. |
| D | Robust score fusion | GroupDRO over published score panels. | Terminally falsified in H8 against its frozen dominant single-system baseline; prohibited from reuse/tuning. |
| E | DFADD generator-family holdout | Train source-balanced detector and test an unseen diffusion/flow generator family. | Attractive but acquisition is tens of GB and its VCTK provenance complicates the available external panel. It is a fallback only if PCR’s source pairing hard-stop fails. |

## Selected hypothesis

**H9-PCR.** Given a fixed architecture, source data, augmentation, optimizer,
and pair count, a content-aligned natural-to-synthetic ranking term will
improve unseen-corpus EER relative to both (a) ordinary BCE and (b) an
equal-budget random opposite-class ranking control.

The key comparison is not merely "add a ranking loss": it is whether a
ranking pair whose documented source text is shared helps more than a randomly
chosen pair with the same labels and loss weight. This is a supervision test,
not evidence that a model's internal representation has causally removed
content or speaker information.

## Evidence boundary and success bar

H9 is paper-worthy only if its predeclared PCR method improves the external
SONAR+ArAD macro EER by at least 10% relative to **both** controls, improves
each target point estimate, and has fixed-panel paired bootstrap intervals
below zero against both controls. A source-only development gain, a single
target gain, a different source split, or a post-result loss weight cannot
rescue the claim.

The study may show that content-matched supervision transfers to an unseen
synthetic-speech panel; it may not establish universal robustness, causal
mechanisms inside a model, or leaderboard superiority over opaque systems.

## Sources used for direction selection

- ODSS dataset card: <https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ODSS>
- ODSS reference record: <https://zenodo.org/records/8370669>
- GroupDRO method reference: <https://arxiv.org/abs/1911.08731> (used only to
  assess the unselected identity-disjoint alternative).
- Recent cross-domain context: <https://arxiv.org/abs/2406.03512>.

The protocol, data firewall, and no-go criteria are frozen in `PLAN.md` before
any H9 data materialization.
