# H9-PCR task

## Objective

Test whether **paired counterfactual ranking** on content-matched natural and
synthetic speech makes a compact, transparent detector transfer better than
ordinary label training to two predeclared unseen corpora.

## Deadline

The user authorized this independent positive-result search through
**2026-08-11 13:00 UTC**. H9 is a fresh loop; it neither repairs nor changes
the interpretation of H1--H8.

## Deliverables

1. Hash-sealed ODSS source and SONAR/ArAD target manifests under the HDD root.
2. A BF16, four-GPU-efficient Res2TCNGuard training/evaluation harness with
   identical-budget BCE, random-pair, and PCR conditions.
3. Exhaustive source-development and two-target results, fixed uncertainty,
   provenance/fingerprint audit, and a hard decision note.
4. A paper update only if the locked H9 gate passes; every `paper/main.tex`
   edit must immediately rebuild `paper/build/main.pdf` in the same commit.

## Non-negotiable boundaries

- H1--H8 outputs cannot choose an H9 target, split, loss, model, or threshold.
- H9 target labels/audio may not select a source hyperparameter, epoch, seed,
  architecture, augmentation, or loss weight.
- No opaque/proprietary Spectra result is a H9 baseline or success criterion.
- All large files remain below
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`; commit only code, text,
  compact ledgers, and summaries.
- A failed source-pairing, provenance, or primary gate stops H9-PCR. It does
  not authorize a changed target panel, source corpus, loss grid, or metric.
