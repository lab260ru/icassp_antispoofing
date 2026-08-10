# Supplementary S1 protocol — crest-factor evidence boundary

Status: locked on 2026-08-10 before any rendering. This is a deterministic
visualization of sealed results, not a new experiment or statistical analysis.

## Fixed question

Can one supplementary figure make the boundary of the frozen crest-factor
finding legible without creating a detector-reliance or causal claim?

## Allowed sealed inputs

The renderer may read only compact committed or hash-sealed records containing:

1. the fixed Spectra-AASIST/full-waveform/spoof adjusted associations across
   the five core corpora and the two held-out confidence intervals;
2. the four locked H2 waveform-quality retention rates and the fixed 90 percent
   arm gate; and
3. the sealed H4 full-waveform crest signed label-AUROC values across the five
   corpora.

It must reject score catalogs, model/audio/ASR paths, feature tables, H2B
outputs, response-like columns, and any input not in its fixed manifest. It
must not compute a new test, bootstrap, pooled estimate, threshold, ranking, or
cross-panel inference.

## Fixed layout and interpretation

The ordered corpora are ASVspoof2019_LA, ASVspoof2021_LA, ASVspoof2021_DF,
InTheWild, and ASVspoof5. Panel A shows only the pre-frozen H1 partial
association, with held-out intervals where sealed. Panel B shows only the four
H2 retention rates against the fixed 90 percent gate and labels detector scoring
unavailable. Panel C shows only H4 signed label AUROC and labels it raw label
separation, not detector reliance. The output is descriptive and cannot change
H1/H2/H2B/H3 gates or make H4 select a cue.

## Required artifacts and stop rule

The implementation must have a validated fixed-input manifest, rejection tests,
vector PDF, 300-DPI PNG, metadata JSON containing all hashes and caption, a
compact result note, and a visual inspection record. If a sealed input cannot
be revalidated or the output is unreadable, stop and record the failure; do not
substitute a result or change a visual scale after inspecting data.
