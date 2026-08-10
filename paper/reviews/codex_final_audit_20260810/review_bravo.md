# Codex review bravo — presentation, template, and supplement audit

**Scope.** Read-only review of `paper/main.tex`, the compiled
`paper/build/main.pdf`, the vendored ICASSP-2026 `spconf`/`IEEEbib` inputs and
local readiness documents, plus the completed H5, H6, and Supplementary S1
artifacts. This is an internal Codex review, not external peer review or an
official ICASSP compliance decision.

## Summary

The draft has a disciplined and unusually clear evidentiary boundary. Its
central conclusion is correctly narrow: the discovery-frozen
Spectra-AASIST/full-waveform/spoof crest-factor association does not meet the
registered five-corpus portability rule, and the attempted H2 waveform panel
cannot yield a detector-sensitivity claim because three arms fail the
predeclared quality gate. The source consistently separates label separation,
score association, and causal sensitivity. The new S1 rendering makes this
boundary especially easy to inspect; the H5 and H6 notes are also explicit
about their descriptive-only scope.

The compiled PDF is legible and structurally clean: it has four technical
pages and a references-only fifth page. The local single-anonymous preflight
passes all of its checks (five US-letter pages, Table 1 on page 3, References
on page 5, and all detected fonts embedded). The named author block matches
the authorized local record and the template source implements the expected
two-column `spconf` layout. This is a strong readable working draft, but it is
not yet an official ICASSP-2027-template or PDF-eXpress validation.

## Major concerns

1. **The one main-paper figure is not the most direct visual evidence for the
   paper's headline.** Figure 1 is a useful registry pass-count summary, but
   it devotes a full technical page to the 19 *other* descriptive units. The
   main result—the fixed crest slice's attenuation/null confirmation and the
   H2 gate failure—remains distributed across the abstract, prose, and Table
   1. A reader can follow it, but the visual hierarchy makes the paper look
   more like a catalogue of positive associations than a conservative
   evidence-boundary audit.

   **Concrete fix:** for the next substantive paper revision, replace or
   substantially compact Figure 1 and use a concise evidence-boundary figure
   centred on the fixed crest slice and the blocked H2 gate. The existing
   deterministic S1 is a suitable *source* for that decision, but it must be
   formally imported as a main-paper figure with a source/claim audit and a
   fresh compile; it should not simply be copied ad hoc. If Figure 1 remains,
   retitle it as a ``five-corpus rule pass-count summary'' so that readers do
   not infer that it displays effect directions or magnitudes.

2. **Final reproducibility and submission provenance are not yet externally
   actionable.** The manuscript says that the accompanying artifact bundle
   preserves code and compact results, but also says a citable archival
   locator will be added later. The current repository provides excellent
   internal hashes, manifests, and result notes, yet a conference reviewer
   cannot resolve an archival artifact from the PDF alone.

   **Concrete fix:** before submission, add an authorized persistent project
   locator (or an explicitly permitted repository URL and release tag) and a
   one-sentence availability statement. It should identify the immutable
   result/manifests without exposing raw audio, model weights, or credentials.
   In the same finalization pass, reconcile the supplied 2026 style files with
   the official ICASSP-2027 kit and run the conference's official checker.

## Minor concerns

1. Figure 1's caption says colour identifies waveform view, but the plot has
   no colour key. The row labels already state the view, so either add a small
   legend or remove the colour assertion. Typesetting feature names with
   reader-facing spaces (for example, ``spectral flatness'') would reduce the
   source-code feel of a page-sized figure.

2. Table 1 correctly names the three failing H2 arms, but a reader may not
   immediately see that polarity is the sole arm that clears the 90% gate.
   Add ``polarity: 99.9%'' or ``only polarity clears the gate'' in the H2 row
   if space permits. This makes the all-four-arm design auditable without
   changing the conclusion.

3. H3 is responsibly marked unrun, yet the third introductory contribution is
   still a prespecification rather than a completed empirical result. Consider
   moving it under a short ``conditional next stage'' sentence, or label the
   contribution explicitly as a registered design component, to prevent a
   skim reader from mistaking it for a mitigation result.

4. Supplementary S1 is visually strong, but its abbreviated corpus labels
   (`ASV19 LA`, `ASV21 LA`, `ASV21 DF`, `ASV5`) differ from the manuscript's
   canonical names. If it is ever imported into the paper package, standardize
   names or define the abbreviations in the caption. The current direct panel
   labelling and explicit detector/causal disclaimers are otherwise excellent.

5. H5 and H6 are sound standalone descriptive extensions, but neither is
   needed to establish the current paper's result. H5 has six explicitly
   degenerate feature/view cells and 17 terminal view-stable entries; H6 finds
   heterogeneous within-class score agreement. Adding either now would require
   a new question, methods description, and space budget, and risks diluting
   the focused negative-portability story.

## Supplement decision

**Do not add citations to S1, H5, or H6 in `main.tex` in the present working
draft.** The manuscript is self-contained for its stated conclusion, and the
local submission documents do not establish that an unarchived supplementary
bundle is accepted by ICASSP 2027. Preserve S1 as a reproducible internal
supporting artifact; consider promoting it to replace/condense Figure 1 only
after the submission package and artifact locator are authorized. Keep H5/H6
as future-direction evidence unless a separate, locked paper-scope revision
integrates their question and interpretation.

## Template and layout finding

No local layout defect blocks continued drafting. The PDF matches the pinned
user-provided ICASSP-2026 `spconf`/`IEEEbib` build inputs, the author block is
appropriate for the documented ICASSP-2027 single-anonymous working stage, and
the local readiness check passes. The only template-level blocker is external:
the official ICASSP-2027 bundle and its final submission checks still need
reconciliation before upload.

## Verdict

**Major revision before submission; accept as an evidence-conservative
internal draft.** The required revisions are presentation/provenance rather
than a request to overclaim or to rerun a failed causal experiment. The core
result should remain unchanged unless a separately locked experiment provides
new evidence.
