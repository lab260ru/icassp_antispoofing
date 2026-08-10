# Codex final red-team audit — reviewer Charlie

**Scope.** This is an internal, Codex-only, read-only novelty/narrative/scope
review. I inspected the compiled five-page `paper/build/main.pdf` (including
its extracted text), `paper/main.tex`, the sealed H1 aggregate, the H2
quality-gate report, the score-free H4 and H5 notes, and the score-only H6
note. The local single-anonymous PDF preflight passes (five US-letter pages,
Table 1 on page 3, references on page 5, embedded fonts). No external
literature search or reviewer runtime was used.

## Summary and belief update

The paper has a real, internally consistent result: the exact
Spectra-AASIST/full-waveform/spoof/crest-factor candidate frozen from the
discovery process is not portable under the declared five-corpus rule, and the
pre-score crest transformation panel cannot support a sensitivity claim because
three arms fail its declared quality-retention gate. The manuscript is notably
careful not to turn either observation into detector reliance or causality. The
repository's freeze, hash, and stop-rule discipline makes that narrow conclusion
more credible than a typical correlation-only shortcut claim.

My belief update is therefore positive about the *audit discipline and negative
result*, but not yet that the work has demonstrated a causal mechanism, a
robustness remedy, or a broadly general property of anti-spoofing models. The
paper should be evaluated and framed as a reproducible five-corpus
association-portability audit with an unsuccessful causal-test prerequisite,
not as a completed causal audit or a mitigation paper.

## Major concerns

1. **The contribution list is broader than the completed empirical evidence.**
   The introduction lists a quality-gated intervention specification and a
   conditional feature-conditioned mitigation evaluation alongside the completed
   association design. H2 has no detector score delta and H3 is not run; those
   are useful registered *future tests*, not empirical contributions on the
   same footing as H1. A skeptical reader could view the paper as a protocol
   paper with one failed candidate rather than a causal-audit result.

   **Evidence-safe improvement:** replace the three contribution bullets with
   two completed contributions and one clearly prospective item. For example:
   (i) a locked five-corpus, eight-score-artifact association-portability
   design; (ii) the observed non-portability of the exploratory
   discovery-frozen crest slice plus the non-promoted 19-unit descriptive
   atlas; and (iii) a *registered follow-up protocol* whose initial crest panel
   is stopped at the quality gate. Remove H3 from the contribution list or
   state there that it is "a conditional protocol, not an evaluated method."
   This is an editorial clarification; it requires no new experiment.

2. **"Pre-registered" and "frozen" need sharper provenance language.**
   The H1 registry and five-corpus criterion were locked before confirmation,
   but the H2 crest follow-up rule was introduced during the discovery loop.
   The paper discloses this in Section 3.3, but the abstract's
   "discovery-frozen" phrasing and references to "pre-registered" units can be
   read more strongly than the evidence supports if no public prospective
   registration is claimed. The sealed aggregate itself calls the H2 candidate
   exploratory.

   **Evidence-safe improvement:** use "registered in the locked repository
   protocol" for the 28-feature registry and "exploratory, discovery-frozen
   crest candidate" at the first abstract mention. Retain the explicit
   Section-3.3 disclosure that the follow-up rule was introduced during
   discovery. Do not describe H2 as confirmatory or imply prospective public
   preregistration.

3. **The title/causal-audit framing can still be read as stronger than the
   actual outcome.** The conclusion correctly says that no causal
   feature-reliance claim is made, yet the project begins from a "causal audit"
   vocabulary and devotes substantial space to a causal estimand that was not
   measured. This mismatch is likely to be the main novelty objection: a
   negative portability result for one exploratory candidate and a failure of
   one intervention family are informative, but do not establish a causal
   account of shortcut learning.

   **Evidence-safe improvement:** make "association-portability audit" the
   first noun phrase in the abstract and conclusion. A low-risk title variant
   is *Beyond Crest Factor: A Five-Corpus Association-Portability Audit for
   Speech Anti-Spoofing*. If the current title is retained, change the first
   conclusion sentence to say the work "defines a reproducible path toward a
   causal sensitivity test" rather than "presents a reproducible path ... to a
   testable claim." No causal terminology should be removed from the planned
   H2 design, only from the claimed result.

4. **The eight-score-artifact design needs one more explicit boundary.** Exact
   joins and orientation checks make released-score analysis reproducible, but
   they do not establish parity with a current executable implementation, score
   calibration comparability, or model independence. The current manuscript
   mostly avoids these claims, but its multi-model breadth is central to the
   story and should not be mistaken for eight independently reproduced model
   evaluations.

   **Evidence-safe improvement:** add one sentence to the existing limitation
   paragraph: "Accordingly, our estimands concern the pinned released score
   artifacts, not re-executed current model implementations or independent
   architectural mechanisms." Keep "published score artifacts" rather than
   "eight independent detectors" throughout. This is consistent with the H1
   inputs and with H6, and introduces no new claim.

5. **The main figure supports the atlas, not the central negative crest
   result.** Figure 1 visualizes only the 19 units satisfying the descriptive
   rule; the frozen crest candidate is absent. The caption tells the reader why,
   but the reader must combine that absence with a dense Table 1 row to see the
   paper's primary result. Also, bars encode the count of qualifying models but
   not effect size, direction, or uncertainty. This is defensible as an
   exhaustive registry display, but it is a weak visual anchor for a
   crest-centered title.

   **Evidence-safe improvement:** if one figure is changed, replace rather than
   add: use the already sealed S1 crest evidence-boundary display, whose fixed
   panels directly show the H1 crest slice, the H2 gate outcome, and the
   score-free H4 context. Its caption must retain the existing no-causal/no-new-
   test language. If Figure 1 remains, add a short Results lead sentence that
   explicitly directs readers to the Table 1 crest row before describing the
   19-unit atlas. Do not rerank the atlas or add a newly selected subset.

## H5 and H6 placement decision

**H5 — supplementary only.** H5 is a well-controlled, independent,
score-free measurement-agreement atlas: 414/420 finite cells, six explicit
InTheWild degeneracies, and 17/84 terminal descriptive view-stable units. It
is useful provenance context for a study that uses three waveform views, but it
does not test feature--score association, labels, an intervention, or the
primary crest portability conclusion. Adding its 17/84 count to the main paper
would invite an unsupported reading of those features as preferred cues or
preprocessing choices; adding the full matrix would consume scarce ICASSP page
space while opening a new measurement-validity narrative. Keep the sealed H5
heatmap and its six unavailable cells in supplementary material. At most, add a
single artifact-availability sentence in a supplement pointer: "A separate,
score- and label-free view-agreement atlas is supplied as supplementary
measurement context." Do not put H5 in the abstract, contribution list,
conclusion, or use it to qualify an H1 association.

**H6 — supplementary only, and do not use it as an independence claim.** H6's
complete 280-cell, within-class agreement matrix shows substantial heterogeneity
(range -0.629261 to 0.868770). This contextualizes why the paper retains an
eight-artifact panel, but H6 was locked after the H1/H2 outcomes, uses no
features, and measures rank agreement rather than detector validity or
architectural independence. A selected high/low pair in the main paper would
look post-hoc, and the headline range is not needed to establish the H1
conclusion. Keep the exhaustive matrix and its provenance as a supplement/data
artifact. A future supplementary H6 figure, if rendered under its own
display-only protocol, should show all 28 pairs and all ten cells per pair;
until then, do not add an H6 result sentence to `main.tex`.

Together, H5/H6 strengthen the research record and reviewer response package,
but neither changes the main paper's evidence gate. They should not be used to
inflate novelty, to claim a validated feature set, or to recast the eight score
artifacts as independent models.

## Minor concerns

1. In the H2 conclusion, prefer "the pre-frozen crest transformation **panel**
failed its quality gate" to wording that could sound like all possible crest
transformations are infeasible. The H2 report only rules out the four fixed
arms on the frozen ASVspoof2019 LA panel.

2. State the BH family once (for example, "across the complete 1,344-cell
within-corpus screen"), because the paper reports adjusted q values and a
large total cell count but currently leaves the correction family implicit.
The H1 analysis note documents this scope.

3. The limitation that metadata availability differs by corpus is appropriate,
but an artifact pointer to the per-corpus control ledger would make it more
auditable. Do not claim uniform covariate adjustment.

4. The paper has only seven background citations. This is not itself an error,
and this review performed no literature search, but the camera-ready revision
should ensure that the method is positioned against directly relevant
anti-spoofing robustness/shortcut and score-artifact evaluation work using only
verified sources.

5. The local PDF preflight is clean, but the user-provided ICASSP-2026 template
still requires reconciliation against the eventual official ICASSP-2027 kit.
This is a submission-readiness condition, not a scientific limitation.

## Verdict

**Major revision before submission; evidence-conservative and potentially
publishable as a narrow audit/null-result paper after the narrative is tightened.**

No new H1/H2/H3 experiment is justified by this review: all evidence-safe
changes above are wording, structure, or supplementary-placement decisions.
The required empirical threshold for a stronger causal or mitigation narrative
remains an independently protocolled, quality-valid transformation panel with
actual paired detector scores; the current H2/H2B gates do not provide it.
