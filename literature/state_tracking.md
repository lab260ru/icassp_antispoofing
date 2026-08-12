# State-Tracking, Circuit-Complexity, and Formal-Language-Expressivity Literature

**Purpose of this file**: close a reviewer-flagged gap — "Adjacent formal/theoretical
literature on transformer state-tracking limits is absent, weakening the 'first formal
hallucination result' and 'quantitative impossibility' claims." This is a distinct body of
work from what `literature/survey.md` already covers (representational collapse, rank
collapse, attractor dynamics in self-attention). That survey is about transformers as
*continuous dynamical systems converging to fixed points*; this one is about transformers as
*bounded computational devices*, characterized by circuit complexity, formal-language
recognition, and RASP-style programmability. Every entry below was checked against a live
arXiv abstract page (and, where the venue is ACL/EMNLP/TACL, the ACL Anthology or MIT Press
page) during this search; titles, authors, venues, volumes and page numbers are as retrieved.
BibTeX is appended to `paper/refs.bib` under the section header "8. Transformer state-tracking,
circuit complexity, and formal-language expressivity limits".

**Bottom line up front, for the impatient reader**: nothing found here does what our paper
does. See "Does anything threaten the novelty claim?" at the end of this file for the full
argument. Short version: no — this literature bounds *architecture* (depth, precision, number
of chain-of-thought steps) against *worst-case* inputs for a *single forward pass*; we bound a
*measured dynamical property* (a contraction constant) against a *specific input structure*
(periodic conditioning) for what an *autoregressive decoder's own state* can support during
*generation*. The two are complementary, not overlapping, and the closest single paper
(Merrill & Sabharwal, "The Parallelism Tradeoff") is discussed in detail below.

---

## 1. Transformer state-tracking limits (Merrill & Sabharwal line, TC⁰/NC¹, RASP)

### Merrill, Sabharwal, Smith — "Saturated Transformers are Constant-Depth Threshold Circuits" (TACL 2022, vol. 10, pp. 843–856, arXiv:2106.16213) `merrill2022saturated`
Proves that transformers with *saturated* attention (a floating-point idealization closer to
what trained transformers learn than hard attention) and floating-point activations can be
simulated by constant-depth threshold circuits, placing the languages such a transformer can
recognize inside the complexity class TC⁰. This is the foundational circuit-complexity upper
bound the rest of the Merrill & Sabharwal line builds on.
**Relevance**: cited in Related Work as the origin of the "transformers are bounded-depth
threshold circuits" argument. Distinction from our result: this bounds what a *single forward
pass of an encoder* can compute as a function of *architecture* (depth, precision), over
*worst-case* input languages; it says nothing about what an autoregressive decoder's *state*
retains across *generation steps*, nor does it involve a measured dynamical/contraction
property of a specific trained model.

### Merrill & Sabharwal — "The Parallelism Tradeoff: Limitations of Log-Precision Transformers" (TACL 2023, vol. 11, pp. 531–545, arXiv:2207.00729) `merrill2023parallelism`
Extends the TC⁰ result to log-precision transformers (precision growing logarithmically in
input length), showing they too are simulable by constant-depth logspace-uniform threshold
circuits, so that (under standard complexity assumptions, L ≠ P) they cannot solve inherently
sequential problems such as many instances of state tracking. Frames this as a general
"parallelism tradeoff": any architecture as parallelizable as the transformer inherits the
same ceiling.
**Relevance**: the single closest prior paper to ours in spirit — it is the one place in this
literature that names a *tradeoff* rather than a bare impossibility, which is the same
rhetorical move our paper makes with the contraction factor $q$. Cited prominently in Related
Work, immediately followed by the distinction: their tradeoff is architecture-vs-parallelism,
indexed by precision and depth, over worst-case inputs, for a function computed in one pass;
ours is state-retention-vs-repetition-count, indexed by a *measured* contraction constant of
the decoder's own per-repetition map, over a specific common input structure (periodic
conditioning), for an autoregressive generation process. Neither entails the other: a
transformer could sit safely inside TC⁰ and still have $q \geq 1$ on a given input (no
collapse), or could contract sharply on periodic input while being architecturally identical
to one that does not, since $q$ is a property of the trained weights and the input, not of the
complexity class the architecture sits in.

### Merrill, Petty, Sabharwal — "The Illusion of State in State-Space Models" (ICML 2024, PMLR 235:35492–35506, arXiv:2404.08819) `merrill2024illusion`
Shows that state-space models (S4, Mamba, and related SSMs) — despite having an explicit
recurrent "state" — have expressive power similarly bounded to transformers (within TC⁰), and
in particular cannot solve the canonical hard state-tracking problem of composing permutations
in $S_5$ (the symmetric group on five elements), an NC¹-complete "word problem" that a plain
RNN can solve trivially. The paper's point is that the *presence* of a state variable in an
architecture's equations does not guarantee that variable can *track* state in the
complexity-theoretic sense.
**Relevance**: cited for the "illusion of state" framing itself, which is a useful contrast to
sharpen against: Merrill et al. ask whether an architecture's state variable can express hard
sequential composition *at all*, as a function of architecture class; we ask, given a state
that empirically *does* update at each step, whether *enough information survives in it* to
support a downstream Lipschitz readout after $k$ repetitions, as a function of a measured
contraction rate. Their negative result is about expressibility in principle (existence of
*any* weight setting); ours is about a specific trained model's measured dynamics. Also useful
rhetorically: "illusion of state" is precisely the right register to borrow when saying that
even where a decoder's hidden state *is* being updated every step, it need not be updating in
a way that preserves count information — our Theorem gives the geometric rate at which that
information is erased.

### Merrill & Sabharwal — "The Expressive Power of Transformers with Chain of Thought" (ICLR 2024, arXiv:2310.07923) `merrill2024expressive`
Gives an exact characterization of how many intermediate "chain of thought" decoding steps a
transformer needs to reach various complexity classes: $O(\log n)$ steps add little power over
a single forward pass, but a linear number of CoT steps (with a mild architectural addition)
lets a transformer recognize all regular languages, i.e., solve state tracking.
**Relevance**: cited to draw the sharpest available contrast for our setting. This paper's
message is that state tracking becomes possible for a transformer *once it is allowed to
externalize state into emitted tokens it can then attend back to* — chain of thought is
exactly extra readable memory. Our decoder is already autoregressive and already "has" a
chain-of-thought-like sequence of emitted tokens (the audio-codec stream itself), yet we show
that under periodic conditioning the relevant readout (a count) is still lost — because the
periodicity collapses the *conditioning* the decoder would need to keep the count-relevant
information distinguishable in its state, not because it lacks emitted intermediate tokens per
se. This sharpens the scope of our claim: we are not claiming transformers can never track
state given room to do so; we are claiming that under a specific, common conditioning
structure, the amount of externalized computation the model actually performs (repeating a
token) does not supply that room.

### Weiss, Goldberg, Yahav — "Thinking Like Transformers" (ICML 2021, PMLR 139:11080–11090, arXiv:2106.06981) `weiss2021thinking`
Introduces RASP (Restricted Access Sequence Processing Language), a programming language whose
primitives (elementwise ops, `select`/`aggregate` attention-like operations, sequential
composition) map onto what a transformer encoder layer can compute, giving a constructive
"assembly language" for reasoning about what a given number of transformer layers/heads can
express, including histograms, sorting and Dyck-language membership.
**Relevance**: cited as the origin of the RASP program-counting style of expressivity argument
underlying items 6–7 below; establishes the vocabulary ("a task is expressible if it has a
short RASP program") that the rest of this sub-literature, including the Chomsky-hierarchy and
length-generalization papers, uses. Distinction: RASP analyses are about what is *constructible*
in principle by hand-compiling a program into weights, a static/architectural question; our
result concerns what a *specific already-trained* decoder's state dynamics do under a specific
recurring input.

### Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran — "What Algorithms can Transformers Learn? A Study in Length Generalization" (ICLR 2024, arXiv:2310.16028) `zhou2024what`
Proposes the "RASP-Generalization Conjecture": a trained transformer tends to length-generalize
on a task exactly when the task has a short RASP-L program (a learnable subset of RASP) that
works uniformly across lengths; validates this across parity, addition, and other algorithmic
tasks, and uses it to explain and fix specific length-generalization failures.
**Relevance**: cited as the clearest statement of *why* counting-style tasks fail to
generalize to longer sequences in transformers trained normally — a program-length argument
about the *training/generalization* axis. Distinguishes cleanly from ours: their unit of
analysis is sequence *length* at *evaluation* time versus *training* time (a generalization
gap), for an encoder-style single computation; ours is fixed-length behavior *during
autoregressive generation*, where the same trained model is evaluated in-distribution on longer
prompts and still loses count, for a reason indexed by periodicity and a measured contraction
rate rather than by a train/test length mismatch.

### Liu, Ash, Goel, Krishnamurthy, Zhang — "Transformers Learn Shortcuts to Automata" (ICLR 2023, arXiv:2210.10749) `liu2023transformers`
Shows a low-depth transformer can exactly replicate any finite-state automaton's computation on
a length-$T$ input using only $o(T)$ (indeed, often $O(\log T)$ or $O(1)$) layers, by
hierarchically reparameterizing the automaton's recurrence into a "shortcut" solution rather
than simulating it step by step; shows trained transformers discover such shortcuts in
practice.
**Relevance**: cited as a positive counterpart to the mostly-negative results above — evidence
that transformers *can* track finite-state structure, given enough depth relative to a
*fixed* input length, via a parallel/shortcut computation rather than genuine recurrence. Useful
contrast for our decoder setting: our theorem's premise is that a periodic-input decoder's
*τ-step update map* is itself close to autonomous and contractive; a shortcut solution of the
kind Liu et al. describe would need to re-derive its output afresh from the *full* prompt at
each step, which is exactly what softmax attention over $k$ near-identical spans (our attention-
dilution Lemma) prevents it from doing cheaply — the shortcut argument assumes attention can
still discriminate positions/copies, which periodic dilution specifically erodes.

### Strobl, Merrill, Weiss, Chiang, Angluin — "What Formal Languages Can Transformers Express? A Survey" (TACL 2024, vol. 12, pp. 543–561, DOI 10.1162/tacl_a_00663, arXiv:2311.00208) `strobl2024formal`
A systematic survey unifying the (at first glance contradictory) formal-language-expressivity
results for transformers under a common set of assumptions (attention type — hard/soft/
saturated; precision; positional encoding; presence of chain-of-thought), situating the TC⁰/
NC¹/regular-language results, RASP-style constructive results, and Chomsky-hierarchy empirical
results within one framework.
**Relevance**: cited as the single reference for readers wanting the state of the art in this
adjacent literature, and as the source we use to state precisely *which* axis of that
literature our contribution sits outside of: every framework the survey organizes is
parameterized by architecture/precision/attention-type/CoT-budget, evaluated over worst-case or
uniformly-generated input families, for a function computed by the network as a static object.
None of the axes in that framework is "a dynamical property (contraction rate) measured on a
specific trained model, under a specific common input structure (periodicity), during
autoregressive generation" — which is the axis our Theorem is indexed by.

### Chiang & Cholak — "Overcoming a Theoretical Limitation of Self-Attention" (ACL 2022, pp. 7654–7664, arXiv:2202.12172) `chiang2022overcoming`
Extends Hahn's 2020 result (already cited in our paper as `hahn2020theoretical`) that
transformer confidence on languages like PARITY degrades (cross-entropy → 1 bit/string) as
input length grows, but shows this is a limitation of a *particular* transformer construction
rather than of self-attention itself: the authors exhibit transformers, using layer
normalization, that recognize PARITY and FIRST with perfect accuracy and improved length
generalization.
**Relevance**: cited alongside `hahn2020theoretical` (already in `refs.bib` and already cited in
our Discussion) as the necessary "on the other hand" — it shows that architectural details
(layer norm, position encoding choice) can rescue a transformer from a specific worst-case
periodic-language failure that looks superficially like ours. This is exactly the kind of
result we must distinguish from: theirs is about whether *some* transformer construction can
recognize a periodic *formal language* over arbitrary strings; ours is about whether a
*specific already-trained* AR-TTS decoder's *state* retains count information when its own
attention profile is empirically observed to flatten under repeated *conditioning* spans. A
reviewer could otherwise ask "doesn't Chiang & Cholak already show self-attention overcomes
periodicity problems?" — the answer is that they show a *possibility* result for a
hand-constructed encoder on a formal-language task, not a claim about what a given trained
decoder's contraction constant $q$ happens to be during generation.

### Bhattamishra, Ahuja, Goyal — "On the Ability and Limitations of Transformers to Recognize Formal Languages" (EMNLP 2020, pp. 7096–7116, arXiv:2009.11264) `bhattamishra2020ability`
An early systematic empirical/constructive study of which regular and counter languages
transformers can learn; provides transformer constructions for a subclass of counter languages
(e.g., Dyck-1, n-ary Boolean expressions) but shows performance on general regular languages
degrades with automaton complexity, in contrast to LSTMs, which handle these more uniformly.
**Relevance**: cited as background establishing that transformers' difficulty with
counter/state-tracking languages relative to recurrent architectures was observed empirically
well before the circuit-complexity explanations (`merrill2022saturated` et seq.) were proved;
supports the framing that our finding continues a long-documented architecture-level weakness,
now given a decoder-dynamics account specific to periodic conditioning during generation rather
than a general regular-language recognition account.

---

## 2. Counting ability specifically — already verified and in `refs.bib`, not yet cited in the paper text

These were found and verified in the earlier literature pass (`literature/survey.md`, Area 4)
and are already present in `paper/refs.bib`, but a check of the compiled paper
(`grep -oE "\\cite\{[^}]+\}"` across all `.tex` files) shows **none of them are actually
`\cite`-d anywhere in the current paper text** — only `barbero2024glasses`,
`dong2021attention`, `geshkovski2023emergence`, and `gao2026clustered` from that group are
cited. That is itself part of the gap the reviewer is pointing at: having a verified reference
sitting unused in the bibliography does not close a related-work gap. Recommend pulling these
into the actual Related Work discussion alongside the new entries above.

- **Hahn — "Theoretical Limitations of Self-Attention in Neural Sequence Models"** (TACL 2020, vol. 8, pp. 156–171, arXiv:1906.06755) `hahn2020theoretical`. Proves self-attention cannot model periodic finite-state languages or hierarchical structure without resources scaling with input length. Directly relevant to our periodic-conditioning setup; pairs naturally with the new `chiang2022overcoming` entry above (the "on the other hand" result).
- **Delétang, Ruoss, Grau-Moya, et al. — "Neural Networks and the Chomsky Hierarchy"** (ICLR 2023, arXiv:2207.02098) `deletang2023chomsky`. Large-scale empirical study (20,910 models) showing transformers and RNNs fail to length-generalize on non-regular formal-language tasks, including counter languages, without structured external memory. This is the paper named explicitly in the task brief under "Chomsky-hierarchy / formal-language expressivity" — confirmed present and verified, just uncited in-text.
- **Yehudai, Kaplan, Dar, et al. — "When Can Transformers Count to n?"** (2024, arXiv:2407.15160) `yehudai2024when`. Proves a sharp phase transition in counting ability governed by the ratio of embedding dimension to vocabulary size.
- **Zhang, Cao, You — "Counting Ability of Large Language Models and Impact of Tokenization"** (2024, arXiv:2410.19730) `zhang2024counting`. Argues constant-depth computation constrains exact counting and that tokenization compounds the problem.
- **Sälzer, Köcher, Kozachinskiy, Zetzsche, Lin — "The Counting Power of Transformers"** (2025, arXiv:2505.11199) `salzer2025counting`. Shows transformers can express certain nonlinear counting properties beyond prior linear-counting results — the "positive" counterpoint that keeps our impossibility claim properly scoped (impossible *past the horizon $N^\ast$*, not blanket).

No new BibTeX was added for this subsection — these five keys already exist in `paper/refs.bib` (see the "4. Counting / algorithmic limits of transformers" section of that file) and were re-verified only by inspection of the existing entries, not re-fetched, since the earlier survey already confirmed them against live sources.

---

## Does anything threaten the novelty claim?

**No — checked explicitly, and reported honestly.** None of the eleven papers surveyed here
(ten new plus the re-flagged Chiang & Cholak / Hahn pairing) proves an impossibility result for
counting *during autoregressive generation*, *indexed by a measured dynamical/contraction
property of the trained decoder*, under *periodic conditioning*. Every result in this
literature is indexed by one or more of: network depth, arithmetic precision, number of
attention heads, number of chain-of-thought steps, or program length in RASP — i.e., by
**architecture**, evaluated on **worst-case or uniformly sampled inputs**, for **a function
computed in a bounded number of layers or CoT steps** (frequently an encoder or a single
forward pass, and where autoregressive decoding is considered at all, as in
`merrill2024expressive`, it is treated as *more* computational budget being purchased by
emitted tokens, not as a source of collapse).

The single closest paper is `merrill2023parallelism` ("The Parallelism Tradeoff"), because it
is the one place this literature frames its bound as a *tradeoff* against a resource
(parallelism) rather than a flat impossibility — structurally analogous to our framing of $N^*$
as a tradeoff against the contraction factor $q$. But its tradeoff variable is precision/depth,
fixed at construction time and architecture-wide; ours is $q$, an empirical property of a
specific trained model's response to a specific recurring input, measured rather than assumed,
and it produces a rate *in repetitions* ($N^\ast$ in Eq.~1 of the paper) rather than an
asymptotic or a circuit-class membership statement. No paper in this search reports a
comparable per-repetition rate, ties it to a Lipschitz-readout impossibility that is agnostic to
decoding rule, or verifies such a result formally.

One methodological point worth flagging for completeness rather than as a threat: `merrill2024illusion`'s
use of $S_5$/permutation-composition as the canonical "hard state tracking" task is a useful
external validation that state tracking is a real, well-posed computational problem in this
literature, independent of our TTS setting — it is not a competing result, but it is the paper
whose vocabulary ("illusion of state") we most directly borrow and should credit for it.

---

## Suggested replacement paragraph for Related Work (≤90 words)

> A separate literature bounds what transformers can compute in principle: constant-depth
> threshold-circuit results tie expressivity to precision and
> depth~\cite{merrill2022saturated,merrill2023parallelism}, and RASP-style and Chomsky-hierarchy
> analyses characterize which formal languages a given architecture can
> express~\cite{weiss2021thinking,deletang2023chomsky,strobl2024formal}, including state-tracking
> tasks like $S_5$ composition~\cite{merrill2024illusion} and periodic
> languages~\cite{hahn2020theoretical,chiang2022overcoming}. These are architecture-vs-worst-case
> bounds for a single forward pass. Ours is a decoder-vs-measured-contraction bound for
> generation: a rate in repetitions, not a circuit class, on real trained models under a specific
> recurring input.

(75 words.)
