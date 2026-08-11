# Literature Survey

**Paper**: "Counting Collapse: A Formally Verified Attractor Theory of Repetition Hallucination in Autoregressive TTS"

This survey covers 39 verified references (38 + the companion paper) organized into the
seven areas requested. Every entry was checked against a live arXiv abstract page (or a
publisher DOI page for non-arXiv venues) during this search — titles, authors, years, and
identifiers below are as retrieved, not reconstructed from memory. BibTeX is in
`/home/kirill/icassp_antispoofing/paper/refs.bib`, keyed to match citations below (`\cite{key}`).

See `gaps.md` for the adversarial "has this been done before?" analysis, and in particular
the flagged near-neighbor paper at the top of that file.

---

## 1. Hallucination / stability failures in LLM-based & AR TTS

### Wang, Du, Xiang, Zhao, Zhao, Chen, Li, Guo, Ling — "Eliminating Stability Hallucinations in LLM-based TTS Models via Attention Guidance" (2025, arXiv:2509.19852) `wang2025eliminating`
Introduces an "Optimal Alignment Score" computed via Viterbi decoding over text–speech token alignment, and uses it as an attention-guided training signal inside CosyVoice2 to suppress repetition/omission artifacts. **Note**: this preprint was withdrawn by the authors on 13 Feb 2026 ("submitted prematurely without final institutional approval"); we cite it only for its problem framing, not as a settled result, and flag the withdrawal explicitly wherever cited.
**Relevance**: cited in Related Work as the most direct prior attempt to *fix* the exact failure mode (repetition/omission in LLM-TTS) we *theoretically characterize*; we differ by explaining the failure mechanistically rather than patching it empirically.

### Liu, Fang, Bi, Su, Wang, Han — "Experience-Calibrated Contrastive Decoding for Mitigating Hallucinations in LM-Based Text-to-Speech" (2026, arXiv:2608.00722) `liu2026experience`
A training-free, decoding-time contrastive-decoding method (ECCD) that separates text-alignment signal from acoustic-context "experience" to reduce hallucinated repeats/omissions across multiple LM-TTS backbones and languages, without architecture changes.
**Relevance**: cited alongside `wang2025eliminating` as evidence the field is actively patching this failure mode at decode time; we argue such fixes are consistent with (not a refutation of) our contraction argument, since they change the readout/decoding policy rather than removing the underlying attractor.

### Peng, Li, Mohamed, Harwath — "VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation" (2025, arXiv:2505.19462) `peng2025voicestar`
An AR encoder-decoder codec-LM TTS model with a Progress-Monitoring RoPE and Continuation-Prompt-Mixed training that gives explicit duration control and extrapolation to much longer utterances than seen at training time, improving robustness on long-form generation.
**Relevance**: cited as a duration-control-based mitigation strategy — orthogonal to our failure analysis but relevant because explicit duration/length conditioning is one practical way to break the periodic-conditioning regime our theorem describes.

### Du, Wang, Chen, Shi, Lv, Zhao, Gao, Yang, Gao, Wang, Yu, Liu, Sheng, Gu, Deng, Wang, Zhang, Yan, Zhou — "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models" (2024, arXiv:2412.10117) `du2024cosyvoice2`
Simplifies the CosyVoice LM backbone (drops the text encoder/speaker embedding, uses a pretrained text LLM), adds finite-scalar-quantization speech tokens and chunk-aware causal flow matching for low-latency streaming and offline synthesis at near-human quality.
**Relevance**: representative modern LLM-backbone AR-TTS system; cited as one of the systems that `wang2025eliminating`'s attention-guidance fix targets, and as a comparison point for the architecture family our theorem applies to (discrete-token AR decoder + cross-attention text conditioning).

### Song, Chen, Wang, Ma, Chen — "ELLA-V: Stable Neural Codec Language Modeling with Alignment-Guided Sequence Reordering" (AAAI 2025, arXiv:2401.07333) `song2024ellav`
Explicitly names "repetitions, transpositions, and omissions" and "infinite silence generation under greedy decoding" as the core AR-TTS failure modes, and fixes them by interleaving phoneme tokens ahead of their corresponding acoustic tokens for finer-grained local alignment.
**Relevance**: cited in Introduction as independent confirmation (from a different lab/architecture) that repetition/looping/truncation is a *known, named* failure class in codec-LM TTS — motivating why a general theory of it is worth having.

### Xin, Tan, Shen, Ju, Yang, Wang, Takamichi, Saruwatari, Liu, Li, Zhao — "RALL-E: Robust Codec Language Modeling with Chain-of-Thought Prompting for Text-to-Speech Synthesis" (2024, arXiv:2404.03204) `xin2024ralle`
Decomposes TTS into predicting prosody (pitch, duration) first, then using that as an explicit chain-of-thought condition and attention-guidance signal for acoustic token prediction, cutting zero-shot WER roughly in half versus VALL-E and reducing hard-sentence error rate from 68% to 4%.
**Relevance**: cited as the clearest existing empirical demonstration that giving the AR decoder an explicit duration/alignment scaffold breaks its worst failures — supporting our claim that decoders "in the wild," without such scaffolding, are left to rely on unconstrained self-attention that our theorem shows collapses under repetition.

### Wang, Chen, Wu, Zhang, Zhou, Liu, Chen, Liu, Wang, Li, He, Zhao, Wei — "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (2023, arXiv:2301.02111) `wang2023valle`
The original VALL-E paper: casts zero-shot TTS as conditional codec-token language modeling trained on 60k hours of speech, establishing the "AR decoder over discrete audio codes conditioned on text/prompt" paradigm that XTTS, Llasa, CosyVoice2, and (in spirit) Qwen3-TTS all descend from.
**Relevance**: cited as the foundational architecture definition for the model family our theorem is about; also the baseline against which `xin2024ralle` and `song2024ellav` report their robustness gains, letting us quote comparable WER numbers for repetition/omission-driven errors.

### Chen, Niu, Ma, Deng, Wang, Zhao, Yu, Chen — "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" (ACL 2025, arXiv:2410.06885) `chen2024f5tts`
A fully non-autoregressive flow-matching/DiT TTS model with no duration model, external aligner, or autoregressive decoding step — text is simply padded to target length and denoised in parallel.
**Relevance**: cited as the non-AR contrast case. Because F5-TTS has no token-by-token AR state map, our τ-periodic-contraction theorem (which is specifically about the AR decoder's step-to-step state update) does not apply to it — used to scope the theorem's applicability and to explain why non-AR systems are comparatively immune to counting/repetition collapse.

### Kim, Kim, Kong, Yoon — "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search" (NeurIPS 2020, arXiv:2005.11129) `kim2020glowtts`
Learns a hard monotonic text–speech alignment via dynamic programming inside a flow-based (non-AR, non-attention) model, removing the free-form soft-attention alignment that AR TTS models rely on and thereby generalizing robustly to long utterances.
**Relevance**: cited as the canonical "monotonic alignment prior" alternative to soft cross-attention; used to frame why architectures that hard-constrain the text-conditioning map to be monotonic and bounded sidestep the periodic-conditioning-plus-unconstrained-attention setup our theorem targets.

### Badlani, Łancucki, Shih, Valle, Ping, Catanzaro — "One TTS Alignment To Rule Them All" (ICASSP 2022, arXiv:2108.10447) `badlani2022onetts`
A generic alignment-learning framework (forward-sum + Viterbi + static prior) that retrofits robust monotonic alignment onto both AR (Tacotron 2, Flowtron) and non-AR (FastPitch, FastSpeech 2, RAD-TTS) models, explicitly motivated by AR attention's tendency to miss/repeat words on long or out-of-domain text.
**Relevance**: cited alongside `kim2020glowtts` as evidence that the field has long known soft cross-attention alignment is the proximate cause of repetition/omission errors; strengthens our motivation for treating attention dilution (Lemma) as the mechanism that produces effectively periodic conditioning.

---

## 2. The model panel's own papers

### Casanova, Davis, Gölge, Göknar, Gulea, Hart, Aljafari, Meyer, Morais, Olayemi, Weber — "XTTS: A Massively Multilingual Zero-Shot Text-to-Speech Model" (Interspeech 2024, pp. 4978–4982, arXiv:2406.04904) `casanova2024xtts`
Builds on the Tortoise architecture to add multilingual (16-language) zero-shot voice cloning with faster training/inference, achieving SOTA in most trained languages. XTTS-v2 is the publicly released checkpoint used in our panel.
**Relevance**: primary citation for the XTTS-v2 model evaluated in our experiments; architecture description (GPT-style AR decoder over discrete audio tokens + text conditioning) is what our theorem's assumptions (τ-periodic conditioning, τ-step Lipschitz state map) are checked against for this model.

### Ye, Zhu, Chan, Wang, Tan, Lei, Peng, Liu, Jin, Dai, Lin, Chen, Du, Xue, Chen, Li, Xie, Kong, Guo, Xue — "Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis" (2025, arXiv:2502.04128) `ye2025llasa`
Aligns TTS with standard LLM scaling recipes: a single-codebook X-Codec2 speech tokenizer plus a vanilla Llama-architecture Transformer trained at 1B/3B/8B scale, with inference-time compute (verifier-guided search) improving emotional expressiveness and content accuracy.
**Relevance**: primary citation for the Llasa-1B/3B/8B models and the X-Codec2 tokenizer in our panel; the 1B/3B/8B scale sweep on an *identical* architecture is exactly the kind of "hold architecture fixed, vary scale" setting our companion Whisper paper (`viakhirev2026dispersion`) exploited, and we use it analogously here to test whether the contraction constant q (and hence N*) shifts systematically with scale.

### Hu, Zhu, He, Guo, Zhang, Wang, Guo, Jiang, Hao, Guo, Zhang, Zhang, Yang, Xu, Zhou, Lin — "Qwen3-TTS Technical Report" (2026, arXiv:2601.15621) `hu2026qwen3tts`
Describes Alibaba's Qwen3-TTS family: a dual-track LM architecture trained on 5M+ hours across 10 languages, with two speech tokenizers (25 Hz single-codebook semantic, 12.5 Hz multi-codebook ultra-low-latency) supporting streaming synthesis with 97 ms first-packet latency.
**Relevance**: primary citation for the Qwen3-TTS model in our panel; used to describe its dual-track AR architecture when stating which parts of our theorem's assumptions (single τ-step state map) require adaptation for a two-track decoder.

### Radford, Kim, Xu, Brockman, McLeavey, Sutskever — "Robust Speech Recognition via Large-Scale Weak Supervision" (ICML 2023, arXiv:2212.04356) `radford2023whisper`
Introduces Whisper, trained on 680k hours of weakly-supervised multilingual audio-transcript pairs, achieving strong zero-shot ASR generalization without dataset-specific fine-tuning.
**Relevance**: (a) the subject model of our companion paper `viakhirev2026dispersion`, establishing continuity between the two works; (b) the ASR backbone we use to compute WER/CER on synthesized speech for our own intelligibility evaluation (Area 7).

---

## 3. Repetition & degeneration in autoregressive text LMs

### Holtzman, Buys, Du, Forbes, Choi — "The Curious Case of Neural Text Degeneration" (ICLR 2020, arXiv:1904.09751) `holtzman2020curious`
Shows maximization-based decoding (greedy/beam search) on likelihood-trained LMs produces bland, repetitive, loop-stuck text, and proposes nucleus (top-p) sampling as a fix by truncating the unreliable distributional tail.
**Relevance**: cited in the Introduction as the founding observation of AR degeneration/repetition-looping in text LMs — the direct textual ancestor of the TTS-specific failure mode we formalize; used to argue the phenomenon is architecture-general (decoder-only AR generation), not a TTS-specific quirk.

### Xu, Liu, Yan, Cai, Li, Li — "Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation" (NeurIPS 2022, arXiv:2206.02369) `xu2022learning`
Empirically characterizes a *self-reinforcement effect*: once a sentence-level repeat begins under greedy decoding, its continuation probability rises with each repeated occurrence, and higher-initial-probability sentences reinforce faster; proposes DITTO to unlearn this via pseudo-repetitive training data.
**Relevance**: closest text-LM analogue to the attention-dilution mechanism in our companion Lemma — their empirically observed "probability increases with repeat count" is the token-probability signature of the same softmax-dilution-into-an-attractor mechanism we formalize for TTS decoders; cited as the strongest empirical precedent for treating repetition as a self-reinforcing dynamical process rather than a one-off sampling error.

### Fu, Lam, So, Shi — "A Theoretical Analysis of the Repetition Problem in Text Generation" (AAAI 2021, arXiv:2012.14660) `fu2021theoretical`
Proves a "high-inflow" theorem: because many context words predict the same high-probability next word, greedy/near-greedy decoding is provably drawn into repeating that word, independent of model capacity; proposes rebalanced encoding as a mitigation.
**Relevance**: the closest *prior theoretical* (not just empirical) account of repetition in AR generation; cited explicitly in Related Work as the nearest formal precedent, and used to sharpen our contribution statement — Fu et al. prove decoding-rule-level inevitability for a single repeated token under standard LM decoding, whereas we prove state-level convergence (in hidden-state space, for a Lipschitz decoder) that also yields a fixed *quantitative* detection horizon N* and a formal (Lean-checked) impossibility result for any Lipschitz counting readout, not only greedy decoding.

### Welleck, Kulikov, Roller, Dinan, Cho, Weston — "Neural Text Generation with Unlikelihood Training" (ICLR 2020, arXiv:1908.04319) `welleck2020neural`
Identifies that standard likelihood training itself (not just decoding) over-assigns probability to repeats and frequent tokens, and introduces an unlikelihood auxiliary loss that directly penalizes unwanted repeat continuations at train time.
**Relevance**: cited as evidence the repetition tendency is baked into the *training objective*, complementing our decoder-dynamics account; used in Related Work to distinguish training-time fixes (unlikelihood, DITTO) from our purely structural/architectural (Lipschitz-contraction) explanation, which applies even to a perfectly-likelihood-trained model.

### Fan, Lewis, Dauphin — "Hierarchical Neural Story Generation" (ACL 2018, arXiv:1805.04833) `fan2018hierarchical`
Introduces top-k sampling for open-ended story generation to avoid the degenerate repetitive/generic text produced by likelihood-maximizing decoding, alongside a hierarchical premise-then-story generation scheme.
**Relevance**: cited as an early (pre-Holtzman) data point establishing that truncated/stochastic decoding was already understood as necessary to avoid repetition collapse in AR text generation — background context for how long this failure mode has been known and "worked around" rather than explained.

---

## 4. Counting / algorithmic limits of transformers

### Barbero, Banino, Kapturowski, Kumaran, Araújo, Vitvitskyi, Pascanu, Veličković — "Transformers Need Glasses! Information Over-Squashing in Language Tasks" (2024, arXiv:2406.04267) `barbero2024glasses`
Proves a "representational collapse" result: distinct long input sequences can be driven to arbitrarily close final-token representations in a decoder-only Transformer (worsened by low-precision arithmetic), provably preventing the model from responding differently to them — directly implicated in counting and copying errors.
**Relevance**: the single closest prior *transformer expressivity* result to our claim. We differ in mechanism (they show representational collapse from over-squashing information across the *whole* sequence into one token; we show a τ-periodic-conditioning-induced *contraction of the τ-step decoder map itself*, giving a quantitative geometric-convergence rate and a horizon N*) and in target (they analyze general last-token representations for arbitrary sequences; we analyze a specific periodic-repetition structure in TTS decoding). Cited prominently as the closest theoretical precedent for "no readout can count."

### Vasudeva, Fu, Zhou, Kau, Huang, Sharan — "Transformers Learn Low Sensitivity Functions: Investigations and Implications" (ICLR 2025, arXiv:2403.06925) `vasudeva2024low`
Shows transformers have an inductive bias toward *low input-sensitivity* functions compared to CNNs/MLPs/LSTMs, linking this to robustness, flat minima, and grokking.
**Relevance**: cited as complementary inductive-bias evidence — a system biased toward low sensitivity to input perturbations is structurally predisposed to exactly the kind of contraction (small output changes under repeated/near-identical inputs) our theorem assumes and exploits; supports plausibility of the Lipschitz-contraction assumption on the τ-step state map.

### Hahn — "Theoretical Limitations of Self-Attention in Neural Sequence Models" (TACL 2020, vol. 8, pp. 156–171, arXiv:1906.06755) `hahn2020theoretical`
Proves self-attention (hard or soft) cannot in general model periodic finite-state languages or hierarchical structure without the number of layers/heads scaling with input length.
**Relevance**: directly relevant precedent since our setting is literally *periodic* text conditioning; Hahn's result that self-attention cannot represent periodicity without unbounded resources is cited as independent theoretical support for why a fixed-size, fixed-depth AR-TTS decoder cannot track period count indefinitely — consistent with (and arguably predicting) our contraction/collapse result.

### Delétang, Ruoss, Grau-Moya, Genewein, Wenliang, Catt, Cundy, Hutter, Legg, Veness, Ortega — "Neural Networks and the Chomsky Hierarchy" (ICLR 2023, arXiv:2207.02098) `deletang2023chomsky`
Large-scale empirical study (20,910 models, 15 tasks) showing Transformers and RNNs fail to generalize out-of-distribution on non-regular formal-language tasks, including counting-type counter languages, unless equipped with structured external memory.
**Relevance**: cited as broad empirical corroboration, across architectures, that plain Transformers (no external memory/stack) are poor at length-generalizing counting-type tasks — situates our specific TTS-repetition-counting failure within a well-documented general limitation rather than treating it as TTS-specific.

### Yehudai, Kaplan, Dar, Rassin, Ghandeharioun, Geva, Globerson — "When Can Transformers Count to n?" (2024, arXiv:2407.15160) `yehudai2024when`
Proves a sharp phase transition in transformer counting ability governed by the ratio of embedding dimension to vocabulary size: below the threshold, non-orthogonal token interference forces exact counting to become numerically unstable, matched by controlled experiments.
**Relevance**: cited as a directly analogous "sharp threshold" result for counting in transformers — its embedding-dimension/vocabulary mechanism is a different source of counting failure than our contraction/attractor mechanism, letting us position our N* as a second, independent bound (arising from decoder dynamics rather than embedding capacity) on how far a transformer can count.

### Zhang, Cao, You — "Counting Ability of Large Language Models and Impact of Tokenization" (2024, arXiv:2410.19730) `zhang2024counting`
Argues transformers' constant-depth computation constrains exact counting, and shows tokenization choice (byte-level vs. character-level) further degrades practical counting ability in LLMs.
**Relevance**: cited as supporting evidence that counting failures are compounded by representational/tokenization choices, relevant background for why our TTS panel's failures (text tokenized into phonemes/BPE units, not counted symbolically) are consistent with, not contrary to, known LLM counting limitations.

### Sälzer, Köcher, Kozachinskiy, Zetzsche, Lin — "The Counting Power of Transformers" (2025, arXiv:2505.11199) `salzer2025counting`
Formal-language-theoretic analysis showing transformers can express certain nonlinear counting properties (polynomial combinations of counts) beyond prior linear-counting results, with new undecidability connections for restricted transformer variants.
**Relevance**: cited to give a balanced picture — transformers are not *uniformly* incapable of counting-related computation in principle; we use this to sharpen our claim's scope: our impossibility result is specifically about counting *past a repetition-induced convergence horizon N\**, not a blanket claim that transformers cannot count anything.

---

## 5. Rank collapse / token uniformity / contraction & fixed points in transformers

### Dong, Cordonnier, Loukas — "Attention is not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth" (ICML 2021, arXiv:2103.03404) `dong2021attention`
Proves that self-attention-only networks (no skip connections/MLPs) converge doubly-exponentially in depth to a rank-1 ("token uniformity") output; shows skip connections and MLPs are necessary to prevent this degeneration.
**Relevance**: foundational precedent for treating transformer depth/iteration as a contraction toward a low-dimensional attractor; cited as the layer-wise (depth-indexed) analogue of our token-position-indexed (τ-step) contraction — we adapt the "attention drives representations together" mechanism from the depth axis to the repeated-token/generation-step axis.

### Geshkovski, Letrouit, Polyanskiy, Rigollet — "A Mathematical Perspective on Transformers" (2023, arXiv:2312.10794; to appear, Bulletin of the AMS, 2025) `geshkovski2023mathematical`
Surveys the interacting-particle-system view of Transformers, where tokens evolve under attention-driven mean-field ODE dynamics and can be analyzed with tools from dynamical systems and PDEs.
**Relevance**: cited for the general mathematical framing (transformer layers as an iterated/continuous-time dynamical system with an associated notion of convergence) that our τ-step contraction argument specializes and applies concretely to the AR-TTS repetition setting.

### Geshkovski, Letrouit, Polyanskiy, Rigollet — "The Emergence of Clusters in Self-Attention Dynamics" (NeurIPS 2023, arXiv:2305.05465) `geshkovski2023emergence`
Models Transformers as interacting particle systems and proves that, as depth (continuous time) → ∞, token representations cluster around a small number of limiting "leader" points, with cluster geometry set by the spectrum of the value matrix.
**Relevance**: closest prior *fixed-point/clustering* result for self-attention; cited as direct mathematical precedent for the idea that iterated self-attention dynamics have provable attracting fixed points — our contribution specializes this from unconditional depth-wise clustering to *conditionally periodic* (repetition-driven) convergence at a fixed depth, across generation steps rather than layers, and connects it to a downstream *readout impossibility* (counting) rather than only geometric clustering.

### Bai, Kolter, Koltun — "Deep Equilibrium Models" (NeurIPS 2019, arXiv:1909.01377) `bai2019deep`
Introduces DEQ: rather than stacking finitely many layers, directly solve for the fixed point of a weight-tied layer via root-finding (Broyden/Newton), equivalent to an infinite-depth network with constant memory; demonstrated on transformer/attention-based sequence models.
**Relevance**: the explicit "fixed point of a contraction as the model's actual computation" ancestor referenced in the prompt; cited as the methodological origin of treating a repeated/iterated network map's fixed point as the object of interest — our theorem is, in effect, an *involuntary* DEQ-like collapse (the model was not designed to compute a fixed point, but periodic conditioning forces its τ-step map into contraction regardless).

### Chen, Luo — "From Condensation to Rank Collapse: A Two-Stage Analysis of Transformer Training Dynamics" (NeurIPS 2025, Oral, arXiv:2510.06954) `chen2025condensation`
Analyzes linearized Transformer training dynamics under gradient flow, showing a two-stage process where query/key matrices first "condense" (align) under asymmetric perturbations, then drive the normalized attention matrices toward asymptotic rank collapse.
**Relevance**: cited as the most recent rank-collapse result and as a *training-dynamics* (not inference-dynamics) account of collapse in transformers — used to explicitly distinguish our claim (an inference-time, per-sequence contraction under periodic conditioning, present in a fixed trained model) from theirs (a training-time phenomenon in parameter space), clarifying that "rank collapse" language has multiple distinct formal meanings in the literature that must not be conflated.

### Gao, Yang, Chen — "Clustered Attractor Manifolds and Dynamical Condensation in Self-Attention" (2026, arXiv:2608.08922) `gao2026clustered`
A statistical-mechanics analysis of normalized self-attention as a feedback dynamical system, identifying an "overlap gap" order parameter that governs a phase transition ("attention condensation") from unstructured Gaussian states to a high-dimensional manifold of clustered fixed points as attention sharpness crosses a critical threshold.
**Relevance**: **flagged as the closest related theoretical work found in this search — see `gaps.md` for a full discussion of overlap and differences.** Briefly: both papers treat self-attention as a dynamical system converging to clustered/attracting fixed points, but Gao et al. study a general statistical-mechanics phase transition in attention sharpness (no text conditioning, no AR generation, no counting/readout consequence, no formal verification), while we study a specific, formally verified, *τ-periodic-text-conditioning-induced* contraction of the AR decoder's τ-step map, with an explicit consequence (Lipschitz counting-readout impossibility past N*) validated on real TTS systems. This paper appeared on arXiv on 2026-08-09, two days before this search — cited for completeness and intellectual honesty, not because it anticipates our TTS/counting result.

---

## 6. Formal verification / Lean in ML

### The mathlib Community — "The Lean Mathematical Library" (CPP 2020, DOI 10.1145/3372885.3373824, arXiv:1910.09336) `mathlib2020lean`
Describes mathlib, the community-maintained Lean formalized-mathematics library (dependently-typed foundations, large-scale automation, now >2M lines) that our Lean 4 + mathlib proof of the attractor/no-counting theorem builds on.
**Relevance**: cited as the library dependency for our machine-checked proof; establishes mathlib's scale/maturity as evidence that a real analysis-flavored theorem (Lipschitz maps, geometric convergence, fixed points — all mathlib primitives) is a reasonable target for full formalization, not a toy example.

### de Moura, Ullrich — "The Lean 4 Theorem Prover and Programming Language" (CADE 28, LNCS 12699, pp. 625–635, 2021) `demoura2021lean4`
Describes Lean 4's design: full extensibility (custom tactics/elaborators), a hygienic macro system, and unification of the proof language with a general-purpose programming language. No arXiv preprint was found for this paper; it is cited via its Springer CADE proceedings DOI.
**Relevance**: cited as the tool citation for the proof assistant (Lean 4) used to machine-check our Theorem and Lemma.

### George, Cruden, Adkisson, Zhong, Zhang, Anandkumar — "TorchLean: Formalizing Neural Networks in Lean" (2026, arXiv:2602.22631) `george2026torchlean`
A Lean 4 framework giving neural networks (including attention/sequence-model layers) a single precise semantics shared between execution and verification, with explicit float32 arithmetic and CROWN/LiRPA-style bound-propagation verification.
**Relevance**: cited as a rare, very recent (Feb 2026) example of formally verified *neural network* claims in Lean, supporting our honest framing in `gaps.md` that formally-verified ML theory is still uncommon — most prior work verifies bounds/robustness certificates on trained weights, not architecture-level dynamical theorems like ours.

### Cipollina, Karatarakis, Wiedijk — "Formalized Hopfield Networks and Boltzmann Machines" (2025, arXiv:2512.07766) `cipollina2025formalized`
Formalizes Hopfield network convergence and Hebbian-learning correctness, and Boltzmann-machine ergodicity/convergence to a unique stationary distribution (via a new Lean formalization of the Perron–Frobenius theorem), in Lean 4.
**Relevance**: cited as the closest prior example of a *formally verified convergence/fixed-point theorem for a neural architecture* — methodologically the nearest analogue to our own machine-checked convergence proof, though for a classical (Hopfield/Boltzmann) rather than Transformer architecture; strengthens our "formally verified ML theory is rare, and rarer still for Transformers" gap claim.

---

## 7. Evaluation of TTS intelligibility / ASR-based objective metrics

### Saeki, Xin, Nakata, Koriyama, Takamichi, Saruwatari — "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022" (Interspeech 2022, arXiv:2204.02152) `saeki2022utmos`
An SSL-feature-based ensemble MOS predictor that won the VoiceMOS Challenge 2022 main and out-of-domain tracks, now widely used as a reference-free automatic naturalness metric for TTS.
**Relevance**: cited as the canonical automatic MOS proxy we use (or could use) to report naturalness alongside WER/CER in our evaluation, standard practice in recent TTS papers cited elsewhere in this survey (e.g. `du2024cosyvoice2`, `peng2025voicestar`).

### Anastassiou, Chen, Chen, Chen, et al. — "Seed-TTS: A Family of High-Quality Versatile Speech Generation Models" (2024, arXiv:2406.02430) `anastassiou2024seedtts`
ByteDance's large-scale AR-TTS technical report; establishes WER (via ASR transcription) and SIM (speaker-embedding cosine similarity) as its primary objective evaluation pair, alongside human MOS, and introduces the widely-used "seed-tts-eval" hard test set (including repetitive/tongue-twister-style prompts).
**Relevance**: cited as the standard-practice reference for our own WER/CER-via-ASR and speaker-similarity evaluation protocol; its hard test set (which includes repetition-heavy prompts) is methodologically close to our stress-test text construction and is a natural evaluation-protocol citation.

### Casanova, Weber, Shulby, Junior, Gölge, Ponti — "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone" (ICML 2022, arXiv:2112.02418) `casanova2022yourtts`
Introduces/popularizes SECS (Speaker Embedding Cosine Similarity, computed via the Resemblyzer speaker encoder) as the standard objective speaker-similarity metric for zero-shot TTS, alongside WER for intelligibility.
**Relevance**: cited as the canonical source for the SECS speaker-similarity metric used in our evaluation; also connects back to the model panel via shared first author with `casanova2024xtts`.

*(Whisper, `radford2023whisper`, doubles as our Area 7 citation for the ASR model used to compute WER/CER on synthesized speech — see Area 2.)*
