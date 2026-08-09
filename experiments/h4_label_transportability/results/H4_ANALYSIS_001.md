# H4 analysis 001 — score-free label--cue transportability atlas

**Status:** complete descriptive, classifier-free analysis. It has no detector
scores, logits, models, audio decoding, ASR, EER, intervention, or causal
quantity. Its results cannot select a cue for H1, H2, H2B, H3, or training.

## Locked inputs and execution

The analyzer consumed only the committed `h4_input_freeze_001` provenance and
its revalidated selection manifest. It rechecked all five input hashes and
the H4 protocol hash before reading the allowed identity/label/view/feature
projection. The frozen panel contains 5,000 source IDs per label per corpus
and all three registered views.

The complete locked output has 420 `dataset × view × feature` cells and 84
`view × feature` aggregations. Every cell is valid: there are no failed cells,
no missing aggregation inputs, and each interval uses 500 valid
label-stratified source-cluster percentile bootstrap replicates (seed 2609).

## Result

Twenty-three of the 84 units meet the **atlas-only stable label association**
rule. This is a corpus-level descriptive property of feature--label separation,
not evidence of detector reliance. The rule requires consistent nonzero sign
in at least four corpora, matching nonzero held-out intervals, held-out
absolute signed AUROC at least 0.10, and at least 80% finite availability for
each label/corpus. The full matrix and all 84 rule components are retained;
the 23 units are not ranked or forwarded to any other hypothesis.

The planned crest-factor diagnostic reinforces the paper's cautious framing:
full-waveform crest has signed label AUROC -0.49276 (ASVspoof2019 LA),
-0.45933 (ASVspoof2021 LA), -0.26194 (ASVspoof2021 DF), +0.34277
(InTheWild), and -0.01947 (ASVspoof5). Its InTheWild sign reversal and
ASVspoof5 interval spanning zero prevent it from satisfying the H4 rule. This
does not explain or alter the separate H1 score-association result.

## Artifact ledger

| Artifact | Location | Bytes | SHA-256 |
|---|---|---:|---|
| H4 matrix | HDD `runs/h4_label_transportability/h4_analysis_001/h4_label_cue_matrix.csv` | 78,730 | `03dc8674db5739bfffa1c2f2b6e2973b31a8294ab5e15f0f7d6b8e35a669b25f` |
| H4 aggregation | HDD `runs/h4_label_transportability/h4_analysis_001/h4_label_cue_aggregation.csv` | 11,143 | `e5ba70eacf676718842d24d0c88326fd71a328a43f2036ab59c48d87914931a5` |
| H4 analysis provenance | HDD `runs/h4_label_transportability/h4_analysis_001/h4_analysis_provenance.json` | 1,055 | `41ceaa6423ad9c933d1f9955797c984fccd45c7f77aa74bca8cbe17e1e7087a3` |

The analysis provenance binds the input-freeze provenance hash
`52526eaa68d7c6d6d39c8223b4f3fc7b4302696e7d8c409ba7fa32297f66ad93`,
selection-manifest hash
`fe11423b4db7afe4c77790f0259fe9c68e587d3f61279c0139fd5509296ddfa7`,
and locked H4 configuration (500 bootstrap replicates, seed 2609).

## Operational note and stopping rule

The command wrapper returned before its child analyzer exited; the single
validated child process was observed to completion and no second process was
started. Outputs appeared only after completion. The full result is now sealed:
do not tune feature orientation, thresholds, views, sample caps, or bootstrap
count, and do not promote the 23 descriptive units into a new detector study.
