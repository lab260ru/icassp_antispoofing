# Claims

## C01: H9 same-item ranking is supported in its frozen setting

- **Statement**: On the sealed ODSS/Res2TCNGuard setup, the four-seed same-item
  ranking ensemble has lower two-target macro EER than same-pool BCE and the
  equal-edge random-pair ranking control.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: A replay that breaks ledger, prediction-order,
  metric, bootstrap, or gate reconstruction; or a result contrary to the
  frozen terminal artifact.
- **Proof**: `experiments/h9_paired_counterfactual/results/H9_TERMINAL_EVALUATION_001.md`;
  `experiments/h9_paired_counterfactual/reviews/H9_TERMINAL_RED_TEAM_AUDIT_20260811.md`.
- **Dependencies**: []
- **Tags**: anti-spoofing, same-item, controlled-transfer, frozen-terminal

## C02: H10 does not pass its fixed practical-effect replication gate

- **Statement**: The frozen H9 panel’s CD-ADD P-versus-B1 relative EER reduction
  is 8.30%, below H10’s precommitted 10% requirement.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: A hash-consistent replay of the sealed H10
  artifacts that produces a different decision object.
- **Proof**: `experiments/future_directions/results/H10_CDADD_TERMINAL_EVALUATION_001.md`;
  `experiments/future_directions/reviews/H10_CDADD_TERMINAL_RED_TEAM_AUDIT_20260811.md`.
- **Dependencies**: [C01]
- **Tags**: anti-spoofing, replication, stop-rule, no-retune

## C03: Counterpart specificity is an untested, better-attributed hypothesis

- **Statement**: Same-item pairing should be tested against a same-voice,
  different-item control before attributing H9’s gain to the documented item
  relation.
- **Status**: hypothesis
- **Provenance**: ai-suggested
- **Falsification criteria**: Same-item fails to beat the matched voice,
  random-pair, or BCE comparator under a committed H11 protocol.
- **Proof**: `experiments/future_directions/H11_IDEA_AUDIT_20260811.md`.
- **Dependencies**: [C01, C02]
- **Tags**: anti-spoofing, pairwise-training, attribution, prospective
