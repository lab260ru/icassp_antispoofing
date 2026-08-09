# H2B Q1 run 001 — DeepVoice detector-free quality calibration

**Status:** completed quality-only feasibility calibration. No detector was
imported, loaded, or run; this is not a feature-selection, score, EER, paired
score-delta, or causal result.

## Scope locked before execution

Q1 consumed the score-blind 256-identity DeepVoice Q0 manifest (128 examples
per label), pinned to `SpeechAntiSpoofingBenchmarks/DeepVoice` revision
`cc3bdf544cfc09bd9cc788f7f022ba1af9daf701`. The committed Q1 arm manifest
defined nine arms: two all-pass variants, four signed spectral tilts, fixed
length endpoint zeroing, and gain/polarity controls. It also fixed a 95%
Wilson lower retention-bound threshold of 0.90, family-specific minimum target
changes, and the lexicographic tie-breaking rule.

The detector-free run used Whisper on physical GPU 3 (logical `cuda:0`) and
wrote 2,304 immutable pair checkpoints (256 identities × 9 arms) on the HDD.
The final table has 2,304 rows, and the selection utility independently
validated its pair IDs before reporting the result below.

## Frozen Q1 outcome

No non-control transform family cleared the precommitted Wilson-bound and
target-change conditions, so the selector returns **no selected arms**. In
particular, the closest arm, `tilt_minus_0p5db_oct`, retained 239/256 pairs
(93.36%) but has a 95% Wilson lower bound of 0.89624, below the locked 0.90
threshold. The endpoint arm retained 15/256 pairs (5.86%). All-pass arms had
lower bounds of 0.83764 and 0.84646. The controls are reported in the quality
summary only and are not selectable.

| Family | Arm | Retained pairs | 95% Wilson lower bound | Median absolute target change | Eligible |
|---|---|---:|---:|---:|---|
| all-pass phase | `allpass_abs_a005` | 226/256 | 0.83764 | 0.000000246 | no |
| all-pass phase | `allpass_abs_a010` | 228/256 | 0.84646 | 0.000000246 | no |
| endpoint fixed-length | `endpoint_zero_fixed_length` | 15/256 | 0.03583 | 0.00000 | no |
| negative spectral tilt | `tilt_minus_0p5db_oct` | 239/256 | 0.89624 | 0.26275 | no |
| negative spectral tilt | `tilt_minus_1p0db_oct` | 224/256 | 0.82886 | 0.52552 | no |
| positive spectral tilt | `tilt_plus_0p5db_oct` | 236/256 | 0.88242 | 0.26275 | no |
| positive spectral tilt | `tilt_plus_1p0db_oct` | 225/256 | 0.83324 | 0.52549 | no |

This is a useful negative feasibility result. Thresholds, transform
parameters, and the source cohort will not be changed post hoc. Q2, Q3, and
Q4 are consequently not authorized by this loop; in particular, no H2B audio
may be sent to a detector as a consequence of this run.

## Artifact ledger

| Artifact | Location | SHA-256 |
|---|---|---|
| Quality pairs | HDD `runs/h2b_quality_calibration/h2b_q1_deepvoice_001/quality_pairs.parquet` | `993acf50cf64352ee0ec34fa99ad032e2a8e247a15be63fd831fead7b69ead9c` |
| Quality provenance | HDD `runs/h2b_quality_calibration/h2b_q1_deepvoice_001/quality_provenance.json` | `337646ddaac1a8713759f0816dff092d8c362a84a2caa14f0665bbb0a85ec58c` |
| Quality summary | HDD `runs/h2b_quality_calibration/h2b_q1_deepvoice_001/quality_summary.json` | `96dda60b07b207237fd2b9b9181363d150c53c1e7201a65e1f78dc7bad0a985f` |
| H2B Q1 summary | `results/q1_quality_runs/h2b_q1_deepvoice_001.summary.json` | `4819d4303ef0e600f28fc0414508629d2f6f57ae4e9e85b4526db5c959c0d668` |
| Selection table | `results/h2b_q1_deepvoice_selection/quality_family_selection.csv` | `4c4f16072eddfb97e9b5a9709f83534752e7e45e18af0f92a80870a6595c109f` |
| Selection report | `results/h2b_q1_deepvoice_selection/quality_family_selection.json` | `bf157bf481640b7d42350ac81d8344d10497f900e6ed231a564c27d329481953` |

The Q1 summary records `detector_scoring_allowed: false`, `complete: true`,
and `remaining_pairs: 0`. The selection report repeats that detector scoring
remains prohibited and has an empty `selected_arms` list.

## Operational note

During a status diagnostic, a second resumptive process was briefly started
and immediately terminated. The original process remained the active writer.
The final run reports 2,304 newly written checkpoints; the final table has
2,304 unique validated pair IDs. The later lock implementation prevents this
class of duplicate invocation for future Q1 runs. No detector or score path
was involved in the incident.
