# H2 score-independent pair and quality gate

`src/h2_pre_score_pairs.py` separates all H2 input selection and waveform
quality work from detector inference. It is deliberately unable to import a
detector, receive a scorer callback, or emit a detector-response column.

## Pre-score contract

`scripts/freeze_h2_pre_score_manifest.py` reads one explicit CSV containing
only `dataset`, `sample_id`, `label`, and optional source metadata. It rejects
columns whose normalized name suggests a score, logit, probability, or
prediction. For every dataset--label stratum, it ranks rows using the stable
SHA-256 key of `(seed, dataset, label, sample_id)` and freezes exactly the
requested count. The generated input CSV, metadata JSON, and arm ledger all
refuse overwrites.

The frozen crest-factor ledger contains exactly four arms:

| Arm | Role | Target condition |
|---|---|---|
| `drc_cf3` | intervention | crest factor decreases |
| `drc_cf6` | intervention | crest factor decreases |
| `small_gain_plus_0p1db` | negative control | crest factor invariant within 1e-6 dB |
| `polarity` | negative control | crest factor invariant within 1e-6 dB |

This reflects the committed H1 discovery freeze; it is not a data-dependent
arm ranking. Silence, spectral-tilt, and all-pass implementations remain
available for a separately frozen hypothesis only.

## Quality contract

`evaluate_quality_pair` records, for every original/arm pair, both waveform
hashes, transform diagnostics, all 28 feature deltas, STOI, original and
transformed transcripts, normalized WER, LUFS, clipping fractions, each gate,
and an explicit failure-reason list. A pair is retained only if all of the
following pass:

- STOI >= 0.95;
- original-to-transformed normalized WER <= 0.05;
- absolute loudness delta <= 0.2 LU;
- original and transformed clipping fractions are zero;
- the arm's registered crest-factor condition holds.

An unavailable metric is a failure, never a silently omitted field. The quality
row remains `blocked_pending_quality_freeze`; a separate scorer may consume
only an immutable table of retained rows after the arm-level 90% pass-rate rule
is evaluated. No actual pair, ASR transcription, detector response, or H2
causal claim is contained in this implementation artifact.

## Focused verification

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_h2_pre_score_pairs.py \
  tests/test_h2_waveform_transforms.py \
  tests/test_h2_asr_wer.py
```
