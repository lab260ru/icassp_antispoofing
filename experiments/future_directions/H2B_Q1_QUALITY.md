# H2B Q1 finite detector-free quality calibration

`h2b_q1_quality_arm_manifest.json` locks nine transforms before any DeepVoice
waveform is decoded: one fixed-length endpoint-zeroing arm, four mild spectral
tilts, two mild all-pass cascades, and polarity/+0.05-dB gain controls.  It
also locks four non-control selection families and their Wilson-bound,
target-change, and WER tie-break rule.  No feature candidate, detector, score,
or EER quantity is present in the manifest.

`scripts/run_h2b_q1_quality.py` verifies that the Q0 CSV/provenance and Q1 arm
manifest are committed and clean, validates all content hashes and source
identity, and writes immutable pair checkpoints/transcripts/full tables on
the HDD.  It deliberately labels the whole 256-row calibration as
`pilot_not_panel_gate`: regardless of quality success, it cannot authorize a
detector to score a waveform.

Run on the reserved ASR GPU only after committing the arm manifest and runner:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/run_h2b_q1_quality.py \
  --run-id h2b_q1_deepvoice_001
```

The logical Whisper device is `cuda:0` under that command.  Large checkpoint,
transcript, audio-derived and Parquet outputs remain under the HDD root.
