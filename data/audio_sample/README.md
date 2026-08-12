# Audio sample

165 clips (7.1 MB), stratified over model, item family, k band and outcome class, including the degenerate outcomes the analysis excludes.

16 kHz mono Opus --- the same sample rate and channel count the CTC judge consumes, so what is shipped is what was scored.

`manifest.csv` pairs each clip with its stimulus text, the judge's transcript, the requested and counted repetitions, the relative error and the outcome label. Listening to a clip and reading its row is enough to check a scoring decision without regenerating anything.

This is a sample for verification, not the dataset: the full 3.4 GB is regenerable from the released code and seeds.

Regenerate with `python scripts/make_audio_sample.py`.
