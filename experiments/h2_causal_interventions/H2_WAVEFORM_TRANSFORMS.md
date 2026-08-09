# H2 waveform-transform implementation

`src/h2_waveform_transforms.py` implements the currently scoped waveform-only
H2 arms: global gain, polarity inversion, energy-VAD endpoint-silence
standardization, crest-factor-targeted deterministic DRC, spectral tilt, and
stable first-order all-pass cascades. It performs no dataset access, detector
scoring, ASR, or causal inference.

Every transform takes a float32 mono waveform and sample rate, returns a
float32 waveform, and emits manifest-friendly before/after diagnostics:
duration, peak, RMS, crest factor, integrated loudness, clipping fraction, and
arm-specific parameters. The later H2 runner—not this module—must compute the
frozen v1_28 feature deltas, STOI/WER, loudness and clipping gates, and decide
which pairs are retained.

Integrated loudness is NaN for clips where EBU R128 measurement is unavailable;
the manifest writer must encode that explicitly as a failed/unavailable gate,
not as a valid numerical result.

Endpoint silence uses a deterministic energy-VAD proxy with explicitly logged
frame/hop/threshold/guard settings. It preserves its detected waveform core
exactly but may change length and detector-window position; those facts are
diagnostics, not evidence of a pure silence effect. DRC uses a block-envelope
compressor with bisection-selected threshold and optional scalar integrated
loudness restoration. It intentionally has no limiter: a clip introduced by
loudness restoration is visible to the later no-clipping quality gate.

Spectral tilt applies a positive real, clipped gain in a fixed Hann STFT:
`tilt_db_per_octave * log2(frequency / reference_hz)`. The default reference
is 1 kHz, the low-frequency floor is 125 Hz, the gain cap is 24 dB, and the
fixed 1024/256 analysis preserves phase bins and never resamples. The transform
allows only finite tilt values from -12 to +12 dB/octave; H2 should use its
locked -3 and +3 arms. It records the requested values and an independent
global FFT slope estimate before/after, while the later runner must still
measure the frozen v1_28 spectral features and quality gates.

Focused tests live in `tests/test_h2_waveform_transforms.py`. They validate
float32/shape invariants, endpoint core preservation, DRC crest reduction,
requested-direction spectral tilt, all-pass stability, and invalid-input
rejection. Passing these tests only establishes DSP behavior; it does not
establish speech preservation or any H2 causal claim.
