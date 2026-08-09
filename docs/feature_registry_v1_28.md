# Feature registry v1_28

The confirmatory H1 registry contains exactly 28 scalar features for each view:
the full 16-kHz mono waveform, a deterministic 64,600-sample center crop with
repeat padding, and a 0.97-pre-emphasized version of that crop.

The amplitude/envelope group is crest factor, peak/RMS level, integrated
loudness, silence, clipping, and frame-RMS dynamics. The spectral group is
centroid, bandwidth, roll-off, flatness, slope, low/high-band ratio, and flux.
The excitation group uses `librosa.pyin` with 50–500 Hz limits and defines HNR,
CPP, jitter, and shimmer as documented frame-level autocorrelation/cepstral
proxies; they are not substituted for Praat measurements. The final group
measures phase dispersion and modulation-spectrum shape.

All numerical implementation details and ordered names are fixed in
`src/audio_features.py`. Missing voice-quality values remain missing rather
than being zero-imputed; downstream models use training-set imputation flags.
