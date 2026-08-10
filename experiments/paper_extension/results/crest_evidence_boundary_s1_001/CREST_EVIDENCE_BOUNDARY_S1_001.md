# Supplementary S1 result — crest-factor evidence boundary

**Status:** complete deterministic supplementary visualization. This is not a
new experiment, test, bootstrap, pooled estimate, ranking, or cross-panel
inference. It cannot change H1, H2, H2B, H3, or H4 gates.

## Locked execution

The renderer was run after validating the fixed manifest
`../../crest_evidence_boundary_input_manifest.json` against its fixed source
paths and SHA-256 pins:

```bash
PYTHONPATH=. python3 scripts/render_crest_evidence_boundary.py
PYTHONPATH=. python3 -m pytest -q tests/test_crest_evidence_boundary.py
```

It accepts no run-time source path. The implementation verifies the protocol
hash; the exact nine-source manifest; source paths, kinds, and SHA-256 values;
the fixed corpus order; the fixed Spectra-AASIST/full-waveform/spoof/crest
slice; and the locked H2 90% gate. It rejects a redirected or altered manifest
input, forbidden score/model/audio/ASR/H2B paths, response-like CSV columns,
and malformed fixed-slice rows before drawing.

## Rendered quantities (pre-existing sealed values only)

| Panel | Quantity | Values in fixed corpus order |
|---|---|---|
| A | H1 partial Spearman $\rho$ | -0.453885, -0.099143, -0.064950, -0.032606, +0.001383 |
| A | Held-out 95% clustered CIs | InTheWild [-0.070045, -0.001718]; ASVspoof5 [-0.029092, +0.032219] |
| B | H2 retained-pair fraction | DRC-3 18.1%; DRC-6 0.6%; +0.1 dB gain 64.6%; polarity 99.9% (locked arm gate 90%) |
| C | H4 signed label AUROC | -0.492761, -0.459334, -0.261936, +0.342773, -0.019475 |

Panel A displays the discovery-frozen H1 slice and held-out intervals, not a
pooled association. Panel B is detector-free quality screening and explicitly
states that scoring is unavailable because the panel was not frozen. Panel C
is the sealed raw label-separation diagnostic and explicitly disclaims detector
reliance and cue selection.

## Figure artifacts

| Artifact | SHA-256 |
|---|---|
| `s1_crest_evidence_boundary.pdf` (vector) | `25ed19cb988bbd037d7d20c6e946bf0d50159a84e2cb56db7e6598bc61f39045` |
| `s1_crest_evidence_boundary.png` (300 DPI) | `6dfac2a0d4f392f9c21d8c65cd4ba6ef56d9692c73d09d8f70896a2f6ba22a9b` |
| `s1_crest_evidence_boundary.metadata.json` | `b8512d28f6540ff2f5bd29dde74fd9c4747c04fe596a743226a42c74f7e9d80f` |

The metadata sidecar records the caption, all input/protocol/manifest/output
hashes, exact rendered values, fixed order, figure dimensions, and claim
boundary. The H4 source remains in its hash-sealed HDD location; no audio,
model weights, detector responses, or large runtime artifacts are copied into
this supplementary figure result.

## Caption

Supplementary Fig. S1. Evidence boundary for the fixed full-waveform,
spoof-class crest-factor slice. (A) The pre-frozen Spectra-AASIST partial
association with published spoof evidence across the five core corpora; 95%
clustered intervals are displayed only for the two sealed held-out
confirmations. (B) Detector-free H2 waveform-quality retention for the four
frozen arms against the locked 90% gate. The panel was refused before any
detector scoring. (C) Score-free signed label AUROC for full-waveform crest
factor across the same corpora. This is raw label separation, not detector
reliance. No pooled estimate, new test, ranking, or cross-panel inference is
computed.

See [visual inspection](VISUAL_INSPECTION.md) for the final readability audit.
