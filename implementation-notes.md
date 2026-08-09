# Implementation Notes

## 2026-08-09 - ICASSP anti-spoofing autoresearch bootstrap

- Decision: Treat published SpeechAntiSpoofingBenchmarks Arena scores as authoritative inputs; analyze them but do not spend compute reproducing them.
- Decision: Use public core datasets first and reserve In-the-Wild and ASVspoof 5 as held-out confirmation data for feature selection made on the discovery datasets.
- Decision: Use one independent job per RTX 6000 Ada GPU; newly trained models use BF16 and per-backbone throughput measurement.
- Tradeoff: The current ICASSP 2027-specific kit is not yet available, so the initial manuscript will use the generic IEEE conference template and be migrated once the official kit is published.
- Constraint: Telegram, GitHub, and Hugging Face runtime credentials are not currently configured. Sensitive values are intentionally not retained here.
- Follow-up: Build the pinned Hub catalogue, validate public download access, and lock H1 feature extraction before result generation.
