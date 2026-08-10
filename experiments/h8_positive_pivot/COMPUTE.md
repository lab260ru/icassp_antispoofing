# H8 compute contract

- **Hardware:** four idle NVIDIA RTX 6000 Ada GPUs, 48 GB each.
- **Runtime:** PyTorch 2.11.0 + CUDA 13.0, torchaudio 2.11.0, accelerate 1.13.0.
- **H8-SF use:** the 8-dimensional fusers are CPU-efficient; use parallel CPU
  workers for target/seed evaluation and do not waste GPUs merely to claim GPU
  use. GPU BF16 throughput probing is reserved for the predeclared SSL-probe
  fallback.
- **No cloud/download required** for H8-SF; all score panels are local on HDD.
