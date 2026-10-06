# Architecture

> 🌐 Language: **English** · [Tiếng Việt](ARCHITECTURE.vi.md)

This repository contains the source-only JAX/Keras 3 runtime used to execute Mage-Flow-Edit-Turbo on TPU v5e-8. Large JAX/Orbax model artifacts are supplied separately through `--artifact-root`.

## Pipeline

1. Qwen3-VL multimodal conditioning encodes the edit instruction and reference image.
2. The VAE encodes the reference image to a latent.
3. The Edit transformer packs target and reference tokens per replica and predicts only the target update.
4. The VAE decodes the final target latent to RGB PNG.

## TPU topology

The qualified topology is Kaggle TPU v5e-8 with eight devices arranged as a `4x2` replica/model topology. Transformer parameters use the recorded 397-leaf binding with 174 sharded and 223 replicated leaves.

## Qualification boundary

- 512×512 / q256 / 4 steps: production-qualified baseline.
- 768×768 / q256: validated scaling.
- 1024×1024 / q128: experimental high-resolution mode.

`q128` and `q256` are deterministic within their tested configurations but are not bit-identical to each other at 512px.
