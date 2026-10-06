# Mage-Flow-Edit-Turbo on TPU v5e-8 — v1.0.0

> 🌐 Language: **English** · [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

This first public release packages the source-only JAX/Keras 3 runtime, TPU orchestration, CPU regression suite, machine-readable qualification evidence, public production notebook and standalone model-artifact links for Mage-Flow-Edit-Turbo on TPU v5e-8.

## Qualification

- **512×512 / q256 / 4 steps** — production-qualified baseline; 3/3 accepted edit cases, seeds 42–45, four unique PNGs per case.
- **768×768 / q256** — validated scaling; repeated four-seed run reproduced 4/4 PNG hashes bit-for-bit.
- **1024×1024 / q128** — experimental high resolution; successful deterministic q128 runs on multiple edit subjects.
- q128 and q256 are not bit-identical at 512 and are qualified separately.

## Approved showcase

Snow Leopard winter (768/q256 seed 52), Portrait golden hour (768/q256 seed 43), Floating Island winter (768/q256 seed 43), and Monorail deep red (768/q256 seed 43).

## Public surfaces

- [Hugging Face model](https://huggingface.co/dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8)
- [Kaggle model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8)
- [Kaggle production notebook](https://www.kaggle.com/code/dangkhoa2016/mage-flow-edit-turbo-tpu-v5e-8-production-demo)
- [GitHub release](https://github.com/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8/releases/tag/v1.0.0)

## Release asset

`Mage-Flow-Edit-Turbo-TPU-v5e8-publish-showcase.tar.gz` — 14,135,561 bytes — SHA256 `4c9b03affc868050c4055127b4a88360a934ae3e1c690dc9443e15d975918c4d`.

See [`docs/RELEASE-EVIDENCE-v1.0.0.md`](docs/RELEASE-EVIDENCE-v1.0.0.md) for the evidence map and licensing boundaries.
