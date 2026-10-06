# Mage-Flow-Edit-Turbo trên TPU v5e-8 — v1.0.0

> 🌐 Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) · **Tiếng Việt**

Public release đầu tiên đóng gói source-only runtime JAX/Keras 3, TPU orchestration, CPU regression suite, machine-readable qualification evidence, public production notebook và standalone model-artifact links cho Mage-Flow-Edit-Turbo trên TPU v5e-8.

## Qualification

- **512×512 / q256 / 4 steps** — production-qualified baseline; 3/3 edit case acceptance, seed 42–45, mỗi case bốn PNG khác nhau.
- **768×768 / q256** — validated scaling; repeat bốn seed tái lập bit-for-bit 4/4 PNG hash.
- **1024×1024 / q128** — high-resolution experimental; nhiều edit subject chạy q128 deterministic thành công.
- q128 và q256 không bit-identical ở 512 và được qualification riêng.

## Showcase đã duyệt

Snow Leopard winter (768/q256 seed 52), Portrait golden hour (768/q256 seed 43), Floating Island winter (768/q256 seed 43), và Monorail deep red (768/q256 seed 43).

## Public surfaces

- [Hugging Face model](https://huggingface.co/dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8)
- [Kaggle model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8)
- [Kaggle Public Acceptance notebook](https://www.kaggle.com/code/dangkhoa2016/mage-flow-edit-turbo-tpu-v5e8-public-acceptance)
- [GitHub release](https://github.com/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8/releases/tag/v1.0.0)

## Release asset

`Mage-Flow-Edit-Turbo-TPU-v5e8-publish-showcase.tar.gz` — 14,135,561 byte — SHA256 `4c9b03affc868050c4055127b4a88360a934ae3e1c690dc9443e15d975918c4d`.

Xem [`docs/RELEASE-EVIDENCE-v1.0.0.vi.md`](docs/RELEASE-EVIDENCE-v1.0.0.vi.md) để biết evidence map và licensing boundary.
