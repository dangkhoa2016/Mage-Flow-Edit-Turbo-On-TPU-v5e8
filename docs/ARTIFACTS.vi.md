# Artifact Contract

> 🌐 Ngôn ngữ: [English](ARTIFACTS.md) · **Tiếng Việt**

Artifact JAX Edit-Turbo công khai là standalone cho đường inference của repository này. Nó chứa text encoder checkpoint, Edit transformer checkpoint, VAE checkpoint, manifest, RoPE capture và metadata phía model mà runner cần.

Các public artifact surface:

- Hugging Face: `dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8`
- Kaggle Models: `dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8`

Git repository cố ý không chứa model/checkpoint blob. Không commit `.safetensors`, cây Orbax checkpoint, tar archive hay weight payload khác.
