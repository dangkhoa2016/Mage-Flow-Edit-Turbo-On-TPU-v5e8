# Artifact Contract

> 🌐 Language: **English** · [Tiếng Việt](ARTIFACTS.vi.md)

The public Edit-Turbo JAX artifact is standalone for this repository's inference path. It carries the text encoder checkpoint, Edit transformer checkpoint, VAE checkpoint, manifests, RoPE captures and the model-side metadata required by the runner.

Public artifact surfaces:

- Hugging Face: `dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8`
- Kaggle Models: `dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8`

The Git repository intentionally excludes model/checkpoint blobs. Do not commit `.safetensors`, Orbax checkpoint trees, tar archives or other weight payloads.
