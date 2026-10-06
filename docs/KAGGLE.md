# Kaggle TPU Guide

> 🌐 Language: **English** · [Tiếng Việt](KAGGLE.vi.md)

The reference environment is Kaggle TPU v5e-8. Use the public standalone Kaggle model artifact and the production notebook. The notebook is designed around one Edit-Turbo artifact source and does not require a separate Turbo-JAX model.

Keep model artifacts outside Git, point `--artifact-root` to the mounted model directory, and write generated outputs under `/kaggle/working`.

Before a public run, verify TPU visibility, JAX/Keras versions, artifact directories, and available disk space. The repository CPU suite validates wiring and contracts; it does not replace TPU qualification.
