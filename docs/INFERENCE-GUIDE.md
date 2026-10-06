# Inference Guide

> 🌐 Language: **English** · [Tiếng Việt](INFERENCE-GUIDE.vi.md)

## Artifact layout

Use the standalone Edit-Turbo artifact root. It must provide `text_encoder/`, `checkpoints/{text_encoder,transformer,vae}`, `manifests/`, and `rope/`. A separate Turbo-JAX model is not required.

## Production-qualified run

```bash
python bootstrap/08_run_tpu_edit.py   --image /path/reference.png   --prompt "your edit instruction"   --output /kaggle/working/edit-output   --artifact-root /path/to/mage-flow-edit-turbo-jax-tpu-v5e8   --resolution 512   --seeds 42,43,44,45
```

The production-qualified baseline is 512×512, q256, four denoising steps. Use 768/q256 as validated scaling and 1024/q128 only as the documented experimental high-resolution configuration.

## Outputs

The runner writes stage outputs plus `generation_summary.json` and final PNG files. Treat measured timings as qualification evidence, not universal latency guarantees.
