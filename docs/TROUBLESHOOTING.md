# Troubleshooting

> 🌐 Language: **English** · [Tiếng Việt](TROUBLESHOOTING.vi.md)

## Artifact path errors

Confirm `--artifact-root` contains `text_encoder/`, `checkpoints/`, `manifests/` and `rope/`.

## TPU/JAX startup errors

Confirm the notebook is attached to TPU v5e-8 and uses the documented TPU environment variables. Avoid importing an unintended CUDA runtime into the TPU process.

## Out-of-memory at high resolution

Return to the production-qualified 512/q256 baseline. The 1024 path is documented only with q128 and remains experimental.

## Output differs across q128/q256

This is expected: the two query-chunk settings are not bit-identical at 512. Compare repeats within the same configuration.
