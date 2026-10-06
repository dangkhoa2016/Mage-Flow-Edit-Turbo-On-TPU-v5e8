# Contributing

> 🌐 Language: **English** · [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for contributing to Mage-Flow-Edit-Turbo-On-TPU-v5e8.

## Scope

This repository maintains the JAX/Keras TPU runtime, production orchestration, tests, acceptance metadata, and documentation for the converted Mage-Flow-Edit-Turbo inference stack.

Please keep changes within that engineering scope. Changes to upstream model weights or claims about upstream training are outside this repository's authority.

## Before opening a pull request

Run:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q runtime bootstrap tests scripts
KERAS_BACKEND=jax USE_TORCH_XLA=0 pytest -q tests
python scripts/check_repo_health.py
git diff --check
```

## Runtime-semantic changes

Changes to attention, sharding, checkpoint mapping, BF16 timestep semantics, denoising steps, or TPU placement require fresh TPU qualification before they can be described as equivalent to the v1.0.0 baseline.

## Documentation

Public documentation is maintained in English and Vietnamese. New files under `docs/` should normally be added as an EN/VI pair with language switches.

## Licensing

Do not remove upstream copyright, attribution, license, or notice information. See [`MODEL_LICENSE.md`](../MODEL_LICENSE.md) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Secrets and private data

Never commit API tokens, credentials, private URLs, private datasets, SSH keys, or runtime secrets.

## Pull requests

Keep each pull request focused. Explain the behavior changed, validation performed, and whether TPU re-qualification is required.
