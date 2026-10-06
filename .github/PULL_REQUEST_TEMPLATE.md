# Pull request

> 🌐 Language: **English** · [Tiếng Việt](PULL_REQUEST_TEMPLATE.vi.md)

## Summary

Describe the focused change and why it is needed.

## Validation

- [ ] `python -m compileall -q runtime bootstrap tests scripts`
- [ ] `KERAS_BACKEND=jax USE_TORCH_XLA=0 pytest -q tests`
- [ ] `python scripts/check_repo_health.py`
- [ ] `git diff --check`

## Qualification impact

- [ ] No runtime-semantic change.
- [ ] Runtime-semantic change; fresh TPU qualification evidence is included.

## Safety

- [ ] No tokens, credentials, private URLs, model weights, or checkpoint blobs are committed.
- [ ] EN/VI public documentation remains aligned where applicable.
