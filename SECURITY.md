# Security Policy

> 🌐 Language: **English** · [Tiếng Việt](SECURITY.vi.md)
## Supported version

The current `main` branch and the published `v1.0.0` release baseline are the supported public surfaces for security reports.

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability that could expose credentials, enable remote compromise, or materially affect users.

Report security-sensitive issues privately to:

`i.am@dangkhoa.dev`

Include, when possible:

- affected commit/tag;
- affected file or component;
- reproduction steps;
- impact;
- suggested mitigation, if known.

## Secrets

The repository must not contain Hugging Face, GitHub, Kaggle, cloud, tunnel, SSH, or other access tokens/secrets.

## Model behavior

Model-output quality, bias, prompt adherence, or content-safety concerns are different from software vulnerabilities. Report those with enough context to reproduce the model behavior, but do not include sensitive personal data.

## Upstream vulnerabilities

If the issue originates in Mage-Flow, Qwen3-VL, JAX, Keras, Orbax, or another dependency, the upstream maintainer may also need to be notified.
