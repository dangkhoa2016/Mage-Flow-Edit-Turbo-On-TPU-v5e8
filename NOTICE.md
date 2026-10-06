# Notices and Upstream Attribution

> 🌐 Language: **English** · [Tiếng Việt](NOTICE.vi.md)

This repository is an independent JAX/Keras/TPU conversion-runtime engineering project for the Mage-Flow-Edit-Turbo inference stack.

It is not an official Microsoft, Qwen, Alibaba, Google, Keras, Kaggle, Hugging Face, or GitHub release.

## Mage-Flow / Mage-Flow-Edit-Turbo

- Upstream project: https://github.com/microsoft/Mage
- Upstream model family: Mage-Flow / Mage-Flow-Edit-Turbo
- Copyright (c) 2026 Microsoft
- Upstream license: MIT

A copy of the upstream MIT license is preserved in [`licenses/Mage-Flow-MIT.txt`](licenses/Mage-Flow-MIT.txt).

## Qwen3-VL-derived Text Encoder lineage

- Upstream model family: Qwen3-VL
- Referenced component: Qwen3-VL-4B-Instruct lineage used by the Mage-Flow inference stack
- Upstream license: Apache License 2.0

A copy of Apache License 2.0 from the upstream Qwen3-VL project is preserved in [`licenses/Qwen3-VL-Apache-2.0.txt`](licenses/Qwen3-VL-Apache-2.0.txt).

## Conversion and TPU engineering

The JAX/Orbax conversion, parameter mapping, checkpoint layout conversion, TPU sharding, runtime integration, packaging, documentation, and TPU v5e-8 qualification were performed by `dangkhoa2016`.

These contributions do not replace, supersede, or relicense upstream model weights or upstream third-party components.

## License boundary

The repository-level [`LICENSE`](LICENSE) applies to original repository engineering contributions for which the repository author can grant MIT rights.

For model-related licensing boundaries, see [`MODEL_LICENSE.md`](MODEL_LICENSE.md). For additional third-party notices, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Trademarks

All upstream project, company, product, and service names are used for attribution and technical identification only. Their trademarks remain the property of their respective owners.

## Visual asset provenance

Visual showcase assets have a separate provenance boundary documented in [`ASSET_LICENSE.md`](ASSET_LICENSE.md). Where the canonical manifest does not record source-image provenance, this project does not infer or claim source-image ownership.
