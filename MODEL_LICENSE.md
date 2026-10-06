# Model Component Licensing and Attribution

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](MODEL_LICENSE.vi.md)

This file documents the licensing boundary for model-related components used by this repository and by the separately distributed JAX/Orbax model artifact.

**It is not a new license and does not replace, supersede, or relicense any upstream model, model weight, tokenizer, configuration, or third-party component.**

## Repository engineering license

Original engineering code and documentation authored for this repository are released under the repository-level [MIT License](LICENSE):

```text
Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>
```

This repository-level MIT grant applies only to material for which the repository author has the right to grant that license.

## Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo components

Upstream project: https://github.com/microsoft/Mage
Upstream model family: Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo
Upstream copyright: Copyright (c) 2026 Microsoft
Applicable upstream license: **MIT License**

A preserved copy of the upstream MIT license is included at [`licenses/Mage-Flow-MIT.txt`](licenses/Mage-Flow-MIT.txt).

The use of the names Mage, Mage-Flow, Mage-Flow-Turbo, Mage-Flow-Edit-Turbo, Microsoft, or related marks is for attribution and source identification only. No trademark rights are granted by this repository.

## Qwen3-VL-derived Text Encoder and tokenizer/configuration components

Upstream project/model family: Qwen3-VL
Upstream model referenced by the converted distribution: Qwen3-VL-4B-Instruct
Applicable upstream license: **Apache License 2.0**

A preserved copy of Apache License 2.0 from the upstream Qwen3-VL project is included at [`licenses/Qwen3-VL-Apache-2.0.txt`](licenses/Qwen3-VL-Apache-2.0.txt).

Apache-2.0 attribution, notice-retention, and other applicable conditions remain in force for those components.

## Converted JAX/Orbax model artifacts

The converted Text Encoder, Transformer, and VAE checkpoints are format/runtime conversions of upstream pretrained components.

The conversion, parameter mapping, JAX/Orbax layout work, TPU sharding, runtime integration, packaging, and qualification performed by `dangkhoa2016` do **not** create ownership of the underlying pretrained model weights and do **not** replace upstream licensing conditions.

Users redistributing the converted model artifact are responsible for preserving all applicable upstream copyright, license, attribution, and notice requirements.

## No relicensing of third-party material

Nothing in this repository should be interpreted as relicensing:

- upstream model weights;
- upstream source code not authored here;
- tokenizers or model configuration originating from upstream projects;
- third-party Python packages or runtime libraries;
- hosted services, datasets, trademarks, or logos.

See also [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [NOTICE.md](NOTICE.md), and the public model distribution's `LICENSES.md`/`NOTICE.md`.

## Public model licensing authority

The public JAX/Orbax model distribution documents the same mixed licensing boundary:

- Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo components — MIT;
- Qwen3-VL-derived Text Encoder/tokenizer components — Apache-2.0;
- conversion/integration work — attribution to `dangkhoa2016`, without superseding upstream terms.

Users should review the licenses shipped with the model artifact before redistribution or commercial deployment.

## Distribution packaging requirement

Standalone model distributions that include converted checkpoints or Qwen3-VL-derived tokenizer/processor assets should ship the preserved upstream license texts together with `LICENSES.md`, `NOTICE.md`, and this file. The companion GitHub root `LICENSE` alone is not sufficient to describe the licensing boundary of the mixed-origin model artifact.

Showcase images are governed separately by [`ASSET_LICENSE.md`](ASSET_LICENSE.md).
