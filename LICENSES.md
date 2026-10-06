# Distribution License Matrix

> Language: **English** · [Tiếng Việt](LICENSES.vi.md)

This project is a mixed-origin distribution. The root MIT license applies only to repository-authored material for which the author can grant MIT rights; it does **not** relicense upstream checkpoints, tokenizers, processor assets, source images, trademarks, or other third-party material.

| Component / material | Origin | Applicable license / status | Preserved authority |
| --- | --- | --- | --- |
| Repository-authored JAX/Keras runtime, bootstrap code, tests and documentation | `dangkhoa2016` | MIT, where the author has the right to grant it | `LICENSE` |
| Mage-Flow / Mage-Flow-Edit-Turbo model lineage, transformer/VAE-related upstream material | Microsoft Mage | MIT | `licenses/Mage-Flow-MIT.txt` |
| Qwen3-VL-derived text encoder / tokenizer / processor lineage | Qwen Team / Alibaba | Apache License 2.0 | `licenses/Qwen3-VL-Apache-2.0.txt` |
| Converted JAX/Orbax checkpoints | Format/runtime conversions of upstream pretrained components | Underlying upstream terms remain applicable; conversion does not create ownership of pretrained weights | `MODEL_LICENSE.md` |
| Showcase and source images | Mixed / release-evidence provenance | See per-asset provenance; no blanket MIT grant | `ASSET_LICENSE.md` |
| Third-party Python/runtime ecosystem | Respective upstream projects | Respective upstream licenses | `THIRD_PARTY_NOTICES.md` |

## Redistribution checklist

When redistributing a standalone model artifact, preserve at minimum:

- this `LICENSES.md` matrix;
- `MODEL_LICENSE.md`;
- `ASSET_LICENSE.md` when showcase assets are included;
- `NOTICE.md` and `THIRD_PARTY_NOTICES.md`;
- `licenses/Mage-Flow-MIT.txt`;
- `licenses/Qwen3-VL-Apache-2.0.txt`;
- applicable copyright/attribution notices already embedded in upstream material.

This matrix is a packaging guide and attribution boundary, not legal advice and not a replacement for the complete upstream license texts.
