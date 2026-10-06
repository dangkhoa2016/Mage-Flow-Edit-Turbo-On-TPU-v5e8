# Mage-Flow-Edit-Turbo on TPU v5e-8

> 🌐 Language: **English** · [Tiếng Việt](README.vi.md)

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8)](https://github.com/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![JAX](https://img.shields.io/badge/JAX-runtime-informational)
![Keras 3](https://img.shields.io/badge/Keras-3-informational)
![TPU v5e-8](https://img.shields.io/badge/TPU-v5e--8-informational)

A source-only **JAX/Keras 3 inference runtime and qualification package for Mage-Flow-Edit-Turbo on Kaggle TPU v5e-8**. This repository documents the engineering required to make the edit pipeline executable, reproducible, reviewable and publishable on TPU: Qwen3-VL multimodal conditioning, reference-image VAE encoding, Edit transformer execution, deterministic RoPE support, TPU sharding, staged orchestration, CPU regression tests, machine-readable acceptance evidence and a standalone JAX/Orbax model artifact.

The public Edit-Turbo artifact is **standalone for this pipeline**. It supplies the text encoder, Edit transformer, VAE, manifests, RoPE captures and model-side metadata required by the runner and does **not** require a separate Turbo-JAX model.

**Public surfaces:** [Hugging Face model](https://huggingface.co/dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8) · [Kaggle model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-edit-turbo-jax-tpu-v5e8) · [Kaggle Public Acceptance notebook](https://www.kaggle.com/code/dangkhoa2016/mage-flow-edit-turbo-tpu-v5e8-public-acceptance) · [GitHub v1.0.0](https://github.com/dangkhoa2016/Mage-Flow-Edit-Turbo-On-TPU-v5e8/releases/tag/v1.0.0)

## Approved TPU showcase preview

![Approved Mage-Flow-Edit-Turbo before/after showcase](docs/images/approved-showcase-before-after.png)

*Approved v1.0.0 before/after contact sheet from the canonical release bundle. All four selected public edits use the validated 768×768 / q256 path; the image is included here so reviewers can evaluate the showcase without leaving GitHub.*

## What v1.0.0 demonstrates

This release is not only a model-conversion snapshot. It is an evidence-backed TPU execution path with explicit qualification boundaries:

- **512×512 / q256 / 4 steps — production-qualified baseline.** Three edit scenarios passed end-to-end; each ran seeds 42–45 and produced four distinct PNG SHA-256 values.
- **768×768 / q256 — validated scaling.** A fresh four-seed repeat reproduced all four PNG hashes bit-for-bit.
- **1024×1024 / q128 — experimental high resolution.** Multiple subjects completed successfully; the dedicated q128 repeat is deterministic inside the tested q128 configuration.
- **q128 is not output-neutral relative to q256.** q128 and q256 are not bit-identical at 512px, so they are documented as separately qualified execution configurations.
- **Source repository stays lightweight.** Model/checkpoint blobs are deliberately kept out of Git while the public JAX/Orbax artifact remains directly mountable by the runner.

## Engineering scope

The repository covers the complete edit-inference path instead of wrapping a pre-existing TPU command:

1. **Qwen3-VL text conditioning** — restore and bind the converted BF16 text encoder, implement the text forward path, RMSNorm, attention, GQA repeat-KV, MLP and multimodal rotary behavior in JAX.
2. **Qwen3-VL vision conditioning** — process the reference image and expose the visual/deep-stack features required by edit conditioning.
3. **Reference-image VAE path** — encode the source image into latent space and decode the final edited target back to RGB.
4. **Mage-Flow Edit transformer** — reconstruct the transformer execution path, bind the converted 397-leaf checkpoint and pack target/reference edit sequences correctly.
5. **Deterministic RoPE** — use captured basis arrays and an explicit runtime provider rather than relying on an unavailable hidden dependency.
6. **TPU partitioning and orchestration** — run the qualified Kaggle TPU v5e-8 topology as `4x2`, with the recorded transformer plan of **174 sharded + 223 replicated = 397 leaves**.
7. **Verification and evidence** — CPU regression tests, repository-health checks, production acceptance JSON, extended TPU validation and exact showcase hashes.

Converted checkpoint contracts represented by the runtime include the **397-leaf Edit transformer**, a **713-leaf BF16 text encoder** restore path and an **839-leaf VAE** restore path. These numbers describe the converted runtime state trees used by this project; they are not model parameter counts.

## Runtime architecture

```text
Edit instruction + reference image
            │
            ▼
  Qwen3-VL multimodal conditioning
     ├─ text encoder
     └─ vision/deep-stack features
            │
            ├──────────────┐
            ▼              │
   Reference VAE encode    │
            │              │
            └──────┬───────┘
                   ▼
       Edit transformer on TPU
       target + reference tokens
                   │
                   ▼
          target latent update
                   │
                   ▼
             VAE decode
                   │
                   ▼
                PNG output
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the runtime decomposition and TPU topology.

## Qualification matrix

| Configuration | Classification | Reproducibility evidence | Representative transformer evidence |
|---|---|---|---:|
| 512×512 · q256 · 4 steps | **Production-qualified baseline** | 3/3 acceptance cases PASS; seeds 42–45; 4 unique PNGs per case | 50.85–53.78 s across accepted cases |
| 768×768 · q256 · 4 steps | **Validated scaling** | fresh four-seed repeat reproduced 4/4 PNG SHA-256 values | 66.59 s repeat; showcase runs ≈67 s |
| 1024×1024 · q128 · 4 steps | **Experimental high resolution** | fox q128 repeat reproduced 4/4 hashes; floating-island edit passed | island 133.22 s |

The extended benchmark record also captured the following representative stage measurements:

| Resolution | Query chunk | Conditioning internal | Transformer compute | Peak HBM/device | VAE cold |
|---:|---:|---:|---:|---:|---:|
| 512 | 256 | 61.10 s | 49.72 s | 10.39 GB | 18.49 s |
| 768 | 256 | 46.31 s | 69.08 s | 13.26 GB | 21.47 s |
| 1024 | 128 | 48.65 s | 129.42 s | 15.93 GB | 20.64 s |

These are run-specific measurements, not service-level guarantees. The authoritative benchmark notes are in [`release-evidence/BENCHMARKS.md`](release-evidence/BENCHMARKS.md) and [`release-evidence/EXTENDED_VALIDATION.json`](release-evidence/EXTENDED_VALIDATION.json).

## Production acceptance

The frozen 512/q256 baseline was accepted on three distinct editing scenarios:

| Case | Seeds | Conditioning tokens | Transformer wall time | Peak HBM/device | Result |
|---|---|---:|---:|---:|---|
| fox-autumn | 42, 43, 44, 45 | 286 | 50.852 s | 10,385,165,824 bytes | PASS · 4/4 unique PNG hashes |
| portrait-golden-hour | 42, 43, 44, 45 | 309 | 52.651 s | 10,425,102,848 bytes | PASS · 4/4 unique PNG hashes |
| island-winter | 42, 43, 44, 45 | 311 | 53.776 s | 10,395,812,864 bytes | PASS · 4/4 unique PNG hashes |

Machine-readable authority: [`acceptance/PRODUCTION_ACCEPTANCE.json`](acceptance/PRODUCTION_ACCEPTANCE.json). The human-readable closeout is [`acceptance/PRODUCTION_ACCEPTANCE_CLOSEOUT.md`](acceptance/PRODUCTION_ACCEPTANCE_CLOSEOUT.md).

## Determinism: what is proven and what is not

Determinism is stated per tested configuration, not assumed globally.

- 512/q256 is the frozen production-qualified baseline.
- 768/q256 has a fresh repeat with all four PNG SHA-256 values reproduced bit-for-bit.
- 1024/q128 has deterministic evidence inside the q128 setup, but remains experimental high resolution.
- 512/q128 and 512/q256 are **not bit-identical**. The `query_chunk` setting therefore belongs to the qualification identity and must not be described as output-neutral.
- CPU regression tests validate contracts and wiring; they do **not** replace TPU numerical qualification.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the reproducibility contract.

## Approved public showcase

The selected v1.0.0 showcase uses the validated **768/q256** path for all four public pairs:

| Edit | Seed | Transformer evidence | Peak HBM/device |
|---|---:|---:|---:|
| Snow Leopard · mountain → snowy winter | 52 | 67.26 s | 13.47 GB |
| Portrait · rain → golden hour | 43 | validated 768/q256 path | validated 768/q256 path |
| Floating Island · green → winter | 43 | validated 768/q256 path | validated 768/q256 path |
| Monorail · white/silver → deep red | 43 | 67.01 s | 13.46 GB |

Exact prompts, selected SHA-256 values, source/after dimensions and the separate 1024/q128 high-resolution proofs are recorded in [`release-evidence/PUBLISH_MANIFEST.json`](release-evidence/PUBLISH_MANIFEST.json). The 1024 floating-island proof is intentionally **not** substituted for the selected 768 public pair.

## Standalone artifact contract

Point `--artifact-root` at the public Edit-Turbo JAX artifact. The runtime expects the artifact to provide:

```text
artifact-root/
├── text_encoder/
├── checkpoints/
│   ├── text_encoder/
│   ├── transformer/
│   └── vae/
├── manifests/
└── rope/
```

The artifact is published separately from Git because the repository intentionally excludes model/checkpoint payloads. See [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for the contract and [`docs/KAGGLE.md`](docs/KAGGLE.md) for the Kaggle mounting workflow.

## One-command TPU edit runner

```bash
python bootstrap/08_run_tpu_edit.py \
  --image /path/reference.png \
  --prompt "Transform the environment while preserving the subject" \
  --output /kaggle/working/edit-output \
  --artifact-root /path/to/mage-flow-edit-turbo-jax-tpu-v5e8 \
  --resolution 512 \
  --seeds 42,43,44,45
```

The production baseline is 512/q256/4 steps. Use [`docs/INFERENCE-GUIDE.md`](docs/INFERENCE-GUIDE.md) before changing resolution or query chunk, because higher-resolution configurations have different qualification status.

## Repository map

| Path | Purpose |
|---|---|
| `runtime/experiment/` | text/vision conditioning, RoPE, VAE and transformer runtime drivers |
| `runtime/source/mage_flow_keras/` | reconstructed Mage transformer source used by the converted checkpoint |
| `bootstrap/` | TPU stage entry points and the one-command orchestration runner |
| `qwen3vl-processor-assets/` | processor/tokenizer metadata only; no model weights |
| `acceptance/` | production qualification authority and closeout |
| `release-evidence/` | extended validation, benchmark evidence and approved showcase manifest |
| `notebooks/` | Kaggle production notebook |
| `tests/` | CPU regression tests for runtime and repository contracts |
| `scripts/` | acceptance builder and repository-health validation |
| `docs/` | bilingual architecture, inference, benchmark, reproducibility, artifact, limitation and release documentation |

## Verify the source repository

```bash
python -m compileall -q runtime bootstrap tests scripts
KERAS_BACKEND=jax USE_TORCH_XLA=0 pytest -q tests
python scripts/check_repo_health.py
git diff --check
```

`check_repo_health.py` additionally enforces the public-documentation contract, exact MIT author line, source-only/no-weight policy, qualification wording and canonical showcase metadata.

## Evidence hierarchy

For review or reproduction, use the most specific authority rather than inferring from screenshots:

1. [`acceptance/PRODUCTION_ACCEPTANCE.json`](acceptance/PRODUCTION_ACCEPTANCE.json) — frozen 512/q256 production acceptance.
2. [`release-evidence/EXTENDED_VALIDATION.json`](release-evidence/EXTENDED_VALIDATION.json) — 768 scaling, 1024/q128 and cross-query-chunk findings.
3. [`release-evidence/PUBLISH_MANIFEST.json`](release-evidence/PUBLISH_MANIFEST.json) — approved public showcase, prompts and exact hashes.
4. [`release-evidence/BENCHMARKS.md`](release-evidence/BENCHMARKS.md) — compact benchmark summary.
5. [`docs/RELEASE-EVIDENCE-v1.0.0.md`](docs/RELEASE-EVIDENCE-v1.0.0.md) — release evidence guide.
6. [`notebooks/kaggle-production-demo.ipynb`](notebooks/kaggle-production-demo.ipynb) — retained repository execution source; not the current public Kaggle notebook surface.

## Scope and limitations

- Production qualification is limited to **512×512 / q256 / 4 steps** on the tested Kaggle TPU v5e-8 `4x2` topology.
- 768/q256 is validated scaling, not the production baseline.
- 1024/q128 is experimental high resolution.
- Timing and HBM measurements are evidence from specific runs, not guaranteed performance targets.
- CPU tests do not establish TPU numerical equivalence by themselves.
- This project is an independent runtime/conversion engineering effort; it does not retrain Mage-Flow-Edit-Turbo and does not relicense upstream weights.

More detail: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Licensing and provenance

Repository-authored engineering code and documentation are licensed under MIT where the author has the right to grant those terms. Upstream model weights/material and third-party components keep their applicable licenses and notices.

Read [`LICENSE`](LICENSE), [`LICENSES.md`](LICENSES.md), [`MODEL_LICENSE.md`](MODEL_LICENSE.md), [`ASSET_LICENSE.md`](ASSET_LICENSE.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`NOTICE.md`](NOTICE.md) and [`licenses/`](licenses/) before redistributing model-side artifacts or showcase assets.

## Release documentation

- [`RELEASE_NOTES_v1.0.0.md`](RELEASE_NOTES_v1.0.0.md) — v1.0.0 release summary.
- [`CHANGELOG.md`](CHANGELOG.md) — project change history.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow.
- [`SECURITY.md`](SECURITY.md) — security policy.
- [`SUPPORT.md`](SUPPORT.md) — support boundaries.

**v1.0.0 is the first community-facing release of this TPU runtime and evidence package.** The goal is not merely to show that an image can be generated, but to make the execution path, numerical boundaries, artifacts and release evidence inspectable by other engineers.
