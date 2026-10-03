# Production Acceptance Closeout

Mage-Flow-Edit-Turbo full multimodal JAX inference was qualified on Kaggle TPU v5e-8 at 512×512, topology `4x2`, four denoising steps, and seeds 42–45.

## Acceptance matrix

| Case | Edit class | Conditioning tokens | Transformer wall | Peak HBM/device | Result |
| --- | --- | ---: | ---: | ---: | --- |
| fox-autumn | winter → autumn environment | 286 | 50.85 s | 10.39 GB | PASS |
| portrait-golden-hour | rainy/cold → dry/golden-hour portrait | 309 | 52.65 s | 10.43 GB | PASS |
| island-winter | green fantasy landscape → snowy winter | 311 | 53.78 s | 10.40 GB | PASS |

Every case produced four distinct PNG SHA-256 values, passed all stage status gates, used the 397-leaf Edit transformer with 174 sharded / 223 replicated leaves, and passed VAE determinism/uniqueness gates.

## Human visual review

- **Fox:** subject, pose, framing, and silhouette were retained while the snowy scene changed to warm autumn foliage.
- **Portrait:** face, hairstyle, pose, clothing, backpack, and framing remained strongly consistent while visible rain/wet appearance was removed and lighting became golden-hour warm.
- **Floating island:** main island geometry, surrounding floating islands, waterfall layout, and camera angle remained recognizable while vegetation and surfaces became snow-covered winter scenery. Small architectural detail variation remained within the expected generative-edit boundary.

Machine-readable hashes and timings are authoritative in [`PRODUCTION_ACCEPTANCE.json`](PRODUCTION_ACCEPTANCE.json).
