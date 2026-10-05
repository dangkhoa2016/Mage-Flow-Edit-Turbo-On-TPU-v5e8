# Mage-Flow-Edit-Turbo TPU v5e-8 benchmark evidence

| Resolution | Query chunk | Conditioning internal | Transformer compute | Peak HBM/device | VAE cold |
|---:|---:|---:|---:|---:|---:|
| 512 | 256 | 61.10s | 49.72s | 10.39 GB | 18.49s |
| 768 | 256 | 46.31s | 69.08s | 13.26 GB | 21.47s |
| 1024 | 128 | 48.65s | 129.42s | 15.93 GB | 20.64s |

## Extended validation

- **512 / q256:** production-qualified frozen baseline; all four frozen acceptance PNGs reproduce bit-for-bit.
- **768 / q256:** successful scaling validation and a fresh repeat reproduced all four PNG SHA-256 values bit-for-bit.
- **1024 / q128:** experimental high-resolution mode; fox repeat reproduced all four hashes bit-for-bit and a separate floating-island winter edit passed end-to-end.
- **q128 vs q256:** 512 q128 outputs are deterministic but are **not** bit-exact with 512 q256; query chunk must not be described as output-neutral.
- **Object/vehicle 768 / q256:** monorail recolor passed; transformer 67.01s, peak HBM 13.46 GB/device.
- **Wildlife showcase 768 / q256:** snow-leopard mountain-to-winter edit passed; selected seed 52; transformer 67.26s, peak HBM 13.47 GB/device.

> 1024 remains experimental rather than part of the original production qualification.
