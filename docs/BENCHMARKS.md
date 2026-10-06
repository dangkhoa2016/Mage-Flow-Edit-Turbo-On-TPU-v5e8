# Benchmarks

> 🌐 Language: **English** · [Tiếng Việt](BENCHMARKS.vi.md)

All numbers below are qualification measurements on Kaggle TPU v5e-8, topology `4x2`, four denoising steps.

| Mode | Status | Transformer evidence |
|---|---|---|
| 512 / q256 | Production-qualified | 50.85–53.78 s across the three accepted cases |
| 768 / q256 | Validated scaling | 66.59 s repeat; selected wildlife 67.26 s; vehicle 67.01 s |
| 1024 / q128 | Experimental | island 133.22 s; peak HBM 15,944,458,240 bytes |

The 512 production acceptance contains three PASS cases, four distinct PNGs per case, seeds 42–45. The 768 repeat reproduced four PNG hashes bit-for-bit in the tested configuration. q128 does not match q256 bit-for-bit at 512, so they are qualified separately.
