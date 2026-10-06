# Benchmark

> 🌐 Ngôn ngữ: [English](BENCHMARKS.md) · **Tiếng Việt**

Các số dưới đây là qualification measurement trên Kaggle TPU v5e-8, topology `4x2`, bốn denoising step.

| Chế độ | Trạng thái | Evidence transformer |
|---|---|---|
| 512 / q256 | Production-qualified | 50.85–53.78 s trên ba case acceptance |
| 768 / q256 | Validated scaling | repeat 66.59 s; wildlife 67.26 s; vehicle 67.01 s |
| 1024 / q128 | Experimental | island 133.22 s; peak HBM 15,944,458,240 bytes |

Acceptance production 512 có ba case PASS, mỗi case bốn PNG khác nhau, seed 42–45. Repeat 768 tái lập bit-for-bit cả bốn PNG hash trong cấu hình đã kiểm thử. q128 không bit-identical với q256 ở 512 nên hai cấu hình được qualification riêng.
