# Extended Validation Benchmarks

> 🌐 Ngôn ngữ: [English](BENCHMARKS.md) · **Tiếng Việt**

Evidence mở rộng giữ ranh giới qualification rõ ràng: 512/q256 là production baseline; 768/q256 là validated scaling; 1024/q128 là experimental high resolution.

Repeat 768/q256 tái lập bit-for-bit 4/4 PNG hash. Wildlife 768 seed 52 có transformer wall time 67.2558 s; object/vehicle 768 seed 43 là 67.0084 s. Island 1024/q128 chạy 133.2225 s với peak HBM 15,944,458,240 byte.

q128 deterministic trong cấu hình đã kiểm thử nhưng không bit-identical với q256 ở 512.
