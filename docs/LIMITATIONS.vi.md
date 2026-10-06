# Giới hạn

> 🌐 Ngôn ngữ: [English](LIMITATIONS.md) · **Tiếng Việt**

- Production qualification chỉ áp dụng cho 512×512 / q256 / bốn step trên topology TPU v5e-8 đã kiểm thử.
- 768×768 / q256 là validated scaling, không phải production baseline.
- 1024×1024 / q128 là high-resolution experimental.
- q128 và q256 không bit-identical ở 512.
- Timing là evidence của từng run, không phải service-level guarantee.
- CPU regression test tự nó không chứng minh numerical equivalence trên TPU.
- Đây là dự án runtime/conversion engineering độc lập; không retrain hay relicense upstream model weights.
