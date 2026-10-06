# Tái lập kết quả

> 🌐 Ngôn ngữ: [English](REPRODUCIBILITY.md) · **Tiếng Việt**

Các tuyên bố reproducibility phụ thuộc cấu hình. Production acceptance cố định dùng 512×512, q256, bốn step và seed 42, 43, 44, 45. Mỗi case acceptance tạo bốn PNG có hash khác nhau.

Repeat 768/q256 tái lập bit-for-bit 4/4 PNG hash. Đường chạy 1024/q128 đã kiểm thử cũng deterministic trong q128. Tuy nhiên q128 và q256 không cross-bit-exact ở 512. Không nên so output giữa các query-chunk setting như thể chúng là cùng một cấu hình đã qualification.

Machine-readable authority nằm trong `acceptance/PRODUCTION_ACCEPTANCE.json` và `release-evidence/`.
