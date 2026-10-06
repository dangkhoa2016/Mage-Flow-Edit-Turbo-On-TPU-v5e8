# Xử lý sự cố

> 🌐 Ngôn ngữ: [English](TROUBLESHOOTING.md) · **Tiếng Việt**

## Lỗi đường dẫn artifact

Kiểm tra `--artifact-root` có `text_encoder/`, `checkpoints/`, `manifests/` và `rope/`.

## Lỗi khởi động TPU/JAX

Kiểm tra notebook đang gắn TPU v5e-8 và dùng các biến môi trường TPU đã tài liệu hóa. Tránh import CUDA runtime ngoài ý muốn vào process TPU.

## Thiếu bộ nhớ ở độ phân giải cao

Quay lại baseline đã production-qualified 512/q256. Đường 1024 chỉ được tài liệu hóa với q128 và vẫn là experimental.

## Output khác nhau giữa q128/q256

Đây là hành vi đã biết: hai query-chunk setting không bit-identical ở 512. Hãy so repeat trong cùng cấu hình.
