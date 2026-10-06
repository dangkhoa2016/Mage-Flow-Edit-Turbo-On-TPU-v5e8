# Kiến trúc

> 🌐 Ngôn ngữ: [English](ARCHITECTURE.md) · **Tiếng Việt**

Repository này chứa runtime JAX/Keras 3 dạng source-only dùng để chạy Mage-Flow-Edit-Turbo trên TPU v5e-8. Các model artifact JAX/Orbax dung lượng lớn được cung cấp riêng qua `--artifact-root`.

## Pipeline

1. Qwen3-VL multimodal conditioning mã hóa edit instruction và ảnh tham chiếu.
2. VAE mã hóa ảnh tham chiếu thành latent.
3. Edit transformer đóng gói token target/reference theo từng replica và chỉ tích phân phần dự đoán của target.
4. VAE giải mã latent target cuối thành PNG RGB.

## Topology TPU

Topology đã qualification là Kaggle TPU v5e-8 với tám device theo bố cục `4x2` replica/model. Transformer dùng binding 397 leaf đã ghi nhận, gồm 174 leaf sharded và 223 leaf replicated.

## Ranh giới qualification

- 512×512 / q256 / 4 steps: baseline đã production-qualified.
- 768×768 / q256: validated scaling.
- 1024×1024 / q128: chế độ high-resolution experimental.

`q128` và `q256` deterministic trong cấu hình đã kiểm thử, nhưng không bit-identical với nhau ở 512px.
