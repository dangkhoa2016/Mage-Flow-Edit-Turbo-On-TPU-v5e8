# Thông báo thành phần bên thứ ba

> 🌐 Ngôn ngữ: [English](THIRD_PARTY_NOTICES.md) · **Tiếng Việt**

Repository chứa hoặc tương tác với các thành phần bên thứ ba. MIT license ở root không thay thế license của các thành phần đó.

## Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo

Các material upstream được giữ dưới điều khoản upstream tương ứng. Bản license đã lưu được đặt tại `licenses/Mage-Flow-MIT.txt`.

## Qwen3-VL

Processor/tokenizer metadata và runtime tương thích với Qwen3-VL phải được sử dụng cùng các điều khoản upstream áp dụng. Bản Apache-2.0 được lưu tại `licenses/Qwen3-VL-Apache-2.0.txt`.

## Model artifact

Checkpoint JAX/Orbax không được cấp lại license bởi MIT của repository. Người dùng chịu trách nhiệm tuân thủ các điều khoản upstream áp dụng cho artifact/model.

Các legal text upstream trong `licenses/` được giữ nguyên văn; bản tiếng Việt này chỉ là giải thích và không thay thế văn bản pháp lý gốc.

## Bộ notice cho standalone artifact

Standalone model artifact trên Hugging Face/Kaggle cần giữ `LICENSES.md`, `MODEL_LICENSE.md`, `ASSET_LICENSE.md`, `NOTICE.md` và toàn bộ upstream license text trong `licenses/`. Như vậy notice đi cùng checkpoint và tokenizer/processor asset được redistribute thay vì chỉ phụ thuộc vào link quay lại GitHub.
