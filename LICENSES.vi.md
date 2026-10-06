# Ma trận giấy phép của distribution

> Ngôn ngữ: [English](LICENSES.md) · **Tiếng Việt**

Dự án này là một distribution có nhiều nguồn gốc. MIT license ở root chỉ áp dụng cho phần do repository tự tạo và tác giả có quyền cấp MIT; nó **không** relicense upstream checkpoint, tokenizer, processor asset, ảnh nguồn, trademark hay tài liệu bên thứ ba khác.

| Thành phần / tài liệu | Nguồn | License / trạng thái áp dụng | Authority được lưu giữ |
| --- | --- | --- | --- |
| JAX/Keras runtime, bootstrap code, test và tài liệu do repository tự viết | `dangkhoa2016` | MIT trong phạm vi tác giả có quyền cấp | `LICENSE` |
| Mage-Flow / Mage-Flow-Edit-Turbo lineage, transformer/VAE upstream material | Microsoft Mage | MIT | `licenses/Mage-Flow-MIT.txt` |
| Qwen3-VL-derived text encoder / tokenizer / processor lineage | Qwen Team / Alibaba | Apache License 2.0 | `licenses/Qwen3-VL-Apache-2.0.txt` |
| Converted JAX/Orbax checkpoint | Chuyển đổi format/runtime từ upstream pretrained components | Các điều khoản upstream nền vẫn áp dụng; conversion không tạo ownership đối với pretrained weights | `MODEL_LICENSE.vi.md` |
| Showcase và ảnh nguồn | Nhiều nguồn / provenance theo release evidence | Theo provenance từng asset; không có blanket MIT grant | `ASSET_LICENSE.vi.md` |
| Python/runtime ecosystem bên thứ ba | Các upstream project tương ứng | License upstream tương ứng | `THIRD_PARTY_NOTICES.vi.md` |

## Checklist khi phân phối lại

Khi redistribute standalone model artifact, tối thiểu cần giữ:

- `LICENSES.md`/`LICENSES.vi.md`;
- `MODEL_LICENSE.md`;
- `ASSET_LICENSE.md` nếu có showcase asset;
- `NOTICE.md` và `THIRD_PARTY_NOTICES.md`;
- `licenses/Mage-Flow-MIT.txt`;
- `licenses/Qwen3-VL-Apache-2.0.txt`;
- các copyright/attribution notice áp dụng đã có trong upstream material.

Ma trận này là hướng dẫn packaging và ranh giới attribution, không phải tư vấn pháp lý và không thay thế nội dung license upstream đầy đủ.
