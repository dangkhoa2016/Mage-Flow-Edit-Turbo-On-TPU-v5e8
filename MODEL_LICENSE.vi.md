# License và Attribution cho các thành phần Model

> 🌐 Ngôn ngữ / Language: [English](MODEL_LICENSE.md) | **Tiếng Việt**

File này mô tả ranh giới license của các thành phần liên quan tới model được repository này sử dụng và public JAX/Orbax model artifact được phân phối riêng.

**Đây không phải một license mới và không thay thế, supersede hoặc relicensing bất kỳ model, model weight, tokenizer, configuration hay thành phần bên thứ ba nào của upstream.**

## License cho phần engineering của repository

Mã nguồn kỹ thuật và tài liệu gốc do repository này đóng góp được phát hành theo [MIT License](LICENSE) của repository:

```text
Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>
```

MIT grant ở cấp repository chỉ áp dụng cho phần mà tác giả repository có quyền cấp license.

## Thành phần Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo

Upstream project: https://github.com/microsoft/Mage
Upstream model family: Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo
Upstream copyright: Copyright (c) 2026 Microsoft
License upstream áp dụng: **MIT License**

Bản sao license MIT upstream được lưu tại [`licenses/Mage-Flow-MIT.txt`](licenses/Mage-Flow-MIT.txt).

Việc sử dụng tên Mage, Mage-Flow, Mage-Flow-Turbo, Mage-Flow-Edit-Turbo, Microsoft hoặc các nhãn liên quan chỉ nhằm attribution và xác định nguồn. Repository này không cấp quyền trademark.

## Text Encoder và tokenizer/configuration có nguồn gốc Qwen3-VL

Upstream project/model family: Qwen3-VL
Model upstream được bản conversion tham chiếu: Qwen3-VL-4B-Instruct
License upstream áp dụng: **Apache License 2.0**

Bản sao Apache License 2.0 từ upstream Qwen3-VL được lưu tại [`licenses/Qwen3-VL-Apache-2.0.txt`](licenses/Qwen3-VL-Apache-2.0.txt).

Các yêu cầu attribution, giữ NOTICE và điều kiện liên quan của Apache-2.0 vẫn có hiệu lực đối với các thành phần đó.

## Converted JAX/Orbax model artifacts

Các checkpoint Text Encoder, Transformer và VAE đã chuyển đổi là bản chuyển đổi format/runtime từ pretrained component upstream.

Công việc conversion, parameter mapping, JAX/Orbax layout, TPU sharding, runtime integration, packaging và qualification do `dangkhoa2016` thực hiện **không** tạo quyền sở hữu đối với pretrained model weights bên dưới và **không** thay thế license upstream.

Người dùng phân phối lại converted model artifact có trách nhiệm giữ các yêu cầu copyright, license, attribution và notice upstream tương ứng.

## Không relicensing tài sản bên thứ ba

Không nội dung nào trong repository này nên được hiểu là relicensing:

- upstream model weights;
- upstream source code không do repository này viết;
- tokenizer hoặc model configuration có nguồn từ upstream;
- Python package hoặc runtime library bên thứ ba;
- hosted service, dataset, trademark hoặc logo.

Xem thêm [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [NOTICE.md](NOTICE.md), và `LICENSES.md`/`NOTICE.md` trong public model distribution.

## Authority license của public model

Public JAX/Orbax model distribution ghi cùng ranh giới mixed licensing:

- Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo — MIT;
- Text Encoder/tokenizer có nguồn Qwen3-VL — Apache-2.0;
- conversion/integration — attribution cho `dangkhoa2016`, không supersede upstream terms.

Người dùng nên đọc các license đi kèm model artifact trước khi redistribution hoặc commercial deployment.

## Yêu cầu packaging khi phân phối model

Standalone model distribution có converted checkpoint hoặc tokenizer/processor asset có nguồn gốc Qwen3-VL cần ship kèm các upstream license text được lưu giữ cùng `LICENSES.md`, `NOTICE.md` và tài liệu này. Chỉ root `LICENSE` của GitHub companion repo là không đủ để mô tả ranh giới licensing của mixed-origin model artifact.

Showcase image được quản lý riêng tại [`ASSET_LICENSE.vi.md`](ASSET_LICENSE.vi.md).
