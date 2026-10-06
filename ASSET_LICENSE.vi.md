# Thông báo về nguồn gốc và giới hạn tái sử dụng asset

> Ngôn ngữ: [English](ASSET_LICENSE.md) · **Tiếng Việt**

Tài liệu này ghi lại ranh giới về nguồn gốc và việc tái sử dụng các visual asset được phân phối cùng repository, GitHub release bundle và standalone model artifact trên Hugging Face/Kaggle.

Đây **không** phải là giấy phép cấp quyền bao trùm cho ảnh nguồn hoặc output được sinh ra. Việc một asset xuất hiện trong repository, release archive, Hugging Face hoặc Kaggle không tự chứng minh repository sở hữu asset đó hoặc có quyền cấp phép lại asset đó.

SHA-256, kích thước, prompt, seed và cấu hình inference dùng để chứng minh **khả năng truy vết và tái lập về mặt kỹ thuật**. Các thông tin này không tự xác lập quyền sở hữu bản quyền hoặc quyền tái sử dụng.

## Showcase chuẩn v1.0.0

Nguồn kỹ thuật có thẩm quyền cho danh sách showcase là `release-evidence/PUBLISH_MANIFEST.json`.

| Cặp showcase | Ranh giới nguồn gốc / quyền đối với ảnh nguồn | Trạng thái output |
| --- | --- | --- |
| Snow Leopard · mountain → snowy winter | Dựa trên một sample asset Mage-Flow-Turbo đã được công bố trước đó. Repository này không đưa ra tuyên bố độc lập về quyền sở hữu ảnh nguồn và cũng không cấp một giấy phép tái sử dụng riêng cho source asset đó. | Sinh bởi pipeline Mage-Flow-Edit-Turbo đã qualification, seed 52, 768/q256. |
| Portrait · rain → golden hour | Repository này không đưa ra tuyên bố độc lập về quyền sở hữu ảnh nguồn và cũng không cấp một giấy phép tái sử dụng riêng cho ảnh nguồn. | Sinh bởi pipeline đã qualification, seed 43, 768/q256. |
| Floating Island · green → winter | Repository này không đưa ra tuyên bố độc lập về quyền sở hữu ảnh nguồn và cũng không cấp một giấy phép tái sử dụng riêng cho ảnh nguồn. | Sinh bởi pipeline đã qualification, seed 43, 768/q256. |
| Monorail · white/silver → deep red | Repository này không đưa ra tuyên bố độc lập về quyền sở hữu ảnh nguồn và cũng không cấp một giấy phép tái sử dụng riêng cho ảnh nguồn. | Sinh bởi pipeline đã qualification, seed 43, 768/q256. |

Hash và metadata inference chính xác vẫn được ghi trong `release-evidence/PUBLISH_MANIFEST.json`.

## Contact sheet

`docs/images/approved-showcase-before-after.png` và contact sheet tương ứng trong release/model artifact chỉ là bản trình bày tổng hợp của các cặp showcase chuẩn ở trên. Việc tổng hợp thành contact sheet không tạo ra hoặc mở rộng quyền đối với các ảnh nguồn bên dưới.

## Generated outputs

Repository không đưa ra tuyên bố bao trùm rằng mọi output do AI sinh ra đều có bản quyền độc lập hoặc thuộc quyền sở hữu độc quyền của `dangkhoa2016`. Việc tái sử dụng output vẫn có thể chịu ảnh hưởng bởi quyền áp dụng cho ảnh nguồn, điều khoản của các thành phần model upstream và pháp luật liên quan.

## Trademark và nội dung bên thứ ba

Tên dự án, tên công ty, logo, trademark và nội dung bên thứ ba được thể hiện trong ảnh vẫn thuộc phạm vi quyền của chủ sở hữu tương ứng. Việc xuất hiện trong technical evidence không đồng nghĩa với endorsement hay chuyển giao trademark, publicity hoặc các quyền khác của bên thứ ba.

## Hướng dẫn tái sử dụng thực tế

- Code và tài liệu do repository tự viết: tuân theo root `LICENSE` trong phạm vi repository có quyền cấp phép.
- Model/checkpoint component: tuân theo `MODEL_LICENSE.md`, `LICENSES.md` và các upstream license text đã được lưu giữ.
- Showcase/source image: không suy luận quyền sở hữu hoặc quyền tái sử dụng chỉ từ việc ảnh có mặt trong repository hoặc release bundle; cần xác minh quyền áp dụng cho ảnh nguồn trước khi xuất bản hoặc phân phối lại.
- Việc asset có thể tải xuống từ GitHub, Hugging Face hoặc Kaggle không tạo ra một license rộng hơn.
