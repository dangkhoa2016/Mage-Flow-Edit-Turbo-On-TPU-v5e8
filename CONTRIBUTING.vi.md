# Đóng góp

> 🌐 Ngôn ngữ: [English](CONTRIBUTING.md) · **Tiếng Việt**

Cảm ơn bạn đã đóng góp cho Mage-Flow-Edit-Turbo-On-TPU-v5e8.

## Phạm vi

Repository duy trì runtime JAX/Keras TPU, orchestration production, regression tests, acceptance metadata và tài liệu cho Mage-Flow-Edit-Turbo đã chuyển đổi. Thay đổi upstream model weights hoặc tuyên bố về upstream training nằm ngoài thẩm quyền repository.

## Trước khi mở pull request

Chạy `compileall`, CPU pytest, `scripts/check_repo_health.py` và `git diff --check` như trong PR template.

## Thay đổi runtime semantic

Thay đổi attention, sharding, checkpoint mapping, BF16 timestep semantics, denoising steps hoặc TPU placement cần qualification TPU mới trước khi được mô tả là tương đương baseline v1.0.0.

## Tài liệu và licensing

Public documentation được duy trì theo cặp English/Vietnamese. Không xóa attribution/license upstream. Không commit token, credential, private URL hoặc dữ liệu nhạy cảm.
