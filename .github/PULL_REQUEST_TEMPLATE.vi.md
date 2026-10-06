# Pull request

> 🌐 Ngôn ngữ: [English](PULL_REQUEST_TEMPLATE.md) · **Tiếng Việt**

## Tóm tắt

Mô tả thay đổi có phạm vi rõ ràng và lý do cần thay đổi.

## Kiểm chứng

- [ ] `python -m compileall -q runtime bootstrap tests scripts`
- [ ] `KERAS_BACKEND=jax USE_TORCH_XLA=0 pytest -q tests`
- [ ] `python scripts/check_repo_health.py`
- [ ] `git diff --check`

## Ảnh hưởng qualification

- [ ] Không thay đổi runtime semantic.
- [ ] Có thay đổi runtime semantic; đã kèm TPU qualification evidence mới.

## An toàn

- [ ] Không commit token, credential, URL riêng tư, model weight hoặc checkpoint blob.
- [ ] Tài liệu public EN/VI vẫn đồng bộ khi áp dụng.
