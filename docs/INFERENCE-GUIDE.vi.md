# Hướng dẫn inference

> 🌐 Ngôn ngữ: [English](INFERENCE-GUIDE.md) · **Tiếng Việt**

## Cấu trúc artifact

Dùng artifact standalone của Edit-Turbo. Root cần có `text_encoder/`, `checkpoints/{text_encoder,transformer,vae}`, `manifests/` và `rope/`. Không cần một model Turbo-JAX riêng.

## Chạy baseline đã production-qualified

```bash
python bootstrap/08_run_tpu_edit.py   --image /path/reference.png   --prompt "your edit instruction"   --output /kaggle/working/edit-output   --artifact-root /path/to/mage-flow-edit-turbo-jax-tpu-v5e8   --resolution 512   --seeds 42,43,44,45
```

Baseline đã production-qualified là 512×512, q256, bốn denoising step. 768/q256 là validated scaling; 1024/q128 chỉ là cấu hình high-resolution experimental đã tài liệu hóa.

## Output

Runner ghi output của từng stage, `generation_summary.json` và các PNG cuối. Timing đã đo là qualification evidence, không phải cam kết latency phổ quát.
