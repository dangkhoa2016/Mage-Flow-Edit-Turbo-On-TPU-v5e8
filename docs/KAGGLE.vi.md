# Hướng dẫn Kaggle TPU

> 🌐 Ngôn ngữ: [English](KAGGLE.md) · **Tiếng Việt**

Môi trường tham chiếu là Kaggle TPU v5e-8. Hãy dùng standalone Kaggle model artifact công khai và production notebook. Notebook dùng một nguồn artifact Edit-Turbo duy nhất, không cần model Turbo-JAX riêng.

Giữ model artifact ngoài Git, trỏ `--artifact-root` tới thư mục model đã mount và ghi output sinh ra dưới `/kaggle/working`.

Trước khi chạy public, kiểm tra TPU visibility, phiên bản JAX/Keras, các thư mục artifact và dung lượng đĩa. CPU test suite của repository kiểm tra wiring/contract nhưng không thay thế TPU qualification.
