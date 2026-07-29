# YC.md — Yêu cầu dự án NCKH

Dự án: Tối ưu hóa độ tin cậy cho phân đoạn ảnh y tế bằng Conformal Prediction
Người thực hiện: T.Hung
Trợ lý: Claude Code

---

## Dự án này làm gì?

Xây dựng hệ thống đánh giá **độ tin cậy (Uncertainty Quantification)** cho mô hình phân đoạn ảnh y tế (nnU-Net). Thay vì chỉ đưa ra kết quả dự đoán duy nhất, cung cấp **khoảng tin cậy (Prediction Interval)** có đảm bảo toán học (ví dụ: đảm bảo 90% thể tích thực tế nằm trong khoảng dự báo).

Framework chính: **CRC-FS** (Conformal Risk Control in Feature Space) — kết hợp 3 bài báo:
- Split Conformal Prediction (Angelopoulos & Bates)
- Conformal Risk Control (Angelopoulos et al., ICLR 2024)
- COMPASS (Cheung et al., ICLR 2026)

---

## Dataset

| Dataset | Trạng thái | Metrics |
|---------|-----------|---------|
| ACDC (tim, MRI) | ✅ Đã test | LV/RV volume, EF, Myo mass |
| LiTS (gan, CT) | 🔄 Đang test | Liver volume, Tumor volume |
| KiTS (thận, CT) | ⏳ Chưa có | Kidney tumor volume |
| LIDC-IDRI (phổi, CT) | ⏳ Chưa có | Nodule diameter + multi-rater |

---

## Cách làm việc với Claude

1. **Code sạch sẽ, có comment tiếng Việt**
2. **Không bịa số liệu** — mọi kết quả phải từ file CSV thực tế
3. **Giải thích ngắn gọn** — không dài dòng
4. **Mọi thứ chạy trên GPU** — tận dụng RTX 5070
5. **Giữ disk** — không copy file thừa, dùng hardlink/symlink nếu được
6. **Khi sửa code** — chạy test ngay để xác nhận không lỗi
7. **Cập nhật nhật ký** (`Nhat_Ky_Nghien_Cuu_CRC_FS.md`) sau mỗi thay đổi lớn
8. **Ghi rõ command** — để tôi copy-paste chạy luôn

---

## Việc cần làm tiếp theo

- [ ] Chạy xong LiTS pipeline — có kết quả 6 methods
- [ ] Phân tích kết quả LiTS — so sánh với ACDC
- [ ] Tối ưu width CRC-FS — tìm uncertainty measure tốt hơn entropy
- [ ] Tải dataset KiTS — mở rộng thêm dataset thứ 3
- [ ] Tải dataset LIDC-IDRI — phân tích annotator disagreement
- [ ] Viết paper

---

## Ghi chú

- Mọi file `.py` quan trọng nằm trong `experiments/`
- File nhật ký: `Nhat_Ky_Nghien_Cuu_CRC_FS.md`
- Kết quả trong `results/`
- Notebook EDA: `notebooks/01_explore_acdc.ipynb` (ACDC), `notebooks/02_explore_lits.ipynb` (LiTS)
