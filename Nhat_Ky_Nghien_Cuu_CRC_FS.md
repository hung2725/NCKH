# Nhật Ký Nghiên Cứu — Dự Án CRC-FS

**Dự án:** NCKH — Tối ưu hóa độ tin cậy cho phân đoạn ảnh y tế
**Người thực hiện:** T.Hung
**Ngày:** 27/07/2026

---

## MỤC LỤC

1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Dataset & pretrained models](#2-dataset--pretrained-models)
3. [Cách chạy](#3-cách-chạy)
4. [Kiến trúc codebase](#4-kiến-trúc-codebase)
5. [Kết quả thực nghiệm](#5-kết-quả-thực-nghiệm)
6. [Thiết kế thuật toán CRC-FS](#6-thiết-kế-thuật-toán-crc-fs)
7. [Quá trình phát triển & các lỗi đã sửa](#7-quá-trình-phát-triển--các-lỗi-đã-sửa)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Cài Đặt Môi Trường

### 1.1 Yêu cầu phần cứng

| Thành phần | Tối thiểu | Đã test |
|------------|-----------|---------|
| GPU | NVIDIA 8GB VRAM | RTX 5070 Laptop 8.5GB |
| RAM | 16GB | 16GB |
| Disk | 100GB free | — |
| OS | Windows 10/11 | Windows 11 build 26200 |

### 1.2 Cài Python + thư viện

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install nnunet nibabel numpy pandas matplotlib seaborn scikit-learn tqdm jupyter
```

### 1.3 Cấu hình nnU-Net

```powershell
[System.Environment]::SetEnvironmentVariable("nnUNet_raw_data_base", "C:\Users\T.Hung\nnunet_v1\raw", "User")
[System.Environment]::SetEnvironmentVariable("nnUNet_preprocessed", "C:\Users\T.Hung\nnunet_v1\preprocessed", "User")
[System.Environment]::SetEnvironmentVariable("RESULTS_FOLDER", "C:\Users\T.Hung\nnunet_v1\results", "User")
New-Item -ItemType Directory -Force -Path "C:\Users\T.Hung\nnunet_v1\raw", "C:\Users\T.Hung\nnunet_v1\preprocessed", "C:\Users\T.Hung\nnunet_v1\results"
```

### 1.4 Tải pretrained models

```powershell
# ACDC (tim) — Task027
Invoke-WebRequest -Uri "https://zenodo.org/records/4003545/files/Task027_ACDC.zip?download=1" -OutFile "C:\Users\T.Hung\Downloads\Task027_ACDC.zip"
nnUNet_install_pretrained_model_from_zip "C:\Users\T.Hung\Downloads\Task027_ACDC.zip"

# LiTS (gan) — Task003
Invoke-WebRequest -Uri "https://zenodo.org/records/4003545/files/Task003_Liver.zip?download=1" -OutFile "C:\Users\T.Hung\Downloads\Task003_Liver.zip"
nnUNet_install_pretrained_model_from_zip "C:\Users\T.Hung\Downloads\Task003_Liver.zip"
```

---

## 2. Dataset & Pretrained Models

### 2.1 ACDC (Cardiac MRI)

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn | [humanheart-project.creatis.insa-lyon.fr](https://humanheart-project.creatis.insa-lyon.fr/database/) |
| Số lượng | 100 bệnh nhân, 200 frames (ED + ES) |
| Nhóm bệnh | DCM, HCM, MINF, NOR, RV (20/group) |
| Metrics | LV_EDV, LV_ESV, LV_EF, RV_EDV, RV_ESV, RV_EF, Myo_mass |
| Labels | 0=bg, 1=RV, 2=Myo, 3=LV |
| Dung lượng | ~2.3GB |
| Thư mục | `data_ACDC/training/patientXXX/` |

### 2.2 LiTS (Liver CT)

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn | Kaggle LiTS Challenge |
| Số lượng | 131 CT volumes + masks |
| Metrics | Liver volume (mL), Tumor volume (mL), Tumor burden (%) |
| Labels | 0=bg, 1=liver, 2=tumor |
| Dung lượng | ~50GB |
| Thư mục | `data_LiTS/images/` + `data_LiTS/masks/` |

---

## 3. Cách Chạy

### 3.1 ACDC

```powershell
cd D:\Hoc_Tap\NCKH

# Buoc 1: Chay nnU-Net inference (~10-15 phut, GPU)
python experiments/run_nnunet_inference.py --model 3d_fullres --save_npz

# Buoc 2: Tinh clinical metrics (<1 phut)
python experiments/compute_metrics.py

# Buoc 3: CRC-FS experiment — 8 methods, 5-fold CV (~5-10 phut, GPU)
python experiments/run_crc_fs_experiment.py
```

Kết quả: `results/crc_fs_results.csv`, `results/figures/*.png`

### 3.2 LiTS

```powershell
cd D:\Hoc_Tap\NCKH

# 1 lenh duy nhat — tu dong 4 buoc
python experiments/run_lits_pipeline.py
```

Pipeline tự động: prepare → inference → metrics → experiment

| Bước | Làm gì | Dùng | Thời gian |
|------|--------|------|-----------|
| prepare | Nén .nii → .nii.gz | CPU | ~20-30 phút |
| inference | nnU-Net 3d_lowres | GPU | ~30-60 phút |
| metrics | Tính liver/tumor volume | CPU | ~5-10 phút |
| experiment | 6 methods | GPU | ~15-30 phút |

### 3.3 Các file experiment

| File | Chạy khi nào | Methods |
|------|-------------|---------|
| `run_nnunet_inference.py` | Bước 1 (bắt buộc) | Chạy nnU-Net, sinh .npz |
| `compute_metrics.py` | Bước 2 (bắt buộc) | Tính metrics từ predictions |
| `run_crc_fs_experiment.py` | Bước 3 ACDC | 8 methods (SCP→CRC-FS-J) |
| `run_lits_pipeline.py` | Tất cả LiTS | 6 methods (tự động 4 bước) |
| `run_real_experiments.py` | Phụ | Baseline CP/CRC/Mondrian |
| `run_pipeline.py` | Dev | Test với simulated data |
| `01_explore_acdc.ipynb` | Khám phá | Notebook EDA |

---

## 4. Kiến Trúc Codebase

```
NCKH/
├── data_ACDC/training/       # Dataset ACDC
├── data_LiTS/images/         # Dataset LiTS (CT volumes)
├── data_LiTS/masks/          # Dataset LiTS (GT masks)
├── nnunet_input/             # Input ACDC cho nnU-Net
├── nnunet_input_lits/        # Input LiTS cho nnU-Net
├── nnunet_output/            # Output ACDC (.npz + .nii.gz)
├── nnunet_output_lits/       # Output LiTS (.npz + .nii.gz)
├── nnunet_data/              # Cấu hình nnU-Net
│
├── src/
│   ├── conformal/
│   │   ├── split_conformal.py    # Baseline Split CP + conformal_quantile()
│   │   ├── crc.py                # Baseline CRC (indicator loss)
│   │   ├── mondrian.py           # Group-conditional CP
│   │   ├── adaptive_scores.py    # Normalized scores + adaptive prediction
│   │   ├── compass.py            # COMPASS-L + compass_l_score_binary()
│   │   ├── compass_j.py          # COMPASS-J (Jacobian PCA) + shared subspace
│   │   └── crc_fs.py             # ★ CRC-FS: dual-calibration (SCP + CRC)
│   ├── metrics/
│   │   └── clinical_metrics.py   # Nonconformity scores
│   └── data/
│       ├── acdc_dataset.py       # ACDC loader
│       └── lits_dataset.py       # LiTS loader
│
├── experiments/
│   ├── run_nnunet_inference.py   # nnU-Net inference (ACDC)
│   ├── compute_metrics.py        # Tính clinical metrics (ACDC)
│   ├── run_crc_fs_experiment.py  # ★ 8 methods so sánh (ACDC)
│   ├── run_lits_pipeline.py      # ★ Pipeline LiTS (6 methods)
│   ├── run_real_experiments.py   # Baseline CP/CRC/Mondrian
│   ├── run_compass_inference.py  # COMPASS experiment cũ
│   ├── run_pipeline.py           # Test với simulated data
│   └── run_full_comparison.py    # So sánh baseline
│
├── notebooks/
│   ├── 01_explore_acdc.ipynb     # Notebook EDA
│   └── build_notebook.py         # Script build notebook
│
└── results/
    ├── acdc_metrics.csv          # GT + Predicted ACDC
    ├── crc_fs_results.csv        # ★ ACDC 8 methods (200 dòng)
    ├── lits_metrics.csv          # GT + Predicted LiTS
    ├── figures/
    │   ├── crc_fs_full_comparison.png
    │   └── crc_fs_pareto.png
    └── ...
```

---

## 5. Kết Quả Thực Nghiệm

### 5.1 ACDC — 8 methods (5-fold CV, LV Volume, Target 90%)

| Method | Coverage | Width (mL) | vs Split CP |
|--------|----------|------------|-------------|
| Split CP (Baseline) | 91.0% | 2.35 | — |
| CRC (Theorem 2.1) | 89.5% | 2.23 | -5.1% |
| Adaptive SCP | 91.5% | 2.55 | +8.6% |
| Adaptive CRC | 90.0% | 2.48 | +5.5% |
| COMPASS-L | 92.5% | 3.06 | +30.3% |
| COMPASS-J | 92.5% | 3.05 | +29.8% |
| **CRC-FS-L** | **92.5%** | **5.54** | +136.1% |
| **CRC-FS-J** | **92.5%** | **4.78** | +103.7% |

### 5.2 LiTS — 6 methods (Liver Volume, Target 90%)

| Method | Loại |
|--------|------|
| Split CP | Output-space |
| CRC | Output-space |
| Adaptive SCP | Output-space |
| Adaptive CRC | Output-space |
| COMPASS-L | Feature-space (binary search) |
| CRC-FS-L | Feature-space (SCP + adaptive) |

> COMPASS-J + CRC-FS-J chưa chạy do GPU 8GB không đủ cho Jacobian trên ảnh CT lớn.

---

## 6. Thiết Kế Thuật Toán CRC-FS

### 6.1 Ý tưởng

Kết hợp 3 bài báo thành 1 framework:
- **COMPASS** (ICLR 2026): Feature-space perturbation
- **CRC** (ICLR 2024): Finite-sample risk control với logistic bounded loss
- **Adaptive CP**: Normalized scores → adaptive per-sample width

### 6.2 Dual-Calibration Framework (Final)

```
CALIBRATION:
  For each sample i:
    R_i = compass_score(probs_i, y_true_i)    ← Feature sensitivity
    sigma_i = entropy(probs_i)                ← Model uncertainty
    R'_i = R_i / sigma_i                      ← Normalized score

  PRIMARY (SCP):
    alpha_cal = max(0, alpha - 1/(n+1))       ← Conservative correction
    q_hat = quantile({R'_i}, ceil((n+1)(1-alpha_cal))/n)

  SECONDARY (CRC):
    L_i(lambda) = R_i / (R_i + lambda)        ← Logistic bounded loss
    lambda_crc = inf{lambda: mean(L_i) <= alpha - (1-alpha)/n}

PREDICTION:
  beta_j = q_hat * sigma_j                    ← Adaptive width!
  interval = [m(-beta_j), m(+beta_j)]
```

### 6.3 Dual-Calibration

| | SCP (Primary) | CRC (Diagnostic) |
|---|---|---|
| Key | `q_hat` | `lambda_crc` |
| Dùng để | Tạo prediction interval | Finite-sample risk guarantee bổ sung |
| Loss | Quantile | Logistic R/(R+λ) ∈ [0,1] |
| Width | Adaptive (× sigma) | — |

### 6.4 Tính mới

| | Split CP | CRC Paper | COMPASS | **CRC-FS** |
|---|---|---|---|---|
| Feature perturbation | ❌ | ❌ | ✅ | ✅ |
| Finite-sample risk guarantee | ❌ | ✅ | ❌ | ✅ |
| Adaptive per-sample width | ❌ | ❌ | ❌ | ✅ |
| Logistic bounded loss ≠ indicator | ❌ | ❌ | ❌ | ✅ |
| Dual calibration (SCP + CRC) | ❌ | ❌ | ❌ | ✅ |

---

## 7. Quá Trình Phát Triển & Các Lỗi Đã Sửa

### 7.1 Các vấn đề phát hiện & sửa

| # | Vấn đề | File | Cách sửa |
|---|--------|------|----------|
| 1 | CRC = Split CP (indicator loss) | `crc.py` | Thêm logistic bounded loss trong CRC-FS |
| 2 | COMPASS-J không dùng shared PCA | `compass_j.py` | Thêm `compute_shared_directions()` + fallback |
| 3 | Coverage SCP = 89.5% < 90% | `split_conformal.py` | Conservative quantile: `alpha_cal = alpha - 1/(n+1)` |
| 4 | Path sai `data/training` → 0 samples | `run_crc_fs_experiment.py` | Sửa thành `data_ACDC/training` |
| 5 | CRC-FS v1 width 23.67 mL | `crc_fs.py` | Đổi sang logistic loss `R/(R+λ)` |
| 6 | CRC-FS v2 coverage 83% | `crc_fs.py` | Calibrate trên normalized scores |
| 7 | Unicode ★✓✗ crash Windows | Tất cả file | Thay bằng ASCII |
| 8 | LiTS OOM GPU 8GB | `run_lits_pipeline.py` | Float16 + empty_cache() |
| 9 | LiTS OOM RAM | `run_lits_pipeline.py` | Load tensor từng sample, slice-by-slice |
| 10 | LiTS grid search quá chậm | `compass.py` | Thêm `compass_l_score_binary()` |
| 11 | Hardlinks không giảm disk | `run_lits_pipeline.py` | Dùng nibabel load/save |
| 12 | Notebook lặp code 3 lần | `01_explore_acdc.ipynb` | Viết lại từ đầu 27 cells |
| 13 | Windows 10/11 detection sai | Notebook | Build ≥ 22000 → Windows 11 |

### 7.2 Vòng đời CRC-FS calibration

| Version | Calibration | Key | Width | Coverage |
|---------|------------|-----|-------|----------|
| v1 | CRC `min(1, R'_i/λ)` | `lambda_hat` | 23.67 mL | 100% |
| v2 | CRC logistic `R_i/(R_i+λ)` | `lambda_hat` | 9.36 mL | 98% |
| v2b | CRC + adaptive `β×σ/σ_med` | `lambda_hat` | 3.62 mL | 83% |
| **v3 Final** | **SCP normalized quantile** | **`q_hat`** | **5.54 mL** | **92.5%** |

CRC không bị bỏ — giữ vai trò diagnostic (`lambda_crc`) bên cạnh SCP primary (`q_hat`).

### 7.3 Troubleshooting nhanh

| Lỗi | Fix |
|-----|-----|
| `CUDA out of memory` | Float16 + `torch.cuda.empty_cache()` |
| `Loaded 0 valid samples` | Path là `data_ACDC/training`, không phải `data/training` |
| `KeyError: 'lambda_hat'` | Đã đổi → dùng `q_hat` |
| `Shapes differ across samples` | Fallback per-sample Jacobian |
| `UnicodeEncodeError` (★✓✗) | Đã thay ASCII |
| `PermissionError` file in use | Kill python.exe, xóa thư mục |
| `VRAM stuck 7.6GB` | `empty_cache()` sau mỗi sample |
| `.nii.gz file endings` | nnU-Net yêu cầu `.nii.gz`, không phải `.nii` |

---

## 8. Tài Liệu Tham Khảo

### Bài báo gốc trong đề cương
1. Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction* (2023)
2. Angelopoulos et al. — *Conformal Risk Control* (ICLR 2024)
3. Cheung et al. — *COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics* (ICLR 2026)

### Bài báo mới 2025-2026
4. Cheung et al. — *ConVOLT: Efficient Conformal Volumetry* (arXiv:2603.00798, 2026) — **cùng nhóm tác giả COMPASS**
5. Viti et al. — *CONSIGN: Conformal Segmentation via Spatial Groupings* (ICLR 2026)
6. Mossina & Friedrich — *Morphological Prediction Sets* (MICCAI 2025)
7. Tan et al. — *Conformal Lesion Segmentation for 3D Medical Images* (arXiv:2510.17897, 2025)
8. Luo & Zhou — *Conditional Conformal Risk Adaptation* (arXiv:2504.07611, 2025)
9. Prinster et al. — *Conformal Policy Control (gCRC)* (arXiv:2603.02196, 2026)
10. Yeh et al. — *Conformal Risk Training: End-to-End* (NeurIPS 2025)
11. Li et al. — *Confidence-Aware Ratio-based Biomarkers* (arXiv:2505.19585, 2025)
12. Pugliese et al. — *Uncertainty Estimation of Volumetric Measurements* (2025)
13. Guennoun et al. — *Segmenting with Confidence: Brain Tumor* (npj Digital Medicine, 2026)
14. CURVAS Challenge — *Multi-Rater Volume Assessment* (CBM, 2025)
15. Badjie et al. — *Semantic Segmentation with Conformal Risk Guarantees* (AEiC 2026)
16. Loaiza-Ganem et al. — *Conf-Gen* (ICML 2026)

### Nền tảng
17. Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* (2005)
18. Lei et al. — *Distribution-Free Predictive Inference for Regression* (2018)
