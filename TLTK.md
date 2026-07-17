## A. CÁC BÀI BÁO GỐC TRONG ĐỀ CƯƠNG

| # | Bài báo | Hội nghị | Vai trò |
|---|---------|----------|---------|
| 1 | Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction* | Tutorial | Nền tảng CP |
| 2 | Angelopoulos et al. — *Conformal Risk Control* | ICLR 2024 | Khung CRC |
| 3 | Cheung et al. — *COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics* | ICLR 2026 | Feature-space CP |

---

## B. CÁC BÀI BÁO MỚI 2025-2026 LIÊN QUAN TRỰC TIẾP

### Nhóm 1: Feature-Space & Metric-Level Conformal Prediction

#### 1.1 ConVOLT — Efficient Conformal Volumetry for Template-Based Segmentation
**Tác giả:** Matt Y. Cheung, Ashok Veeraraghavan, Guha Balakrishnan (Rice University)
**Đăng tại:** arXiv:2603.00798 (02/2026)
**Code:** [github.com/matthewyccheung/convolt](https://github.com/matthewyccheung/convolt)

**Điểm quan trọng:** CÙNG NHÓM TÁC GIẢ VỚI COMPASS! Đây là paper tiếp nối trực tiếp từ COMPASS.

**Ý tưởng:** Áp dụng CP cho template-based segmentation (dùng deformable registration thay vì deep learning). Học multiplicative scaling factor từ deformation-field features (Jacobian statistics, displacement magnitude) rồi calibrate bằng split conformal.

**Kết quả chính:**
- Target coverage 90% đạt được, interval hẹp hơn nhiều so với output-space baselines
- ThoraxCBCT: 1,222 mL vs 2,707-4,631 mL baselines
- OASIS brain MRI: 105 mL vs 130-159 mL baselines

**Tại sao quan trọng với bạn:** Đây là follow-up work của COMPASS, cho thấy nhóm tác giả đang mở rộng hướng này. Bạn cần đọc kỹ để tránh trùng lặp và tìm điểm khác biệt.

---

#### 1.2 Beyond Segmentation: Confidence-Aware and Debiased Estimation of Ratio-based Biomarkers
**Tác giả:** Li et al. (KU Leuven)
**Đăng tại:** arXiv:2505.19585 (2025)

**Ý tưởng:** Phân rã prediction interval cho ratio-based biomarkers (ví dụ: necrosis-to-tumor ratio) thành:
- Estimation-based bounds (Taylor expansion của expected squared error)
- Calibration-based bounds (dùng calibration error làm upper bound cho volume bias)

**Tại sao quan trọng:** Phương pháp decompose uncertainty → liên quan trực tiếp đến clinical metrics như EF (EDV/ESV ratio).

---

### Nhóm 2: Spatial-Aware & Morphological Conformal Prediction

#### 2.1 CONSIGN — Conformal Segmentation Informed by Spatial Groupings via Decomposition
**Tác giả:** Viti, Karabelas & Holler
**Đăng tại:** ICLR 2026 (Poster)
**Link:** [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007344)

**Ý tưởng:** Kết hợp spatial correlations giữa các pixels vào CP (pixel-wise CP bỏ qua spatial structure). Tương thích với mọi pre-trained segmentation model. Đánh giá trên 3 medical imaging datasets + 2 COCO subsets.

**Tại sao quan trọng:** ICLR 2026 — cùng venue với COMPASS. Hướng tiếp cận khác (spatial thay vì feature-space) → có thể so sánh/kết hợp.

---

#### 2.2 Morphological Prediction Sets for Segmentation (MICCAI 2025)
**Tác giả:** Mossina & Friedrich
**Đăng tại:** MICCAI 2025, LNCS Vol. 15963, pp. 78-88
**Code:** [github.com/deel-ai-papers/consema](https://github.com/deel-ai-papers/consema)

**Ý tưởng:** Dùng mathematical morphology (dilation) để tạo nested prediction sets cho binary segmentation. Ưu điểm: chỉ cần predicted binary mask, không cần logits/confidence scores. Dilation margin size = interpretable uncertainty indicator.

**Tại sao quan trọng:** Approach đơn giản, model-agnostic, có thể dùng làm baseline.

---

#### 2.3 Conformal Lesion Segmentation (CLS) for 3D Medical Images
**Tác giả:** Tan, Wang, Duan, Xu, Shen & Shen
**Đăng tại:** arXiv:2510.17897 (2025)

**Ý tưởng:** Kiểm soát False Negative Rate (FNR) trong 3D lesion segmentation. Định nghĩa FNR-specific nonconformity score dựa trên per-sample critical thresholds.

**Kết quả:** Validate trên 6 3D medical imaging datasets, 5 backbone models (Med3D, nnUNet, UNETR, Swin-UNETR, SAM-Med3D).

**Tại sao quan trọng:** Dùng nnU-Net trong experiments → có thể so sánh trực tiếp. Focus vào FNR control (clinical relevance).

---

### Nhóm 3: Conditional & Adaptive Risk Control

#### 3.1 Conditional Conformal Risk Adaptation (CRA)
**Tác giả:** Luo & Zhou
**Đăng tại:** arXiv:2504.07611 (04/2025)

**Ý tưởng:** Giải quyết vấn đề CRC chỉ có marginal guarantee, trong khi conditional risk (per-image) thay đổi nhiều. Đề xuất:
- CRA: Adaptive prediction sets lấy cảm hứng từ APS (Adaptive Prediction Sets)
- CCRA: Calibrated CRA kết hợp probability calibration
- CCRA-S: Stratified variant với group-conditional approach

**Validate trên:** Polyp segmentation tasks.

**Tại sao quan trọng:** Đây chính là vấn đề group-conditional coverage mà bạn đang gặp với Mondrian! CRA family cung cấp giải pháp cho conditional risk consistency.

---

#### 3.2 Conformal Policy Control (CPC)
**Tác giả:** Prinster, Fannjiang, Park, Cho, Liu, Saria, Stanton
**Đăng tại:** arXiv:2603.02196 (07/2026)

**Điểm quan trọng:** Giới thiệu **generalized CRC (gCRC)** cho **non-monotonic bounded loss functions** — giải quyết bài toán mở trong CRC trước đây.

**Ý tưởng:**
- Dùng conformal calibration trên safe-policy data
- Ứng dụng: medical QA (kiểm soát FDR), constrained active learning, biomolecular sequence optimization

**Tại sao quan trọng:** gCRC mở rộng CRC ra ngoài monotonic loss → CRC-FS của bạn có thể tận dụng framework này để có theoretical contribution mạnh hơn.

---

### Nhóm 4: End-to-End & Training-Based CRC

#### 4.1 Conformal Risk Training: End-to-End Optimization of Conformal Risk Control
**Tác giả:** Yeh, Christianson, Wierman, Yue
**Đăng tại:** NeurIPS 2025

**Ý tưởng:** Khắc phục hạn chế của post-hoc CRC bằng cách **differentiate qua conformal OCE risk control** trong quá trình training/fine-tuning. Mở rộng ra ngoài expected loss đến tail risks như CVaR.

**Ứng dụng:** Kiểm soát false negative rate của classifier, financial risk.

**Tại sao quan trọng:** Cho thấy hướng đi mới: CRC không chỉ post-hoc mà có thể end-to-end. Nếu bạn muốn fine-tune nnU-Net với CRC loss, đây là paper gốc.

---

### Nhóm 5: Uncertainty Quantification cho Medical Metrics

#### 5.1 Uncertainty Estimation of AI-Driven Volumetric Measurements in CT Scans
**Tác giả:** Pugliese et al.
**Đăng tại:** ScienceDirect (2025)

**Ý tưởng:** Monte Carlo sampling ở sub-cluster level (cluster voxel-wise probability scores dùng spatial derivatives) → credibility intervals.

**Kết quả:** Median interval width 70 mL (24% relative), ECE = 2×10⁻⁴, Spearman correlation 0.81 giữa tumor volume và interval width.

**Tại sao quan trọng:** Cách đo uncertainty khác với CP — có thể dùng làm baseline hoặc kết hợp.

---

#### 5.2 Segmenting with Confidence — Brain Tumor Imaging
**Tác giả:** Guennoun et al. (UCSF)
**Đăng tại:** npj Digital Medicine (2026)

**Ý tưởng:** Evidential deep learning ensembles (1,655 MRIs) → aleatoric + epistemic uncertainty → volumetric credible intervals.

**Kết quả:** Median Dice 0.93, external validation 353 patients (Dice 0.92). Uncertainty maps align với neuroradiologist-identified ambiguous regions.

**Tại sao quan trọng:** Uncertainty map tương quan với ý kiến bác sĩ → liên quan đến Research Question 3 của bạn.

---

#### 5.3 CURVAS Challenge Results — Multi-Rater Volume Assessment
**Đăng tại:** Computers in Biology and Medicine (2025)

**Ý tưởng:** Benchmark calibration và uncertainty dưới multi-rater uncertainty cho pancreas, kidney, liver CT.

**Kết luận chính:** Well-calibrated models có correlation mạnh nhất giữa confidence và accuracy. ECE và CRPS là metrics quan trọng.

**Tại sao quan trọng:** Dataset multi-rater → liên quan đến annotator disagreement analysis.

---

### Nhóm 6: Semantic Segmentation với CRC

#### 6.1 Model-Agnostic Uncertainty-Aware Semantic Segmentation with Conformal Risk Guarantees
**Tác giả:** Badjie et al.
**Đăng tại:** AEiC 2026

**Ý tưởng:** Pipeline model-agnostic cho autonomous navigation, tương thích DINOv2, Mask2Former, SegFormer. Dùng evidential deep learning + pixel-wise, class-conditional split-conformal calibration.

**Đánh giá trên:** Lisbon street scene (LiSS) dataset + COCO.

---

## C. BẢNG TỔNG HỢP THEO HƯỚNG NGHIÊN CỨU

| Hướng | Bài báo | Venue | Code |
|-------|---------|-------|------|
| **Feature-space CP** | COMPASS | ICLR 2026 | Không public? |
| **Template CP** | ConVOLT | arXiv 2026 | [github.com/matthewyccheung/convolt](https://github.com/matthewyccheung/convolt) |
| **Spatial CP** | CONSIGN | ICLR 2026 | — |
| **Morphological CP** | Morphological Sets | MICCAI 2025 | [github.com/deel-ai-papers/consema](https://github.com/deel-ai-papers/consema) |
| **FNR Control 3D** | CLS | arXiv 2025 | — |
| **Conditional CRC** | CRA / CCRA | arXiv 2025 | — |
| **Generalized CRC** | CPC | arXiv 2026 | — |
| **End-to-end CRC** | Conformal Risk Training | NeurIPS 2025 | — |
| **Ratio Biomarkers** | Confidence-Aware Ratio | arXiv 2025 | — |
| **Volumetric UQ** | MC Sub-cluster | ScienceDirect 2025 | — |
| **Evidential DL** | Brain Tumor UQ | npj Digital Medicine 2026 | — |
| **Multi-Rater** | CURVAS Challenge | CBM 2025 | — |
| **Seg + CRC** | Semantic Seg CRC | AEiC 2026 | — |

---

## D. PHÂN TÍCH: BẠN CÓ THỂ ĐÓNG GÓP GÌ MỚI?

### Khoảng trống #1: Chưa ai kết hợp CRC + COMPASS
- COMPASS (ICLR 2026) dùng Split CP đơn thuần
- ConVOLT (2026) cũng dùng Split CP
- CRC papers (NeurIPS 2025, ICML 2026) chưa áp dụng cho medical imaging
- **→ CRC-FS của bạn là đóng góp thực sự!**

### Khoảng trống #2: Chưa có adaptive normalization trong feature-space CP
- COMPASS dùng chung 1 β cho mọi sample
- CRA (2025) làm adaptive ở output space, chưa vào feature space
- **→ Adaptive width trong CRC-FS là novelty**

### Khoảng trống #3: Generalized CRC (non-monotonic loss) chưa áp dụng cho segmentation
- CPC (2026) giới thiệu gCRC framework
- Chưa ai áp dụng gCRC cho medical image segmentation metrics
- **→ Có thể extend CRC-FS dùng gCRC framework**

### Khoảng trống #4: Chưa có multi-dataset comparison giữa feature-space CP methods
- COMPASS: 4 tasks (skin, polyp, thyroid, CRC)
- ConVOLT: 3 datasets (lung CT, ThoraxCBCT, brain MRI)
- **→ Bạn có thể benchmark trên ACDC + LiTS + KiTS + LIDC-IDRI**

### Khoảng trống #5: Annotator disagreement vs interval width
- CURVAS có multi-rater nhưng không link với CP interval width
- **→ Research Question 3 của bạn vẫn chưa ai làm**

---

## E. RECOMMENDED READING ORDER

1. **COMPASS** (ICLR 2026) — bài gốc, nắm chắc phương pháp
2. **ConVOLT** (arXiv 2026) — follow-up của cùng nhóm tác giả, hiểu hướng mở rộng
3. **CRA/CCRA** (arXiv 2025) — conditional risk control, liên quan đến Mondrian
4. **CPC** (arXiv 2026) — generalized CRC với non-monotonic loss
5. **Conformal Risk Training** (NeurIPS 2025) — end-to-end CRC
6. **CONSIGN** (ICLR 2026) — spatial approach để so sánh
7. **CLS** (arXiv 2025) — FNR control, dùng nnU-Net
8. **CURVAS Challenge** (2025) — multi-rater calibration benchmark

---

## F. CÁCH CRC-FS CỦA BẠN KHÁC BIỆT

| | COMPASS | ConVOLT | CRA | **CRC-FS (BẠN)** |
|---|---|---|---|---|
| Feature perturbation | ✅ | ✅ (deformation) | ❌ | ✅ |
| CRC calibration | ❌ (SCP) | ❌ (SCP) | ✅ (output) | ✅ |
| Adaptive per-sample | ❌ | ❌ | ✅ | ✅ |
| Bounded loss ≠ indicator | ❌ | ❌ | ✅ | ✅ (logistic) |
| Medical volumes | ✅ | ✅ | ❌ (polyp seg) | ✅ (ACDC) |
| Multi-dataset plan | ✅ (4 tasks) | ✅ (3 datasets) | ❌ | ✅ (4 datasets planned) |


## G. KẾT LUẬN

Hướng nghiên cứu của bạn là **đúng thời điểm và có giá trị**. Các bài báo 2025-2026 xác nhận:
1. Feature-space CP đang là hot topic (COMPASS, ConVOLT)
2. Conditional/adaptive CRC đang phát triển mạnh (CRA, CPC, Conformal Risk Training)
3. Việc kết hợp CRC + COMPASS + Adaptive là genuinely novel
4. Chưa ai benchmark trên ACDC + LiTS + KiTS + LIDC-IDRI với framework này

**Việc cần làm ngay:**
- Đọc kỹ ConVOLT (cùng nhóm tác giả COMPASS) để tránh overlap
- Đọc CRA để cải thiện phần Mondrian/conditional coverage
- Đọc CPC để cân nhắc dùng gCRC framework cho phần lý thuyết
