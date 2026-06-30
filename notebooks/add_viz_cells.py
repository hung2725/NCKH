"""
Script them cac cells visualize anh MRI vao notebook 01_explore_acdc.ipynb
"""
import json, nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
from pathlib import Path

NB_PATH = Path(__file__).parent / "01_explore_acdc.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb_obj = nbformat.read(f, as_version=4)

# ============================================================
# CELL MD 1 — Header
# ============================================================
md1 = new_markdown_cell(
    "## Visualize: MRI + Ground Truth + nnU-Net Prediction\n\n"
    "So sanh truc quan:\n"
    "- **Anh MRI goc** (grayscale)\n"
    "- **Ground Truth mask** (tu annotation bac si)\n"
    "- **Predicted mask** (tu nnU-Net pretrained)\n"
    "- **Error map** (pixel sai khac giua GT vs Pred)\n\n"
    "Labels: `0`=Background, `1`=RV (do), `2`=Myocardium (xanh la), `3`=LV (xanh duong)"
)

# ============================================================
# CELL 1 — Imports + setup
# ============================================================
c1 = new_code_cell("""\
import sys, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

DATA_DIR   = Path("../data/training")
PRED_DIR   = Path("../nnunet_output")
RESULT_CSV = Path("../results/acdc_metrics.csv")

# Color map cho mask: 0=bg(trong suot), 1=RV(do), 2=Myo(xanh la), 3=LV(xanh duong)
CMAP_COLORS = np.array([
    [1.0, 1.0, 1.0, 0.0],   # 0: background
    [1.0, 0.0, 0.0, 0.65],  # 1: RV
    [0.0, 0.85, 0.0, 0.65], # 2: Myocardium
    [0.1, 0.4, 1.0, 0.65],  # 3: LV
])

def overlay_mask(ax, mri_slice, mask_slice, title=""):
    \"\"\"Hien thi anh MRI voi overlay mask mau.\"\"\"
    ax.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="auto",
              vmin=np.percentile(mri_slice, 2), vmax=np.percentile(mri_slice, 98))
    rgba = CMAP_COLORS[mask_slice.astype(int).T]
    ax.imshow(rgba, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

legend_patches = [
    mpatches.Patch(color=(1,0,0,0.8),    label="RV (Right Ventricle)"),
    mpatches.Patch(color=(0,0.85,0,0.8), label="Myocardium"),
    mpatches.Patch(color=(0.1,0.4,1,0.8),label="LV (Left Ventricle)"),
]

def get_frames(patient_id):
    \"\"\"Doc file Info.cfg de lay so frame ED va ES.\"\"\"
    with open(DATA_DIR / patient_id / "Info.cfg") as f:
        lines = f.readlines()
    info = {}
    for l in lines:
        if ":" in l:
            k, v = l.split(":", 1)
            info[k.strip()] = v.strip()
    ed = f"frame{int(info.get('ED', 1)):02d}"
    es = f"frame{int(info.get('ES', 12)):02d}"
    group = info.get("Group", "?")
    return ed, es, group

print("Setup OK — ready to visualize!")
print(f"Dataset: {len(list(DATA_DIR.iterdir()))} patients")
print(f"Predictions: {len(list(PRED_DIR.glob('*.nii.gz')))} files")
""")

# ============================================================
# CELL 2 — Hàm visualize 1 bệnh nhân
# ============================================================
c2 = new_code_cell("""\
def visualize_patient(patient_id, frame="frame01", mid_slice=None, figsize=(14, 4)):
    \"\"\"
    Hien thi 4 panel: MRI | GT | Predicted | Error map
    cho 1 benh nhan, 1 frame, 1 slice.
    \"\"\"
    mri_path  = DATA_DIR / patient_id / f"{patient_id}_{frame}.nii.gz"
    gt_path   = DATA_DIR / patient_id / f"{patient_id}_{frame}_gt.nii.gz"
    pred_path = PRED_DIR / f"{patient_id}_{frame}.nii.gz"

    mri  = nib.load(mri_path).get_fdata()
    gt   = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata() if pred_path.exists() else None

    n_slices = mri.shape[2]
    mid = mid_slice if mid_slice is not None else n_slices // 2

    _, _, group = get_frames(patient_id)
    has_pred = pred is not None

    fig, axes = plt.subplots(1, 4 if has_pred else 2, figsize=figsize)
    fig.suptitle(
        f"{patient_id}  |  Group: {group}  |  {frame}  |  Slice {mid}/{n_slices-1}",
        fontsize=12, fontweight="bold", y=1.01
    )

    # Panel 1: MRI only
    axes[0].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=np.percentile(mri[:,:,mid], 2),
                   vmax=np.percentile(mri[:,:,mid], 98))
    axes[0].set_title("MRI goc", fontsize=10)
    axes[0].axis("off")

    # Panel 2: GT
    overlay_mask(axes[1], mri[:,:,mid], gt[:,:,mid], "Ground Truth (bac si)")

    if has_pred:
        # Panel 3: Predicted
        overlay_mask(axes[2], mri[:,:,mid], pred[:,:,mid], "nnU-Net Predicted")

        # Panel 4: Error map
        error = (gt[:,:,mid] != pred[:,:,mid]).astype(float)
        pixel_acc = 1 - error.mean()
        axes[3].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                       vmin=np.percentile(mri[:,:,mid], 2),
                       vmax=np.percentile(mri[:,:,mid], 98))
        axes[3].imshow(error.T, cmap="Reds", alpha=0.7,
                       origin="lower", aspect="auto", vmin=0, vmax=1)
        axes[3].set_title(f"Error map  |  pixel acc={pixel_acc:.3f}", fontsize=10)
        axes[3].axis("off")

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.show()

print("Function visualize_patient() da san sang!")
""")

# ============================================================
# CELL 3 — 1 đại diện mỗi nhóm bệnh
# ============================================================
md2 = new_markdown_cell(
    "### 1 benh nhan dai dien moi nhom benh (ED frame)\n\n"
    "| Nhom | Mo ta |\n"
    "|------|-------|\n"
    "| **DCM** | Dilated Cardiomyopathy — tim gian no, EF thap |\n"
    "| **HCM** | Hypertrophic Cardiomyopathy — thanh tim day |\n"
    "| **MINF** | Myocardial Infarction — nhoi mau co tim |\n"
    "| **NOR** | Normal — tim binh thuong |\n"
    "| **RV** | Abnormal Right Ventricle — RV bat thuong |"
)

c3 = new_code_cell("""\
# 1 benh nhan dai dien moi nhom (patient ID tuong ung nhom trong ACDC)
demo_patients = [
    ("patient001", "DCM  — Dilated Cardiomyopathy"),
    ("patient021", "HCM  — Hypertrophic Cardiomyopathy"),
    ("patient041", "MINF — Myocardial Infarction"),
    ("patient061", "NOR  — Normal"),
    ("patient081", "RV   — Abnormal Right Ventricle"),
]

for pid, desc in demo_patients:
    ed, es, group = get_frames(pid)
    print(f"\\n{'='*60}")
    print(f"{desc}  ({pid})")
    print(f"  ED={ed}, ES={es}")
    visualize_patient(pid, ed)
""")

# ============================================================
# CELL 4 — So sánh ED vs ES
# ============================================================
md3 = new_markdown_cell(
    "### So sanh ED vs ES — cung 1 benh nhan\n\n"
    "- **ED (End-Diastole)**: Tim phong to nhat — volume lon nhat (EDV)\n"
    "- **ES (End-Systole)**: Tim co lai nhat — volume nho nhat (ESV)\n"
    "- **EF = (EDV - ESV) / EDV × 100%** — ty le mau tim bom ra moi chu ky\n"
    "  - EF binh thuong: 55–70%\n"
    "  - EF thap (<40%): suy tim nang"
)

c4 = new_code_cell("""\
pid = "patient001"  # DCM - EF thap
ed, es, group = get_frames(pid)

df_metrics = pd.read_csv(RESULT_CSV)
row = df_metrics[df_metrics["patient_id"] == pid].iloc[0]
print(f"{pid} (Group: {group})")
print(f"  LV_EDV: GT={row['LV_EDV_gt']:.1f} mL  |  Pred={row['LV_EDV_pred']:.1f} mL")
print(f"  LV_ESV: GT={row['LV_ESV_gt']:.1f} mL  |  Pred={row['LV_ESV_pred']:.1f} mL")
print(f"  LV_EF:  GT={row['LV_EF_gt']:.1f}%   |  Pred={row['LV_EF_pred']:.1f}%")
print()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"{pid} ({group}) — So sanh ED vs ES", fontsize=13, fontweight="bold")

for row_idx, (frame, label) in enumerate([(ed, "ED (End-Diastole)"), (es, "ES (End-Systole)")]):
    mri  = nib.load(DATA_DIR / pid / f"{pid}_{frame}.nii.gz").get_fdata()
    gt   = nib.load(DATA_DIR / pid / f"{pid}_{frame}_gt.nii.gz").get_fdata()
    pred = nib.load(PRED_DIR / f"{pid}_{frame}.nii.gz").get_fdata()
    mid  = mri.shape[2] // 2

    vmin = np.percentile(mri[:,:,mid], 2)
    vmax = np.percentile(mri[:,:,mid], 98)

    axes[row_idx, 0].imshow(mri[:,:,mid].T, cmap="gray", origin="lower",
                            aspect="auto", vmin=vmin, vmax=vmax)
    axes[row_idx, 0].set_title(f"{label}\\nMRI", fontsize=10)
    axes[row_idx, 0].axis("off")

    overlay_mask(axes[row_idx, 1], mri[:,:,mid], gt[:,:,mid],   f"{label}\\nGround Truth")
    overlay_mask(axes[row_idx, 2], mri[:,:,mid], pred[:,:,mid], f"{label}\\nnnU-Net Pred")

    error = (gt[:,:,mid] != pred[:,:,mid]).astype(float)
    axes[row_idx, 3].imshow(mri[:,:,mid].T, cmap="gray", origin="lower",
                            aspect="auto", vmin=vmin, vmax=vmax)
    axes[row_idx, 3].imshow(error.T, cmap="Reds", alpha=0.7, origin="lower", aspect="auto")
    axes[row_idx, 3].set_title(f"{label}\\nError map", fontsize=10)
    axes[row_idx, 3].axis("off")

fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.show()
""")

# ============================================================
# CELL 5 — Sweep tất cả slices
# ============================================================
md4 = new_markdown_cell(
    "### Quet toan bo slices chieu doc (basal → apical)\n\n"
    "Tim nhìn tu day tim xuong dinh tim — moi slice la 1 mat cat ngang."
)

c5 = new_code_cell("""\
pid = "patient061"  # NOR group - tim binh thuong
ed, es, group = get_frames(pid)
mri  = nib.load(DATA_DIR / pid / f"{pid}_{ed}.nii.gz").get_fdata()
gt   = nib.load(DATA_DIR / pid / f"{pid}_{ed}_gt.nii.gz").get_fdata()
pred = nib.load(PRED_DIR / f"{pid}_{ed}.nii.gz").get_fdata()

n_slices = mri.shape[2]
fig, axes = plt.subplots(3, n_slices, figsize=(n_slices * 2.8, 7))
fig.suptitle(f"{pid} ({group}) — {ed} — Tat ca {n_slices} slices (basal to apical)",
             fontsize=12, fontweight="bold")

row_labels = ["MRI goc", "Ground Truth", "nnU-Net Predicted"]
for s in range(n_slices):
    vmin = np.percentile(mri[:,:,s], 2)
    vmax = np.percentile(mri[:,:,s], 98)

    axes[0, s].imshow(mri[:,:,s].T, cmap="gray", origin="lower",
                      aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, s].set_title(f"S{s}", fontsize=9)

    overlay_mask(axes[1, s], mri[:,:,s], gt[:,:,s],   "")
    overlay_mask(axes[2, s], mri[:,:,s], pred[:,:,s], "")

    for r in range(3):
        axes[r, s].axis("off")

for r, lbl in enumerate(row_labels):
    axes[r, 0].set_ylabel(lbl, fontsize=9, labelpad=4)

fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
plt.show()
""")

# ============================================================
# CELL 6 — Worst cases
# ============================================================
md5 = new_markdown_cell(
    "### Worst cases — nnU-Net sai nhat (LV_EF error cao nhat)\n\n"
    "Nhung truong hop nay la kho nhat de predict — thuong la vi:\n"
    "- Hinh dang tim bat thuong (HCM)\n"
    "- Chat luong anh kem (noise cao)\n"
    "- Ranh gioi cau truc mo (nhoi mau)"
)

c6 = new_code_cell("""\
df_metrics = pd.read_csv(RESULT_CSV)
df_metrics["LV_EF_error"] = (df_metrics["LV_EF_pred"] - df_metrics["LV_EF_gt"]).abs()

worst = df_metrics.nlargest(5, "LV_EF_error")[
    ["patient_id", "group", "LV_EF_gt", "LV_EF_pred", "LV_EF_error"]
].reset_index(drop=True)

print("Top 5 worst cases (LV Ejection Fraction error):")
display(worst.style
    .format({"LV_EF_gt": "{:.2f}%", "LV_EF_pred": "{:.2f}%", "LV_EF_error": "{:.2f}%"})
    .background_gradient(subset="LV_EF_error", cmap="Reds")
)

for _, row_data in worst.iterrows():
    pid = row_data["patient_id"]
    ed, es, group = get_frames(pid)
    print(f"\\n{pid} ({row_data['group']}) — "
          f"GT={row_data['LV_EF_gt']:.1f}%  Pred={row_data['LV_EF_pred']:.1f}%  "
          f"Error={row_data['LV_EF_error']:.1f}%")
    visualize_patient(pid, ed)
""")

# ============================================================
# CELL 7 — Best cases
# ============================================================
md6 = new_markdown_cell(
    "### Best cases — nnU-Net chinh xac nhat (LV_EF error thap nhat)"
)

c7 = new_code_cell("""\
best = df_metrics.nsmallest(5, "LV_EF_error")[
    ["patient_id", "group", "LV_EF_gt", "LV_EF_pred", "LV_EF_error"]
].reset_index(drop=True)

print("Top 5 best cases (LV Ejection Fraction error):")
display(best.style
    .format({"LV_EF_gt": "{:.2f}%", "LV_EF_pred": "{:.2f}%", "LV_EF_error": "{:.2f}%"})
    .background_gradient(subset="LV_EF_error", cmap="Greens_r")
)

for _, row_data in best.iterrows():
    pid = row_data["patient_id"]
    ed, es, group = get_frames(pid)
    print(f"\\n{pid} ({row_data['group']}) — "
          f"GT={row_data['LV_EF_gt']:.1f}%  Pred={row_data['LV_EF_pred']:.1f}%  "
          f"Error={row_data['LV_EF_error']:.2f}%")
    visualize_patient(pid, ed)
""")

# ============================================================
# CELL 8 — Phân phối error theo nhóm
# ============================================================
md7 = new_markdown_cell(
    "### Phan bo sai so nnU-Net theo nhom benh\n\n"
    "Xem nhom nao nnU-Net predict chinh xac hon / kem hon."
)

c8 = new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Phan bo sai so nnU-Net theo nhom benh", fontsize=13, fontweight="bold")

metrics_to_plot = [
    ("LV_EF_error", "LV EF Error (%)"),
    ("LV_EDV_pred", "LV EDV Predicted (mL)"),
    ("RV_EF_pred",  "RV EF Predicted (%)"),
]

df_metrics["LV_EF_error"] = (df_metrics["LV_EF_pred"] - df_metrics["LV_EF_gt"]).abs()

groups_order = ["NOR", "DCM", "HCM", "MINF", "RV"]
colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]

for ax, (col, label) in zip(axes, metrics_to_plot):
    data_by_group = [df_metrics[df_metrics["group"] == g][col].dropna().values
                     for g in groups_order]
    bp = ax.boxplot(data_by_group, labels=groups_order, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Nhom benh")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()
""")

# ============================================================
# Thêm tất cả cells vào notebook
# ============================================================
new_cells = [md1, c1, c2, md2, c3, md3, c4, md4, c5, md5, c6, md6, c7, md7, c8]
nb_obj.cells.extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb_obj, f)

print(f"Done! Them {len(new_cells)} cells moi vao notebook.")
print(f"File: {NB_PATH}")
