"""
Tao notebook 01_explore_acdc.ipynb voi output hinh anh nhung san.
Chay code truc tiep, capture matplotlib figures, embed vao .ipynb cells.
"""
import sys, io, base64, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "training"
PRED_DIR   = ROOT / "nnunet_output"
RESULT_CSV = ROOT / "results" / "acdc_metrics.csv"
NB_OUT     = ROOT / "notebooks" / "01_explore_acdc.ipynb"

# ── helpers ─────────────────────────────────────────────────
CMAP_COLORS = np.array([
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.65],
    [0.0, 0.85, 0.0, 0.65],
    [0.1, 0.4,  1.0, 0.65],
])

legend_patches = [
    mpatches.Patch(color=(1,0,0,0.8),     label="RV"),
    mpatches.Patch(color=(0,0.85,0,0.8),  label="Myocardium"),
    mpatches.Patch(color=(0.1,0.4,1,0.8), label="LV"),
]

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def overlay_mask(ax, mri_s, mask_s, title=""):
    vmin = np.percentile(mri_s, 2); vmax = np.percentile(mri_s, 98)
    ax.imshow(mri_s.T, cmap="gray", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.imshow(CMAP_COLORS[mask_s.astype(int).T], origin="lower", aspect="auto")
    ax.set_title(title, fontsize=9); ax.axis("off")

def get_frames(pid):
    with open(DATA_DIR / pid / "Info.cfg") as f:
        info = {l.split(":")[0].strip(): l.split(":",1)[1].strip()
                for l in f if ":" in l}
    return (f"frame{int(info.get('ED',1)):02d}",
            f"frame{int(info.get('ES',12)):02d}",
            info.get("Group","?"))

def load_trio(pid, frame):
    mri  = nib.load(DATA_DIR / pid / f"{pid}_{frame}.nii.gz").get_fdata()
    gt   = nib.load(DATA_DIR / pid / f"{pid}_{frame}_gt.nii.gz").get_fdata()
    pred = nib.load(PRED_DIR / f"{pid}_{frame}.nii.gz").get_fdata()
    return mri, gt, pred

def make_png_output(b64):
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "image/png": b64,
            "text/plain": ["<Figure>"]
        }
    }

def make_text_output(text):
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": [text]
    }

def make_html_output(html, plain):
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "text/html": [html],
            "text/plain": [plain]
        }
    }

# ── build cells ─────────────────────────────────────────────
cells = []

# ══════════════════════════════════════════════════════════
# Section 1 — Gioi thieu dataset
# ══════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
    "# ACDC Cardiac MRI — Visualize Dataset & nnU-Net Predictions\n\n"
    "Notebook nay so sanh:\n"
    "- **Anh MRI goc** (cardiac MRI, dinh dang NIfTI)\n"
    "- **Ground Truth mask** (annotation tu bac si)\n"
    "- **nnU-Net Predicted mask** (pretrained model, khong retrain)\n"
    "- **Error map** (pixel sai khac)\n\n"
    "**Labels:** `0`=Background | `1`=RV (do) | `2`=Myocardium (xanh la) | `3`=LV (xanh duong)\n\n"
    "**Dataset:** ACDC — 100 benh nhan, 5 nhom: DCM / HCM / MINF / NOR / RV"
))

# ── Cell: thong ke dataset ──────────────────────────────────
df = pd.read_csv(RESULT_CSV)
n  = len(df)
grp_counts = df["group"].value_counts().to_dict()
txt = (f"Dataset: {n} benh nhan\n"
       f"Nhom: {grp_counts}\n"
       f"CSV columns: {list(df.columns)}\n")

c_stat = new_code_cell("# Thong ke dataset\ndf = pd.read_csv('../results/acdc_metrics.csv')\nprint(df.shape)\ndf.head()")
c_stat["outputs"] = [make_text_output(txt)]
cells.append(c_stat)

# ══════════════════════════════════════════════════════════
# Section 2 — 1 benh nhan dai dien moi nhom
# ══════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
    "## 1. Mot benh nhan dai dien moi nhom benh (ED frame)\n\n"
    "| Nhom | N | Mo ta |\n"
    "|------|---|-------|\n"
    "| DCM  | 20 | Dilated Cardiomyopathy — buong tim gian to, EF thap |\n"
    "| HCM  | 20 | Hypertrophic Cardiomyopathy — thanh co tim day bat thuong |\n"
    "| MINF | 20 | Myocardial Infarction — nhoi mau co tim |\n"
    "| NOR  | 20 | Normal — tim binh thuong |\n"
    "| RV   | 20 | Abnormal Right Ventricle — that phai bat thuong |"
))

demo = [("patient001","DCM"), ("patient021","HCM"),
        ("patient041","MINF"), ("patient061","NOR"), ("patient081","RV")]

for pid, grp_name in demo:
    ed, es, group = get_frames(pid)
    mri, gt, pred = load_trio(pid, ed)
    mid = mri.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    fig.suptitle(f"{pid}  |  Group: {group}  |  {ed}  |  Slice {mid}",
                 fontsize=12, fontweight="bold")

    vmin = np.percentile(mri[:,:,mid], 2); vmax = np.percentile(mri[:,:,mid], 98)
    axes[0].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax)
    axes[0].set_title("MRI goc"); axes[0].axis("off")
    overlay_mask(axes[1], mri[:,:,mid], gt[:,:,mid],   "Ground Truth (bac si)")
    overlay_mask(axes[2], mri[:,:,mid], pred[:,:,mid], "nnU-Net Predicted")

    error = (gt[:,:,mid] != pred[:,:,mid]).astype(float)
    axes[3].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax)
    axes[3].imshow(error.T, cmap="Reds", alpha=0.7, origin="lower", aspect="auto")
    axes[3].set_title(f"Error map | acc={1-error.mean():.3f}"); axes[3].axis("off")

    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    b64 = fig_to_b64(fig)
    src = f"# {grp_name} — {pid}\nvisualize_patient('{pid}', '{ed}')"
    c = new_code_cell(src)
    c["outputs"] = [make_png_output(b64)]
    cells.append(c)
    print(f"  Done: {pid} ({grp_name})")

# ══════════════════════════════════════════════════════════
# Section 3 — ED vs ES so sanh
# ══════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
    "## 2. ED vs ES — So sanh cuoi tam truong va cuoi tam thu\n\n"
    "- **ED (End-Diastole):** Tim phong to nhat — do day buong tim (EDV)\n"
    "- **ES (End-Systole):** Tim co lai nhat — the tich con lai (ESV)\n"
    "- **EF = (EDV − ESV) / EDV × 100%**\n"
    "  - Binh thuong: 55–70% | Suy tim: <40%"
))

for pid_idx, (pid_ed_es, grp_lbl) in enumerate([
        ("patient001", "DCM — EF thap (~25%)"),
        ("patient061", "NOR — EF binh thuong (~60%)")]):
    ed, es, group = get_frames(pid_ed_es)
    row = df[df["patient_id"] == pid_ed_es].iloc[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        f"{pid_ed_es} ({group}) — {grp_lbl}\n"
        f"LV_EDV: {row['LV_EDV_gt']:.1f} mL (GT) vs {row['LV_EDV_pred']:.1f} mL (Pred) | "
        f"LV_EF: {row['LV_EF_gt']:.1f}% (GT) vs {row['LV_EF_pred']:.1f}% (Pred)",
        fontsize=11, fontweight="bold"
    )
    for r, (frame, label) in enumerate([(ed,"ED (End-Diastole)"), (es,"ES (End-Systole)")]):
        mri, gt, pred = load_trio(pid_ed_es, frame)
        mid = mri.shape[2] // 2
        vmin = np.percentile(mri[:,:,mid],2); vmax = np.percentile(mri[:,:,mid],98)
        axes[r,0].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                         vmin=vmin, vmax=vmax)
        axes[r,0].set_title(f"{label}\nMRI goc"); axes[r,0].axis("off")
        overlay_mask(axes[r,1], mri[:,:,mid], gt[:,:,mid],   f"{label}\nGround Truth")
        overlay_mask(axes[r,2], mri[:,:,mid], pred[:,:,mid], f"{label}\nnnU-Net Pred")
        err = (gt[:,:,mid] != pred[:,:,mid]).astype(float)
        axes[r,3].imshow(mri[:,:,mid].T, cmap="gray", origin="lower",
                         aspect="auto", vmin=vmin, vmax=vmax)
        axes[r,3].imshow(err.T, cmap="Reds", alpha=0.7, origin="lower", aspect="auto")
        axes[r,3].set_title(f"{label}\nError map"); axes[r,3].axis("off")

    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    b64 = fig_to_b64(fig)
    c = new_code_cell(f"# ED vs ES — {pid_ed_es} ({group})\ncompare_ED_ES('{pid_ed_es}')")
    c["outputs"] = [make_png_output(b64)]
    cells.append(c)
    print(f"  Done ED/ES: {pid_ed_es}")

# ══════════════════════════════════════════════════════════
# Section 4 — Sweep slices
# ══════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
    "## 3. Quet toan bo slices (basal → apical)\n\n"
    "Moi slice la 1 mat cat ngang qua tim, tu day tim (basal) xuong dinh tim (apical)."
))

for pid_sweep, lbl_sweep in [("patient061","NOR"), ("patient001","DCM")]:
    ed, es, group = get_frames(pid_sweep)
    mri, gt, pred = load_trio(pid_sweep, ed)
    n_slices = mri.shape[2]

    fig, axes = plt.subplots(3, n_slices, figsize=(n_slices * 2.6, 6.5))
    fig.suptitle(f"{pid_sweep} ({group}) — {ed} — {n_slices} slices",
                 fontsize=12, fontweight="bold")
    for s in range(n_slices):
        vmin = np.percentile(mri[:,:,s],2); vmax = np.percentile(mri[:,:,s],98)
        axes[0,s].imshow(mri[:,:,s].T, cmap="gray", origin="lower",
                         aspect="auto", vmin=vmin, vmax=vmax)
        axes[0,s].set_title(f"S{s}", fontsize=8); axes[0,s].axis("off")
        overlay_mask(axes[1,s], mri[:,:,s], gt[:,:,s],   "")
        overlay_mask(axes[2,s], mri[:,:,s], pred[:,:,s], "")

    for r, lbl in enumerate(["MRI", "GT", "Pred"]):
        axes[r,0].set_ylabel(lbl, fontsize=9)

    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    b64 = fig_to_b64(fig)
    c = new_code_cell(f"# Sweep slices — {pid_sweep} ({group})\nslice_sweep('{pid_sweep}')")
    c["outputs"] = [make_png_output(b64)]
    cells.append(c)
    print(f"  Done sweep: {pid_sweep}")

# ══════════════════════════════════════════════════════════
# Section 5 — Worst / Best cases
# ══════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
    "## 4. Worst Cases — nnU-Net sai nhat (LV_EF error cao nhat)"
))
df["LV_EF_error"] = (df["LV_EF_pred"] - df["LV_EF_gt"]).abs()
worst = df.nlargest(4, "LV_EF_error")

for _, row_data in worst.iterrows():
    pid = row_data["patient_id"]
    ed, es, group = get_frames(pid)
    mri, gt, pred = load_trio(pid, ed)
    mid = mri.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    err_val = row_data["LV_EF_error"]
    fig.suptitle(
        f"{pid} ({group}) — GT_EF={row_data['LV_EF_gt']:.1f}%  "
        f"Pred_EF={row_data['LV_EF_pred']:.1f}%  Error={err_val:.1f}%  ← WORST",
        fontsize=11, fontweight="bold", color="darkred"
    )
    vmin = np.percentile(mri[:,:,mid],2); vmax = np.percentile(mri[:,:,mid],98)
    axes[0].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax); axes[0].set_title("MRI goc"); axes[0].axis("off")
    overlay_mask(axes[1], mri[:,:,mid], gt[:,:,mid],   "Ground Truth")
    overlay_mask(axes[2], mri[:,:,mid], pred[:,:,mid], "nnU-Net Pred")
    error = (gt[:,:,mid] != pred[:,:,mid]).astype(float)
    axes[3].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax)
    axes[3].imshow(error.T, cmap="Reds", alpha=0.7, origin="lower", aspect="auto")
    axes[3].set_title(f"Error map"); axes[3].axis("off")
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5,-0.06))
    b64 = fig_to_b64(fig)
    c = new_code_cell(f"# WORST: {pid} ({group}) — EF error={err_val:.1f}%")
    c["outputs"] = [make_png_output(b64)]
    cells.append(c)
    print(f"  Done worst: {pid}")

cells.append(new_markdown_cell(
    "## 5. Best Cases — nnU-Net chinh xac nhat (LV_EF error thap nhat)"
))
best = df.nsmallest(4, "LV_EF_error")

for _, row_data in best.iterrows():
    pid = row_data["patient_id"]
    ed, es, group = get_frames(pid)
    mri, gt, pred = load_trio(pid, ed)
    mid = mri.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    err_val = row_data["LV_EF_error"]
    fig.suptitle(
        f"{pid} ({group}) — GT_EF={row_data['LV_EF_gt']:.1f}%  "
        f"Pred_EF={row_data['LV_EF_pred']:.1f}%  Error={err_val:.2f}%  ✓ BEST",
        fontsize=11, fontweight="bold", color="darkgreen"
    )
    vmin = np.percentile(mri[:,:,mid],2); vmax = np.percentile(mri[:,:,mid],98)
    axes[0].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax); axes[0].set_title("MRI goc"); axes[0].axis("off")
    overlay_mask(axes[1], mri[:,:,mid], gt[:,:,mid],   "Ground Truth")
    overlay_mask(axes[2], mri[:,:,mid], pred[:,:,mid], "nnU-Net Pred")
    error = (gt[:,:,mid] != pred[:,:,mid]).astype(float)
    axes[3].imshow(mri[:,:,mid].T, cmap="gray", origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax)
    axes[3].imshow(error.T, cmap="Greens", alpha=0.5, origin="lower", aspect="auto")
    axes[3].set_title(f"Error map (nearly zero!)"); axes[3].axis("off")
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5,-0.06))
    b64 = fig_to_b64(fig)
    c = new_code_cell(f"# BEST: {pid} ({group}) — EF error={err_val:.3f}%")
    c["outputs"] = [make_png_output(b64)]
    cells.append(c)
    print(f"  Done best: {pid}")

# ══════════════════════════════════════════════════════════
# Section 6 — Boxplot error theo nhom
# ══════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
    "## 6. Phan bo sai so LV_EF theo nhom benh\n\n"
    "Nhom nao nnU-Net du bao chinh xac hon / kem hon?"
))
groups_order = ["NOR","DCM","HCM","MINF","RV"]
colors_box   = ["#2ecc71","#e74c3c","#3498db","#f39c12","#9b59b6"]
data_by_grp  = [df[df["group"]==g]["LV_EF_error"].values for g in groups_order]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bp = axes[0].boxplot(data_by_grp, labels=groups_order, patch_artist=True)
for patch, col in zip(bp["boxes"], colors_box):
    patch.set_facecolor(col); patch.set_alpha(0.75)
axes[0].set_title("LV_EF Error (%) theo nhom benh", fontsize=11)
axes[0].set_ylabel("|GT_EF - Pred_EF| (%)"); axes[0].grid(axis="y", linestyle=":", alpha=0.5)

data_ef_gt   = [df[df["group"]==g]["LV_EF_gt"].values   for g in groups_order]
data_ef_pred = [df[df["group"]==g]["LV_EF_pred"].values for g in groups_order]
x = np.arange(len(groups_order)); w = 0.35
axes[1].bar(x-w/2, [np.mean(d) for d in data_ef_gt],   w, label="GT",   color="#2c3e50", alpha=0.8)
axes[1].bar(x+w/2, [np.mean(d) for d in data_ef_pred], w, label="Pred", color="#e74c3c", alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(groups_order)
axes[1].set_title("LV_EF trung binh: GT vs Predicted", fontsize=11)
axes[1].set_ylabel("LV EF (%)"); axes[1].legend(); axes[1].grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
b64 = fig_to_b64(fig)
c = new_code_cell("# Boxplot error theo nhom\nplot_error_by_group()")
c["outputs"] = [make_png_output(b64)]
cells.append(c)
print("  Done boxplot")

# ── Luu notebook ──────────────────────────────────────────
nb = new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}
}
with open(NB_OUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"\nXong! Notebook da luu: {NB_OUT}")
print(f"Tong so cells: {len(cells)}")
