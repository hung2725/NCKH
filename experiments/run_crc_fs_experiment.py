"""
experiments/run_crc_fs_experiment.py

Đánh giá TOÀN BỘ 8 phương pháp Conformal Prediction trên ACDC:
  1. SCP        - Split Conformal Prediction (Baseline)
  2. CRC        - Conformal Risk Control (Theorem 2.1)
  3. ASCP       - Adaptive Split CP (Normalized scores)
  4. ACRC       - Adaptive CRC (Normalized + Binary search)
  5. COMPASS-L  - Feature Perturbation (Uniform logit shift)
  6. COMPASS-J  - Feature Perturbation (Jacobian PCA subspace) [FIXED]
  7. CRC-FS-L   - [NEW] CRC + COMPASS + Adaptive (Logit shift)
  8. CRC-FS-J   - [NEW] CRC + COMPASS + Adaptive (Jacobian PCA)

Mục tiêu: Coverage >= 90% + Width NHỎ NHẤT.
"""

import sys
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conformal.split_conformal import calibrate, predict_interval, evaluate_coverage
from src.conformal.crc import find_lambda
from src.conformal.adaptive_scores import (
    compute_uncertainty_from_probs,
    calibrate_normalized,
    predict_interval_normalized,
    find_lambda_adaptive,
)
from src.conformal.compass import (
    compass_l_score,
    calibrate_compass,
    predict_interval_compass,
)
from src.conformal.compass_j import (
    compute_volume_jacobian,
    compass_j_score,
    calibrate_compass_j,
    predict_interval_compass_j,
    compute_shared_directions,
    project_jacobian_to_subspace,
)
from src.conformal.crc_fs import (
    calibrate_crc_fs_l,
    predict_interval_crc_fs_l,
    calibrate_crc_fs_j,
    predict_interval_crc_fs_j,
)

#  Config 
WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
DATA_DIR      = WORKSPACE_DIR / "data_ACDC/training"
PRED_DIR      = WORKSPACE_DIR / "nnunet_output"
RESULTS_DIR   = WORKSPACE_DIR / "results"
FIG_DIR       = RESULTS_DIR / "figures"
ALPHA         = 0.1       # Target: 90% coverage
TARGET_CLASS  = 3         # Left Ventricle in ACDC
N_FOLDS       = 5
N_PCA         = 5         # PCA components for subspace methods
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Target coverage: {(1-ALPHA)*100:.0f}%")


#  1. Load data 

def load_all_samples():
    """Load all .npz + .nii.gz GT pairs."""
    npz_files = sorted(PRED_DIR.glob("*.npz"))
    if not npz_files:
        print("ERROR: No .npz files found. Run run_nnunet_inference.py --save_npz first.")
        sys.exit(1)

    print(f"Found {len(npz_files)} .npz files.")
    samples = []

    for npz_path in tqdm(npz_files, desc="Loading samples"):
        stem = npz_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        patient_id = parts[0]
        frame_id   = parts[1]
        gt_path    = DATA_DIR / patient_id / f"{patient_id}_{frame_id}_gt.nii.gz"

        if not gt_path.exists():
            continue

        data  = np.load(str(npz_path))
        probs = data['softmax'].astype(np.float32)

        gt_img   = nib.load(str(gt_path))
        gt_mask  = gt_img.get_fdata()
        spacing  = gt_img.header.get_zooms()
        voxel_vol = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0

        gt_binary = (gt_mask == TARGET_CLASS).astype(np.uint8)
        y_true    = float(np.sum(gt_binary)) * voxel_vol

        pred_mask  = np.argmax(probs, axis=0)
        pred_binary = (pred_mask == TARGET_CLASS).astype(np.uint8)
        y_pred     = float(np.sum(pred_binary)) * voxel_vol

        sigma = compute_uncertainty_from_probs(probs, TARGET_CLASS, method='entropy')

        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=DEVICE)

        def make_metric_fn(vvol):
            def metric_fn(binary_mask):
                if isinstance(binary_mask, torch.Tensor):
                    return float(binary_mask.sum().item()) * vvol
                return float(np.sum(binary_mask)) * vvol
            return metric_fn

        samples.append({
            'patient_id' : patient_id,
            'frame'      : frame_id,
            'y_true'     : y_true,
            'y_pred'     : y_pred,
            'sigma'      : sigma,
            'probs'      : probs_tensor,
            'voxel_vol'  : voxel_vol,
            'metric_fn'  : make_metric_fn(voxel_vol),
        })

    print(f"Loaded {len(samples)} valid samples.")
    return samples


#  2. Cross-Validation 

def run_kfold(samples):
    """5-fold CV comparing all 8 methods."""
    indices = np.arange(len(samples))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_results = []
    fold_summaries = []

    for fold_idx, (cal_idx, test_idx) in enumerate(kf.split(indices)):
        t_start = time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*60}")

        cal_samples  = [samples[i] for i in cal_idx]
        test_samples = [samples[i] for i in test_idx]

        n_cal = len(cal_samples)
        n_test = len(test_samples)

        # Arrays for standard methods
        cal_true   = np.array([s['y_true'] for s in cal_samples])
        cal_pred   = np.array([s['y_pred'] for s in cal_samples])
        cal_sigmas = np.array([s['sigma']  for s in cal_samples])
        cal_voxel_vols = np.array([s['voxel_vol'] for s in cal_samples])

        test_true  = np.array([s['y_true'] for s in test_samples])
        test_pred  = np.array([s['y_pred'] for s in test_samples])
        test_sigmas = np.array([s['sigma'] for s in test_samples])

        cal_probs  = [s['probs'] for s in cal_samples]
        cal_metric_fns = [s['metric_fn'] for s in cal_samples]

        #  SCP 
        scp = calibrate(cal_true, cal_pred, ALPHA, mode='abs')
        lo_scp, up_scp = predict_interval(test_pred, scp['q_hat'], mode='abs')
        ev_scp = evaluate_coverage(test_true, lo_scp, up_scp)

        #  CRC 
        crc = find_lambda(cal_true, cal_pred, ALPHA, mode='abs', n_grid=1000)
        lo_crc, up_crc = predict_interval(test_pred, crc['lambda'], mode='abs')
        ev_crc = evaluate_coverage(test_true, lo_crc, up_crc)

        #  ASCP (Adaptive SCP) 
        ascp = calibrate_normalized(cal_true, cal_pred, cal_sigmas, ALPHA)
        lo_ascp, up_ascp = predict_interval_normalized(test_pred, test_sigmas, ascp['q_hat'])
        ev_ascp = evaluate_coverage(test_true, lo_ascp, up_ascp)

        #  ACRC (Adaptive CRC) 
        acrc = find_lambda_adaptive(cal_true, cal_pred, cal_sigmas, ALPHA)
        lo_acrc, up_acrc = predict_interval_normalized(test_pred, test_sigmas, acrc['lambda'])
        ev_acrc = evaluate_coverage(test_true, lo_acrc, up_acrc)

        #  COMPASS-L 
        print("  Calibrating COMPASS-L...", flush=True)
        scores_l = []
        for i in tqdm(range(n_cal), desc="  COMPASS-L cal", leave=False):
            sc = compass_l_score(cal_probs[i], float(cal_true[i]),
                                 TARGET_CLASS, cal_metric_fns[i],
                                 b_max=5.0, steps=100)
            scores_l.append(sc)
        beta_l = calibrate_compass(np.array(scores_l), ALPHA)

        covs_l, wids_l = [], []
        for s in test_samples:
            lo, hi = predict_interval_compass(s['probs'], beta_l, TARGET_CLASS, s['metric_fn'])
            covs_l.append(1.0 if lo <= s['y_true'] <= hi else 0.0)
            wids_l.append(hi - lo)
        ev_cl = {'coverage': float(np.mean(covs_l)), 'mean_width': float(np.mean(wids_l))}

        #  COMPASS-J (FIXED: shared PCA subspace) ─
        print("  Calibrating COMPASS-J (shared PCA subspace)...", flush=True)
        V_L_j, cal_dirs_j = compute_shared_directions(
            cal_probs, TARGET_CLASS,
            [float(s['voxel_vol']) for s in cal_samples],
            n_components=N_PCA,
        )

        scores_j = []
        for i in tqdm(range(n_cal), desc="  COMPASS-J cal", leave=False):
            sc = compass_j_score(cal_probs[i], float(cal_true[i]),
                                 TARGET_CLASS, cal_metric_fns[i],
                                 cal_dirs_j[i], b_max=5.0, steps=100)
            scores_j.append(sc)
        beta_j = calibrate_compass_j(np.array(scores_j), ALPHA)

        covs_j, wids_j = [], []
        for s in test_samples:
            J_test = compute_volume_jacobian(s['probs'], TARGET_CLASS, s['voxel_vol'])
            if V_L_j is not None:
                dir_test = project_jacobian_to_subspace(J_test, V_L_j)
            else:
                # Fallback: per-sample direction
                dir_test = J_test
            abs_max = dir_test.abs().max()
            if abs_max > 1e-8:
                dir_test = dir_test / abs_max
            lo, hi = predict_interval_compass_j(
                s['probs'], beta_j, TARGET_CLASS, s['metric_fn'], dir_test,
            )
            covs_j.append(1.0 if lo <= s['y_true'] <= hi else 0.0)
            wids_j.append(hi - lo)
        ev_cj = {'coverage': float(np.mean(covs_j)), 'mean_width': float(np.mean(wids_j))}

        #  CRC-FS-L [NEW] (OUR CONTRIBUTION) 
        print("  Calibrating CRC-FS-L [NEW]...", flush=True)
        crcfsl = calibrate_crc_fs_l(
            cal_probs, cal_true, cal_sigmas,
            TARGET_CLASS, cal_metric_fns, ALPHA,
            b_max=5.0, steps=100,
        )
        covs_fsl, wids_fsl = [], []
        q_l = crcfsl['q_hat']
        med_l = crcfsl['median_sigma']
        for s in test_samples:
            lo, hi = predict_interval_crc_fs_l(
                s['probs'], q_l, s['sigma'], med_l,
                TARGET_CLASS, s['metric_fn'],
            )
            covs_fsl.append(1.0 if lo <= s['y_true'] <= hi else 0.0)
            wids_fsl.append(hi - lo)
        ev_fsl = {'coverage': float(np.mean(covs_fsl)), 'mean_width': float(np.mean(wids_fsl))}

        #  CRC-FS-J [NEW] (OUR CONTRIBUTION) 
        print("  Calibrating CRC-FS-J [NEW]...", flush=True)
        crcfsj = calibrate_crc_fs_j(
            cal_probs, cal_true, cal_sigmas, cal_voxel_vols,
            TARGET_CLASS, cal_metric_fns, ALPHA,
            b_max=5.0, steps=100, n_pca_components=N_PCA,
        )
        covs_fsj, wids_fsj = [], []
        V_L_fsj = crcfsj.get('V_L')
        q_j = crcfsj['q_hat']
        med_j = crcfsj['median_sigma']
        for s in test_samples:
            lo, hi = predict_interval_crc_fs_j(
                s['probs'], q_j, s['sigma'], med_j,
                TARGET_CLASS, s['metric_fn'], s['voxel_vol'],
                V_L_fsj,
            )
            covs_fsj.append(1.0 if lo <= s['y_true'] <= hi else 0.0)
            wids_fsj.append(hi - lo)
        ev_fsj = {'coverage': float(np.mean(covs_fsj)), 'mean_width': float(np.mean(wids_fsj))}

        #  Collect per-sample results ─
        for i, s in enumerate(test_samples):
            all_results.append({
                'fold': fold_idx + 1,
                'patient_id': s['patient_id'],
                'y_true': s['y_true'],
                'y_pred': s['y_pred'],
                'sigma': s['sigma'],
                # SCP
                'scp_covered': int(lo_scp[i] <= s['y_true'] <= up_scp[i]),
                'scp_width': float(up_scp[i] - lo_scp[i]),
                # CRC
                'crc_covered': int(lo_crc[i] <= s['y_true'] <= up_crc[i]),
                'crc_width': float(up_crc[i] - lo_crc[i]),
                # ASCP
                'ascp_covered': int(lo_ascp[i] <= s['y_true'] <= up_ascp[i]),
                'ascp_width': float(up_ascp[i] - lo_ascp[i]),
                # ACRC
                'acrc_covered': int(lo_acrc[i] <= s['y_true'] <= up_acrc[i]),
                'acrc_width': float(up_acrc[i] - lo_acrc[i]),
                # COMPASS-L
                'compassl_covered': covs_l[i],
                'compassl_width': wids_l[i],
                # COMPASS-J
                'compassj_covered': covs_j[i],
                'compassj_width': wids_j[i],
                # CRC-FS-L [NEW]
                'crcfsl_covered': covs_fsl[i],
                'crcfsl_width': wids_fsl[i],
                # CRC-FS-J [NEW]
                'crcfsj_covered': covs_fsj[i],
                'crcfsj_width': wids_fsj[i],
            })

        #  Fold summary 
        t_elapsed = time() - t_start
        fold_summaries.append({
            'fold': fold_idx + 1,
            'scp_cov': ev_scp['coverage'], 'scp_wid': ev_scp['mean_width'],
            'crc_cov': ev_crc['coverage'], 'crc_wid': ev_crc['mean_width'],
            'ascp_cov': ev_ascp['coverage'], 'ascp_wid': ev_ascp['mean_width'],
            'acrc_cov': ev_acrc['coverage'], 'acrc_wid': ev_acrc['mean_width'],
            'compassl_cov': ev_cl['coverage'], 'compassl_wid': ev_cl['mean_width'],
            'compassj_cov': ev_cj['coverage'], 'compassj_wid': ev_cj['mean_width'],
            'crcfsl_cov': ev_fsl['coverage'], 'crcfsl_wid': ev_fsl['mean_width'],
            'crcfsj_cov': ev_fsj['coverage'], 'crcfsj_wid': ev_fsj['mean_width'],
        })

        print(f"\n  Fold {fold_idx+1} Results (time: {t_elapsed:.0f}s):")
        print(f"  {'Method':<16} {'Coverage':>10} {'Width':>10}")
        print(f"  {'-'*36}")
        for name, cov, wid in [
            ('SCP', ev_scp['coverage'], ev_scp['mean_width']),
            ('CRC', ev_crc['coverage'], ev_crc['mean_width']),
            ('ASCP', ev_ascp['coverage'], ev_ascp['mean_width']),
            ('ACRC', ev_acrc['coverage'], ev_acrc['mean_width']),
            ('COMPASS-L', ev_cl['coverage'], ev_cl['mean_width']),
            ('COMPASS-J', ev_cj['coverage'], ev_cj['mean_width']),
            ('CRC-FS-L [NEW]', ev_fsl['coverage'], ev_fsl['mean_width']),
            ('CRC-FS-J [NEW]', ev_fsj['coverage'], ev_fsj['mean_width']),
        ]:
            marker = ' ' if cov >= (1-ALPHA) - 0.02 else ' '
            print(f"  {name:<16} {cov*100:>9.1f}% {wid:>9.2f} mL{marker}")

    return pd.DataFrame(all_results), pd.DataFrame(fold_summaries)


#  3. Summarize 

def print_summary(df: pd.DataFrame, fold_df: pd.DataFrame):
    """In bảng tổng hợp + phân tích thống kê."""
    methods = [
        ('scp',       'Split CP (Baseline)',      '#E74C3C'),
        ('crc',       'CRC (Theorem 2.1)',         '#3498DB'),
        ('ascp',      'Adaptive SCP',              '#E67E22'),
        ('acrc',      'Adaptive CRC',              '#9B59B6'),
        ('compassl',  'COMPASS-L (Logit Shift)',   '#1ABC9C'),
        ('compassj',  'COMPASS-J (Jacobian PCA)',  '#F39C12'),
        ('crcfsl',    'CRC-FS-L [NEW] (OURS)',         '#27AE60'),
        ('crcfsj',    'CRC-FS-J [NEW] (OURS)',         '#2C3E50'),
    ]

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS — ACDC LV Volume, Target Coverage = {(1-ALPHA)*100:.0f}%")
    print("=" * 80)
    print(f"{'Method':<35} {'Coverage':>10} {'Width':>10} {'vs SCP':>9} {'Status':>8}")
    print("-" * 80)

    scp_width = df['scp_width'].mean()
    scp_cov   = df['scp_covered'].mean()

    best_wid = float('inf')
    best_name = ""
    valid_methods = []

    for key, name, _ in methods:
        cov_col = f'{key}_covered'
        wid_col = f'{key}_width'
        if cov_col not in df.columns:
            continue

        cov = df[cov_col].mean() * 100
        wid = df[wid_col].mean()

        # Std from folds
        if key in fold_df.columns or f'{key}_cov' in fold_df.columns:
            fold_key_cov = f'{key}_cov' if f'{key}_cov' in fold_df.columns else key
            fold_key_wid = f'{key}_wid' if f'{key}_wid' in fold_df.columns else key

        diff = (wid - scp_width) / scp_width * 100
        status = 'VALID' if cov >= (1-ALPHA)*100 - 1 else 'LOW'
        marker = 'BEST' if wid < best_wid and cov >= (1-ALPHA)*100 - 1 else ''
        if wid < best_wid and cov >= (1-ALPHA)*100 - 1:
            best_wid = wid
            best_name = name

        print(f"  {name:<33} {cov:>8.1f}% {wid:>9.2f} mL {diff:>+8.1f}% {status:>8}{marker}")
        valid_methods.append((name, cov, wid, diff, status.strip(), key))

    print("-" * 80)
    print(f"  [NEW] Best valid method: {best_name} (width={best_wid:.2f} mL)")
    print("=" * 80)

    # Statistical comparison: CRC-FS vs baselines
    print("\n--- Statistical Analysis (paired per-sample) ---")
    for our_key, our_name in [('crcfsl', 'CRC-FS-L'), ('crcfsj', 'CRC-FS-J')]:
        if f'{our_key}_width' not in df.columns:
            continue
        our_wids = df[f'{our_key}_width'].values
        scp_wids = df['scp_width'].values
        cl_wids  = df['compassl_width'].values

        delta_scp = scp_wids - our_wids  # positive = ours better
        delta_cl  = cl_wids - our_wids

        pct_better_scp = float(np.mean(delta_scp > 0)) * 100
        pct_better_cl  = float(np.mean(delta_cl > 0)) * 100
        mean_improve_scp = float(np.mean(delta_scp))
        mean_improve_cl  = float(np.mean(delta_cl))

        print(f"  {our_name} vs Split CP:  {pct_better_scp:.0f}% samples narrower, "
              f"mean Δwidth = {mean_improve_scp:+.2f} mL")
        print(f"  {our_name} vs COMPASS-L: {pct_better_cl:.0f}% samples narrower, "
              f"mean Δwidth = {mean_improve_cl:+.2f} mL")

    return valid_methods


#  4. Plot 

def plot_comparison(df: pd.DataFrame):
    """Biểu đồ so sánh 8 methods."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    methods = [
        ('scp',       'Split CP',        '#E74C3C'),
        ('crc',       'CRC',             '#3498DB'),
        ('ascp',      'Adaptive\nSCP',   '#E67E22'),
        ('acrc',      'Adaptive\nCRC',   '#9B59B6'),
        ('compassl',  'COMPASS-L',       '#1ABC9C'),
        ('compassj',  'COMPASS-J',       '#F39C12'),
        ('crcfsl',    'CRC-FS-L\n[NEW]',     '#27AE60'),
        ('crcfsj',    'CRC-FS-J\n[NEW]',     '#2C3E50'),
    ]
    methods = [(k, n, c) for k, n, c in methods if f'{k}_width' in df.columns]
    labels = [n for _, n, _ in methods]
    colors = [c for _, _, c in methods]

    covs = [df[f'{k}_covered'].mean() * 100 for k, _, _ in methods]
    wids = [df[f'{k}_width'].mean() for k, _, _ in methods]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('CRC-FS: Conformal Risk Control in Feature Space\n'
                 f'ACDC LV Volume — 5-Fold CV — Target {(1-ALPHA)*100:.0f}% Coverage',
                 fontsize=13, fontweight='bold')

    # Coverage
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(labels)), covs, color=colors, alpha=0.9,
                    edgecolor='black', linewidth=0.8)
    ax1.axhline((1-ALPHA)*100, color='red', linestyle='--', linewidth=1.5,
                label=f'Target {(1-ALPHA)*100:.0f}%')
    ax1.set_ylabel('Empirical Coverage (%)', fontsize=12)
    ax1.set_ylim(min(min(covs)-5, 80), max(max(covs)+5, 100))
    ax1.set_title('Coverage Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    for bar, val in zip(bars1, covs):
        color = '#27AE60' if val >= (1-ALPHA)*100 else '#E74C3C'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                 fontweight='bold', color=color)

    # Width
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(labels)), wids, color=colors, alpha=0.9,
                    edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Average Interval Width (mL)', fontsize=12)
    ax2.set_title('Interval Width (Smaller = Better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, max(wids) * 1.2)
    ax2.grid(axis='y', linestyle=':', alpha=0.4)
    best_wid = min(wids)
    for bar, val in zip(bars2, wids):
        color = '#27AE60' if abs(val - best_wid) < 1e-6 else '#333333'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(wids)*0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold', color=color)

    # Highlight our methods
    for i, (key, _, _) in enumerate(methods):
        if 'crcfs' in key:
            for ax in axes:
                ax.get_children()[i].set_edgecolor('#FFD700')
                ax.get_children()[i].set_linewidth(2.5)

    plt.tight_layout()
    out_path = FIG_DIR / "crc_fs_full_comparison.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved -> {out_path}")

    #  Extra: Width vs Coverage scatter 
    fig2, ax = plt.subplots(figsize=(10, 7))

    for key, name, color in methods:
        cov = df[f'{key}_covered'].mean() * 100
        wid = df[f'{key}_width'].mean()
        size = 250 if 'crcfs' in key else 120
        edge = '#FFD700' if 'crcfs' in key else 'black'
        lw = 3 if 'crcfs' in key else 1
        ax.scatter(wid, cov, c=color, s=size, edgecolors=edge, linewidth=lw,
                   label=name, zorder=5 if 'crcfs' in key else 3)

    ax.axhline((1-ALPHA)*100, color='red', linestyle='--', linewidth=1.5,
               label=f'Target {(1-ALPHA)*100:.0f}%')
    ax.set_xlabel('Average Width (mL) — Smaller → Better', fontsize=12)
    ax.set_ylabel('Coverage (%) — Higher → Better', fontsize=12)
    ax.set_title('CRC-FS: Optimal Point = Top-Left Corner\n(Best Coverage + Smallest Width)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier arrow
    best_idx = np.argmin(wids)
    ax.annotate('← OPTIMAL', xy=(wids[best_idx], covs[best_idx]),
                xytext=(wids[best_idx] + max(wids)*0.1, covs[best_idx] + 2),
                fontsize=10, fontweight='bold', color='#27AE60',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

    plt.tight_layout()
    out_path2 = FIG_DIR / "crc_fs_pareto.png"
    plt.savefig(str(out_path2), dpi=300, bbox_inches='tight')
    print(f"Figure saved -> {out_path2}")


# Main
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Load
    samples = load_all_samples()
    # 2. K-Fold CV
    df, fold_df = run_kfold(samples)
    out_csv = RESULTS_DIR / "crc_fs_results.csv"
    df.to_csv(str(out_csv), index=False)
    print(f"\nCSV saved -> {out_csv}")
    fold_csv = RESULTS_DIR / "crc_fs_fold_summary.csv"
    fold_df.to_csv(str(fold_csv), index=False)
    # 4. Summary
    print_summary(df, fold_df)
    # 5. Plot
    plot_comparison(df)
    print("\n Experiment complete!")
