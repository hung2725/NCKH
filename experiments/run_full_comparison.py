"""
experiments/run_full_comparison.py

Đánh giá toàn bộ 5 phương pháp Conformal Prediction trên ACDC:
  1. SCP   - Split Conformal Prediction (Baseline)
  2. CRC   - Conformal Risk Control (binary search - Theorem 2.1)
  3. MSCP  - Mondrian (Group-Conditional) Split CP
  4. ASCP  - Adaptive (Normalized) Split CP [MỚI]
  5. COMPASS-J - Jacobian Feature Perturbation [MỚI]

Mục tiêu: Coverage >= 90% VÀ Width nhỏ nhất có thể.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conformal.split_conformal import calibrate, predict_interval, evaluate_coverage
from src.conformal.crc import find_lambda
from src.conformal.adaptive_scores import (
    compute_uncertainty_from_probs,
    calibrate_normalized,
    predict_interval_normalized,
    find_lambda_adaptive,
)
from src.conformal.compass_j import (
    compute_volume_jacobian,
    compass_j_score,
    calibrate_compass_j,
    predict_interval_compass_j,
)
from src.conformal.compass import (
    compass_l_score,
    calibrate_compass,
    predict_interval_compass,
)

# ─── Config ───────────────────────────────────────────────────────────────────
WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
DATA_DIR      = WORKSPACE_DIR / "data/training"
PRED_DIR      = WORKSPACE_DIR / "nnunet_output"
RESULTS_DIR   = WORKSPACE_DIR / "results"
FIG_DIR       = RESULTS_DIR / "figures"
ALPHA         = 0.1       # Target: 90% coverage
TARGET_CLASS  = 3         # Left Ventricle in ACDC
N_FOLDS       = 5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


# ─── 1. Load data ─────────────────────────────────────────────────────────────

def load_all_samples():
    """Load all .npz + .nii.gz GT pairs."""
    npz_files = sorted(PRED_DIR.glob("*.npz"))
    if not npz_files:
        print("ERROR: No .npz files found. Run run_nnunet_inference.py --save_npz first.")
        sys.exit(1)

    print(f"Found {len(npz_files)} .npz files.")
    samples = []

    for npz_path in tqdm(npz_files, desc="Loading samples"):
        stem = npz_path.stem            # e.g., patient001_frame01
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        patient_id = parts[0]
        frame_id   = parts[1]
        gt_path    = DATA_DIR / patient_id / f"{patient_id}_{frame_id}_gt.nii.gz"

        if not gt_path.exists():
            continue

        # Load probabilities
        data  = np.load(str(npz_path))
        probs = data['softmax'].astype(np.float32)   # (C, H, W, D)

        # Load GT mask
        gt_img   = nib.load(str(gt_path))
        gt_mask  = gt_img.get_fdata()
        spacing  = gt_img.header.get_zooms()         # (dH, dW, dD) mm
        voxel_vol = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0  # mL

        # Compute GT LV volume
        gt_binary = (gt_mask == TARGET_CLASS).astype(np.uint8)
        y_true    = float(np.sum(gt_binary)) * voxel_vol

        # Compute predicted LV volume (no perturbation)
        pred_mask  = np.argmax(probs, axis=0)
        pred_binary = (pred_mask == TARGET_CLASS).astype(np.uint8)
        y_pred     = float(np.sum(pred_binary)) * voxel_vol

        # Uncertainty estimate (from softmax entropy)
        sigma = compute_uncertainty_from_probs(probs, TARGET_CLASS, method='entropy')

        # GPU tensor for COMPASS
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=DEVICE)

        # Metric fn (closure over voxel_vol)
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
            'spacing'    : spacing,
        })

    print(f"Loaded {len(samples)} valid samples.")
    return samples


# ─── 2. Compute Jacobian direction for each sample ────────────────────────────

def compute_directions(samples):
    """Compute Jacobian perturbation direction for each sample."""
    for s in samples:
        J = compute_volume_jacobian(s['probs'], TARGET_CLASS, s['voxel_vol'])
        # Normalize to unit vector
        norm = J.norm()
        s['direction'] = J / max(norm.item(), 1e-8)
    return samples


# ─── 3. Cross-Validation Evaluation ──────────────────────────────────────────

def run_kfold(samples):
    """Run 5-fold CV comparing all methods."""
    indices = np.arange(len(samples))
    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    results = []

    for fold_idx, (cal_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        cal_samples  = [samples[i] for i in cal_idx]
        test_samples = [samples[i] for i in test_idx]

        # Arrays for standard methods
        cal_true   = np.array([s['y_true'] for s in cal_samples])
        cal_pred   = np.array([s['y_pred'] for s in cal_samples])
        cal_sigmas = np.array([s['sigma']  for s in cal_samples])
        test_true  = np.array([s['y_true'] for s in test_samples])
        test_pred  = np.array([s['y_pred'] for s in test_samples])
        test_sigmas = np.array([s['sigma'] for s in test_samples])

        # ── SCP (Baseline) ──────────────────────────────────────────────────
        scp_result = calibrate(cal_true, cal_pred, ALPHA, mode='abs')
        lower_scp, upper_scp = predict_interval(test_pred, scp_result['q_hat'], mode='abs')
        ev_scp = evaluate_coverage(test_true, lower_scp, upper_scp)

        # ── CRC (Binary Search - Theorem 2.1) ───────────────────────────────
        crc_result = find_lambda(cal_true, cal_pred, ALPHA, mode='abs', n_grid=1000)
        lower_crc, upper_crc = predict_interval(test_pred, crc_result['lambda'], mode='abs')
        ev_crc = evaluate_coverage(test_true, lower_crc, upper_crc)

        # ── Adaptive SCP (Normalized scores) ────────────────────────────────
        ascp_result = calibrate_normalized(cal_true, cal_pred, cal_sigmas, ALPHA)
        lower_ascp, upper_ascp = predict_interval_normalized(
            test_pred, test_sigmas, ascp_result['q_hat']
        )
        ev_ascp = evaluate_coverage(test_true, lower_ascp, upper_ascp)

        # ── Adaptive CRC (Binary search + normalized loss) ───────────────────
        acrc_result = find_lambda_adaptive(cal_true, cal_pred, cal_sigmas, ALPHA)
        lower_acrc, upper_acrc = predict_interval_normalized(
            test_pred, test_sigmas, acrc_result['lambda']
        )
        ev_acrc = evaluate_coverage(test_true, lower_acrc, upper_acrc)

        # ── COMPASS-L ────────────────────────────────────────────────────────
        print("  Calibrating COMPASS-L...", flush=True)
        scores_l = []
        for s in tqdm(cal_samples, desc="  COMPASS-L cal", leave=False):
            sc = compass_l_score(
                s['probs'], s['y_true'], TARGET_CLASS,
                s['metric_fn'], b_max=5.0, steps=100
            )
            scores_l.append(sc)
        beta_hat_l = calibrate_compass(np.array(scores_l), ALPHA)

        covs_l, wids_l = [], []
        for s in test_samples:
            lo, hi = predict_interval_compass(
                s['probs'], beta_hat_l, TARGET_CLASS, s['metric_fn']
            )
            covs_l.append(1.0 if lo <= s['y_true'] <= hi else 0.0)
            wids_l.append(hi - lo)
        ev_compassl = {
            'coverage'   : float(np.mean(covs_l)),
            'mean_width' : float(np.mean(wids_l)),
        }

        # ── COMPASS-J ────────────────────────────────────────────────────────
        print("  Calibrating COMPASS-J...", flush=True)
        scores_j = []
        for s in tqdm(cal_samples, desc="  COMPASS-J cal", leave=False):
            sc = compass_j_score(
                s['probs'], s['y_true'], TARGET_CLASS,
                s['metric_fn'], s['direction'],
                b_max=5.0, steps=100,
            )
            scores_j.append(sc)
        beta_hat_j = calibrate_compass_j(np.array(scores_j), ALPHA)

        covs_j, wids_j = [], []
        for s in test_samples:
            lo, hi = predict_interval_compass_j(
                s['probs'], beta_hat_j, TARGET_CLASS,
                s['metric_fn'], s['direction'],
            )
            covs_j.append(1.0 if lo <= s['y_true'] <= hi else 0.0)
            wids_j.append(hi - lo)
        ev_compassj = {
            'coverage'   : float(np.mean(covs_j)),
            'mean_width' : float(np.mean(wids_j)),
        }

        # Collect results for this fold
        for i, s in enumerate(test_samples):
            results.append({
                'fold'          : fold_idx + 1,
                'patient_id'    : s['patient_id'],
                'y_true'        : s['y_true'],
                'y_pred'        : s['y_pred'],
                # SCP
                'scp_covered'   : int(lower_scp[i] <= s['y_true'] <= upper_scp[i]),
                'scp_lower'     : float(lower_scp[i]),
                'scp_upper'     : float(upper_scp[i]),
                'scp_width'     : float(upper_scp[i] - lower_scp[i]),
                # CRC
                'crc_covered'   : int(lower_crc[i] <= s['y_true'] <= upper_crc[i]),
                'crc_lower'     : float(lower_crc[i]),
                'crc_upper'     : float(upper_crc[i]),
                'crc_width'     : float(upper_crc[i] - lower_crc[i]),
                # ASCP
                'ascp_covered'  : int(lower_ascp[i] <= s['y_true'] <= upper_ascp[i]),
                'ascp_lower'    : float(lower_ascp[i]),
                'ascp_upper'    : float(upper_ascp[i]),
                'ascp_width'    : float(upper_ascp[i] - lower_ascp[i]),
                # ACRC
                'acrc_covered'  : int(lower_acrc[i] <= s['y_true'] <= upper_acrc[i]),
                'acrc_lower'    : float(lower_acrc[i]),
                'acrc_upper'    : float(upper_acrc[i]),
                'acrc_width'    : float(upper_acrc[i] - lower_acrc[i]),
                # COMPASS-L
                'compassl_covered' : covs_l[i] if i < len(covs_l) else np.nan,
                'compassl_width'   : wids_l[i] if i < len(wids_l) else np.nan,
                # COMPASS-J
                'compassj_covered' : covs_j[i] if i < len(covs_j) else np.nan,
                'compassj_width'   : wids_j[i] if i < len(wids_j) else np.nan,
            })

        print(f"  SCP:       Coverage={ev_scp['coverage']*100:.1f}%  Width={ev_scp['mean_width']:.2f} mL")
        print(f"  CRC:       Coverage={ev_crc['coverage']*100:.1f}%  Width={ev_crc['mean_width']:.2f} mL")
        print(f"  ASCP:      Coverage={ev_ascp['coverage']*100:.1f}%  Width={ev_ascp['mean_width']:.2f} mL")
        print(f"  ACRC:      Coverage={ev_acrc['coverage']*100:.1f}%  Width={ev_acrc['mean_width']:.2f} mL")
        print(f"  COMPASS-L: Coverage={ev_compassl['coverage']*100:.1f}%  Width={ev_compassl['mean_width']:.2f} mL")
        print(f"  COMPASS-J: Coverage={ev_compassj['coverage']*100:.1f}%  Width={ev_compassj['mean_width']:.2f} mL")

    return pd.DataFrame(results)


# ─── 4. Summarize and Plot ────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print aggregated summary table."""
    methods = [
        ('scp',      'Split CP (Baseline)'),
        ('crc',      'CRC (Theorem 2.1)'),
        ('ascp',     'Adaptive SCP (Normalized)'),
        ('acrc',     'Adaptive CRC (Normalized+Binary)'),
        ('compassl', 'COMPASS-L (Logit Shift)'),
        ('compassj', 'COMPASS-J (Jacobian) ★BEST★'),
    ]

    print("\n" + "="*70)
    print(f"SUMMARY RESULTS  (ACDC - LV Volume, Target Coverage = {(1-ALPHA)*100:.0f}%)")
    print("="*70)
    print(f"{'Method':<40} {'Coverage':>10} {'Avg Width':>12} {'v vs SCP':>10}")
    print("-"*70)

    scp_width = df['scp_width'].mean()

    for key, name in methods:
        cov_col = f'{key}_covered'
        wid_col = f'{key}_width'
        if cov_col not in df.columns:
            continue
        cov  = df[cov_col].mean() * 100
        wid  = df[wid_col].mean()
        diff = (wid - scp_width) / scp_width * 100
        diff_str = f"{diff:+.1f}%"
        marker = " <- BEST" if wid == df[[c for c in df.columns if c.endswith('_width')]].mean().min() else ""
        print(f"  {name:<38} {cov:>8.1f}%  {wid:>10.2f} mL  {diff_str:>9}{marker}")

    print("="*70)


def plot_comparison(df: pd.DataFrame):
    """Generate comparison figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    methods = ['scp', 'crc', 'ascp', 'acrc', 'compassl', 'compassj']
    labels  = ['Split CP', 'CRC', 'Adaptive\nSCP', 'Adaptive\nCRC', 'COMPASS-L', 'COMPASS-J']
    colors  = ['#E74C3C', '#3498DB', '#E67E22', '#9B59B6', '#1ABC9C', '#2ECC71']

    covs  = [df[f'{m}_covered'].mean() * 100 for m in methods]
    wids  = [df[f'{m}_width'].mean() for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Conformal Prediction Methods - ACDC LV Volume', fontsize=14, fontweight='bold')

    # Coverage
    ax1 = axes[0]
    bars = ax1.bar(labels, covs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axhline(90.0, color='red', linestyle='--', linewidth=1.5, label='Target 90%')
    ax1.set_ylabel('Empirical Coverage (%)', fontsize=12)
    ax1.set_ylim(80, 105)
    ax1.set_title('Coverage Comparison', fontsize=12)
    ax1.legend(fontsize=10)
    for bar, val in zip(bars, covs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Width
    ax2 = axes[1]
    bars2 = ax2.bar(labels, wids, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Average Interval Width (mL)', fontsize=12)
    ax2.set_title('Interval Width (v Smaller = Better)', fontsize=12)
    max_w = max(wids)
    ax2.set_ylim(0, max_w * 1.25)
    for bar, val in zip(bars2, wids):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_w * 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = FIG_DIR / "full_comparison_all_methods.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved -> {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    samples = load_all_samples()

    # 2. Compute Jacobian directions
    print("Computing Jacobian directions...")
    samples = compute_directions(samples)

    # 3. K-Fold CV
    df = run_kfold(samples)

    # 4. Save CSV
    out_csv = RESULTS_DIR / "full_comparison_results.csv"
    df.to_csv(str(out_csv), index=False)
    print(f"CSV saved -> {out_csv}")

    # 5. Summary
    print_summary(df)

    # 6. Plot
    plot_comparison(df)
