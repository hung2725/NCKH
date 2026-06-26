import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conformal.split_conformal import calibrate, predict_interval, evaluate_coverage
from src.conformal.crc import find_lambda, predict_interval_crc
from src.conformal.mondrian import (
    calibrate_mondrian, predict_interval_mondrian,
    calibrate_mondrian_crc, predict_interval_mondrian_crc
)

# Cấu hình
WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
CSV_PATH = WORKSPACE_DIR / "results/acdc_metrics.csv"
FIG_DIR = WORKSPACE_DIR / "results/figures"
ALPHA = 0.1  # Target coverage = 90%
N_TRIALS = 100
CAL_PROP = 0.6  # 60% calibration, 40% test

METRICS = [
    'LV_EDV', 'LV_ESV', 'LV_EF',
    'RV_EDV', 'RV_ESV', 'RV_EF',
    'Myo_mass'
]


def winkler_score(true, lower, upper, alpha=0.1):
    """Tính Winkler score cho prediction intervals (càng nhỏ càng tốt)."""
    widths = upper - lower
    penalty_lower = (2.0 / alpha) * (lower - true) * (true < lower)
    penalty_upper = (2.0 / alpha) * (true - upper) * (true > upper)
    return widths + penalty_lower + penalty_upper


def run_experiments():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    
    print("=" * 75)
    print(f"CONFORMAL PREDICTION EXPERIMENTS ON REAL nnU-Net METRICS (ACDC)")
    print(f"Number of patients: {len(df)}")
    print(f"Target coverage: {(1 - ALPHA)*100:.1f}% (alpha = {ALPHA})")
    print(f"Number of trials: {N_TRIALS}")
    print("=" * 75)
    
    # Kết quả tổng hợp
    summary_data = []
    
    # Nhóm bệnh
    groups = df['group'].values
    unique_groups = np.unique(groups)
    
    # Để lưu coverage group-wise cho phân tích conditional coverage
    group_coverage_marginal = {g: [] for g in unique_groups}
    group_coverage_mondrian = {g: [] for g in unique_groups}
    
    for metric in METRICS:
        true_col = f"{metric}_gt"
        pred_col = f"{metric}_pred"
        
        true_vals = df[true_col].values
        pred_vals = df[pred_col].values
        
        # Lưu kết quả từng trial
        cov_scp_all, wid_scp_all, win_scp_all = [], [], []
        cov_mscp_all, wid_mscp_all, win_mscp_all = [], [], []
        cov_crc_all, wid_crc_all, win_crc_all = [], [], []
        cov_mcrc_all, wid_mcrc_all, win_mcrc_all = [], [], []
        
        np.random.seed(42)
        
        for trial in range(N_TRIALS):
            # Split data
            indices = np.arange(len(df))
            np.random.shuffle(indices)
            
            n_cal = int(len(df) * CAL_PROP)
            idx_cal = indices[:n_cal]
            idx_test = indices[n_cal:]
            
            # Calibration set
            cal_true = true_vals[idx_cal]
            cal_pred = pred_vals[idx_cal]
            cal_grp = groups[idx_cal]
            
            # Test set
            test_true = true_vals[idx_test]
            test_pred = pred_vals[idx_test]
            test_grp = groups[idx_test]
            
            # 1. Marginal Split Conformal (SCP)
            scp_res = calibrate(cal_true, cal_pred, alpha=ALPHA, mode='abs')
            lo_scp, up_scp = predict_interval(test_pred, scp_res['q_hat'], mode='abs')
            eval_scp = evaluate_coverage(test_true, lo_scp, up_scp)
            cov_scp_all.append(eval_scp['coverage'])
            wid_scp_all.append(eval_scp['mean_width'])
            win_scp_all.append(np.mean(winkler_score(test_true, lo_scp, up_scp, ALPHA)))
            
            # 2. Mondrian Split Conformal (M-SCP)
            mscp_res = calibrate_mondrian(cal_true, cal_pred, cal_grp, alpha=ALPHA, mode='abs')
            lo_mscp, up_mscp = predict_interval_mondrian(test_pred, test_grp, mscp_res)
            eval_mscp = evaluate_coverage(test_true, lo_mscp, up_mscp)
            cov_mscp_all.append(eval_mscp['coverage'])
            wid_mscp_all.append(eval_mscp['mean_width'])
            win_mscp_all.append(np.mean(winkler_score(test_true, lo_mscp, up_mscp, ALPHA)))
            
            # 3. Marginal Conformal Risk Control (CRC)
            crc_res = find_lambda(cal_true, cal_pred, alpha=ALPHA, mode='abs')
            lo_crc, up_crc = predict_interval_crc(test_pred, crc_res['lambda'], mode='abs')
            eval_crc = evaluate_coverage(test_true, lo_crc, up_crc)
            cov_crc_all.append(eval_crc['coverage'])
            wid_crc_all.append(eval_crc['mean_width'])
            win_crc_all.append(np.mean(winkler_score(test_true, lo_crc, up_crc, ALPHA)))
            
            # 4. Mondrian Conformal Risk Control (M-CRC)
            mcrc_res = calibrate_mondrian_crc(cal_true, cal_pred, cal_grp, alpha=ALPHA, mode='abs')
            lo_mcrc, up_mcrc = predict_interval_mondrian_crc(test_pred, test_grp, mcrc_res)
            eval_mcrc = evaluate_coverage(test_true, lo_mcrc, up_mcrc)
            cov_mcrc_all.append(eval_mcrc['coverage'])
            wid_mcrc_all.append(eval_mcrc['mean_width'])
            win_mcrc_all.append(np.mean(winkler_score(test_true, lo_mcrc, up_mcrc, ALPHA)))
            
            # Phân tích conditional coverage cho LV_EF
            if metric == 'LV_EF':
                for g in unique_groups:
                    g_idx = (test_grp == g)
                    if np.sum(g_idx) > 0:
                        # Marginal coverage trên group g
                        cov_g_marginal = np.mean((test_true[g_idx] >= lo_scp[g_idx]) & (test_true[g_idx] <= up_scp[g_idx]))
                        # Mondrian coverage trên group g
                        cov_g_mondrian = np.mean((test_true[g_idx] >= lo_mscp[g_idx]) & (test_true[g_idx] <= up_mscp[g_idx]))
                        
                        group_coverage_marginal[g].append(cov_g_marginal)
                        group_coverage_mondrian[g].append(cov_g_mondrian)
                        
        # Append kết quả trung bình
        summary_data.append({
            'Metric': metric,
            'SCP_Coverage': f"{np.mean(cov_scp_all)*100:.2f}% ± {np.std(cov_scp_all)*100:.2f}%",
            'SCP_Width': f"{np.mean(wid_scp_all):.2f}",
            'SCP_Winkler': f"{np.mean(win_scp_all):.2f}",
            
            'MSCP_Coverage': f"{np.mean(cov_mscp_all)*100:.2f}% ± {np.std(cov_mscp_all)*100:.2f}%",
            'MSCP_Width': f"{np.mean(wid_mscp_all):.2f}",
            'MSCP_Winkler': f"{np.mean(win_mscp_all):.2f}",
            
            'CRC_Coverage': f"{np.mean(cov_crc_all)*100:.2f}% ± {np.std(cov_crc_all)*100:.2f}%",
            'CRC_Width': f"{np.mean(wid_crc_all):.2f}",
            'CRC_Winkler': f"{np.mean(win_crc_all):.2f}",
            
            'MCRC_Coverage': f"{np.mean(cov_mcrc_all)*100:.2f}% ± {np.std(cov_mcrc_all)*100:.2f}%",
            'MCRC_Width': f"{np.mean(wid_mcrc_all):.2f}",
            'MCRC_Winkler': f"{np.mean(win_mcrc_all):.2f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nSUMMARY METRICS TABLE:")
    print("=" * 110)
    print(summary_df.to_string(index=False))
    print("=" * 110)
    
    # Save table to results
    summary_df.to_csv(WORKSPACE_DIR / "results/conformal_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # PLOT 1: Group-conditional coverage comparison (LV_EF)
    # -------------------------------------------------------------------------
    print("\nGenerating figures...")
    g_labels = []
    g_cov_marg = []
    g_cov_mond = []
    
    for g in unique_groups:
        g_labels.append(g)
        g_cov_marg.append(np.mean(group_coverage_marginal[g]))
        g_cov_mond.append(np.mean(group_coverage_mondrian[g]))
        
    x = np.arange(len(g_labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, g_cov_marg, width, label='Marginal Split CP', color='#E74C3C', alpha=0.85)
    plt.bar(x + width/2, g_cov_mond, width, label='Mondrian Split CP', color='#2ECC71', alpha=0.85)
    plt.axhline(y=1-ALPHA, color='black', linestyle='--', label='Target (90%)')
    plt.xticks(x, g_labels)
    plt.ylabel('Empirical Group Coverage')
    plt.title('Group-Conditional Coverage per Pathology (ACDC LV_EF)')
    plt.legend()
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    fig_path1 = FIG_DIR / "group_conditional_coverage.png"
    plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure 1 to: {fig_path1}")
    
    # -------------------------------------------------------------------------
    # PLOT 2: 4 sub-figures phan tich conformal prediction
    # -------------------------------------------------------------------------
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    n_cal = int(len(df) * CAL_PROP)
    idx_cal = indices[:n_cal]
    idx_test = indices[n_cal:]

    metric = 'LV_EF'
    true_vals = df[f"{metric}_gt"].values
    pred_vals = df[f"{metric}_pred"].values
    grp_vals  = df['group'].values

    scp_res  = calibrate(true_vals[idx_cal], pred_vals[idx_cal], alpha=ALPHA, mode='abs')
    crc_res  = find_lambda(true_vals[idx_cal], pred_vals[idx_cal], alpha=ALPHA, mode='abs')
    mscp_res = calibrate_mondrian(true_vals[idx_cal], pred_vals[idx_cal], grp_vals[idx_cal], alpha=ALPHA, mode='abs')
    mcrc_res = calibrate_mondrian_crc(true_vals[idx_cal], pred_vals[idx_cal], grp_vals[idx_cal], alpha=ALPHA, mode='abs')

    lo_scp,  up_scp  = predict_interval(pred_vals[idx_test], scp_res['q_hat'], mode='abs')
    lo_crc,  up_crc  = predict_interval_crc(pred_vals[idx_test], crc_res['lambda'], mode='abs')
    lo_mscp, up_mscp = predict_interval_mondrian(pred_vals[idx_test], grp_vals[idx_test], mscp_res)
    lo_mcrc, up_mcrc = predict_interval_mondrian_crc(pred_vals[idx_test], grp_vals[idx_test], mcrc_res)
    residuals = np.abs(true_vals[idx_test] - pred_vals[idx_test])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Conformal Prediction Analysis — LV_EF (Target: {int((1-ALPHA)*100)}% Coverage)',
                 fontsize=14, fontweight='bold')

    # Sub-plot 1: Histogram sai so + threshold
    ax = axes[0, 0]
    ax.hist(residuals, bins=15, color='#BDC3C7', edgecolor='white', label='|Error| distribution')
    ax.axvline(scp_res['q_hat'],  color='#E74C3C', lw=2, linestyle='-',  label=f"SCP q={scp_res['q_hat']:.3f}%")
    ax.axvline(crc_res['lambda'], color='#3498DB', lw=2, linestyle='--', label=f"CRC lambda={crc_res['lambda']:.3f}%")
    ax.set_xlabel('|Ground Truth - Prediction| (%)')
    ax.set_ylabel('So benh nhan')
    ax.set_title('Phan phoi sai so nnU-Net va nguong CP')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # Sub-plot 2: Bar chart so sanh interval width
    ax = axes[0, 1]
    methods = ['Split CP', 'CRC', 'Mondrian CP', 'Mondrian CRC']
    widths  = [np.mean(up_scp-lo_scp), np.mean(up_crc-lo_crc),
               np.mean(up_mscp-lo_mscp), np.mean(up_mcrc-lo_mcrc)]
    covs = [np.mean((true_vals[idx_test]>=lo_scp)&(true_vals[idx_test]<=up_scp)),
            np.mean((true_vals[idx_test]>=lo_crc)&(true_vals[idx_test]<=up_crc)),
            np.mean((true_vals[idx_test]>=lo_mscp)&(true_vals[idx_test]<=up_mscp)),
            np.mean((true_vals[idx_test]>=lo_mcrc)&(true_vals[idx_test]<=up_mcrc))]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    bars = ax.bar(methods, widths, color=colors, alpha=0.85, edgecolor='white')
    for bar, cov in zip(bars, covs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'cov={cov*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean Interval Width (%)')
    ax.set_title('So sanh do rong khoang du bao (LV_EF)')
    ax.set_ylim(0, max(widths)*1.35)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # Sub-plot 3: Residuals theo benh nhan (sorted) + threshold
    ax = axes[1, 0]
    sorted_res = np.sort(residuals)
    n_test = len(sorted_res)
    covered = sorted_res <= scp_res['q_hat']
    ax.scatter(np.arange(n_test)[covered],  sorted_res[covered],  color='#2ECC71', s=50, label='Covered', zorder=3)
    ax.scatter(np.arange(n_test)[~covered], sorted_res[~covered], color='#E74C3C', s=70, marker='x', label='Missed', zorder=4)
    ax.axhline(scp_res['q_hat'],  color='#E74C3C', lw=2, linestyle='--', label=f"SCP q={scp_res['q_hat']:.3f}%")
    ax.axhline(crc_res['lambda'], color='#3498DB', lw=2, linestyle=':',  label=f"CRC lam={crc_res['lambda']:.3f}%")
    ax.set_xlabel('Benh nhan test (sort theo |error|)')
    ax.set_ylabel('|Ground Truth - Prediction| (%)')
    ax.set_title('Diem xanh = covered, do = missed')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # Sub-plot 4: Coverage theo nhom benh (1 split)
    ax = axes[1, 1]
    groups_u = np.unique(grp_vals[idx_test])
    method_info = [('Split CP','#E74C3C',lo_scp,up_scp), ('Mondrian CP','#2ECC71',lo_mscp,up_mscp), ('CRC','#3498DB',lo_crc,up_crc)]
    bwidth = 0.25
    x_pos = np.arange(len(groups_u))
    for i, (mname, mcolor, lo, up) in enumerate(method_info):
        vals = []
        for g in groups_u:
            g_mask = grp_vals[idx_test] == g
            n_g = np.sum(g_mask)
            cov = np.mean((true_vals[idx_test][g_mask]>=lo[g_mask])&(true_vals[idx_test][g_mask]<=up[g_mask])) if n_g>0 else 0
            vals.append(cov)
        ax.bar(x_pos+(i-1)*bwidth, vals, bwidth, label=mname, color=mcolor, alpha=0.85)
    ax.axhline(y=1-ALPHA, color='black', linestyle='--', lw=1.5, label=f'Target ({int((1-ALPHA)*100)}%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups_u)
    ax.set_ylabel('Empirical Coverage')
    ax.set_ylim(0, 1.15)
    ax.set_title('Coverage theo nhom benh (1 split)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig_path2 = FIG_DIR / "conformal_analysis_lv_ef.png"
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure 2 to: {fig_path2}")
    
    print("\nExperiments complete!")

if __name__ == "__main__":
    run_experiments()
