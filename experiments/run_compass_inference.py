import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

# Them thu muc goc vao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conformal.compass import compass_l_score, calibrate_compass, predict_interval_compass

WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
DATA_DIR = WORKSPACE_DIR / "data/training"
PRED_DIR = WORKSPACE_DIR / "nnunet_output"

# Class label for Left Ventricle in ACDC
TARGET_CLASS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_patient_data():
    """
    Load cac benh nhan co file .npz.
    Tra ve danh sach cac tuple (patient_id, frame_id, npz_path, gt_path)
    """
    records = []
    
    # Kiem tra xem co file npz chua
    npz_files = list(PRED_DIR.glob("*.npz"))
    if not npz_files:
        print("ERROR: Khong tim thay file .npz trong nnunet_output.")
        print("Vui long chay: python experiments/run_nnunet_inference.py --model 3d_fullres --save_npz")
        sys.exit(1)
        
    print(f"Tim thay {len(npz_files)} file .npz.")
    
    for npz_path in npz_files:
        # Ten file co dang: patient001_frame01.npz
        basename = npz_path.stem
        parts = basename.split("_")
        if len(parts) >= 2:
            patient_id = parts[0]
            frame_id = parts[1]
            gt_path = DATA_DIR / patient_id / f"{patient_id}_{frame_id}_gt.nii.gz"
            if gt_path.exists():
                records.append({
                    "patient_id": patient_id,
                    "frame": frame_id,
                    "npz_path": str(npz_path),
                    "gt_path": str(gt_path)
                })
                
    return records

def compute_metric_from_binary_mask(mask, voxel_spacing: tuple) -> float:
    """Tinh the tich (mL) tu binary mask (np.ndarray hoac torch.Tensor)"""
    voxel_vol_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    if isinstance(mask, torch.Tensor):
        vol_ml = mask.sum().item() * voxel_vol_mm3 / 1000.0
    else:
        vol_ml = np.sum(mask) * voxel_vol_mm3 / 1000.0
    return vol_ml

def run_compass_experiment():
    records = load_patient_data()
    
    # Buoc 1: Tinh Ground Truth volume va load prob
    print("Loading probabilities va computing Ground Truth...")
    
    data = []
    
    for r in tqdm(records):
        npz_data = np.load(r["npz_path"], allow_pickle=True)
        # nnU-Net luu probabilities o npz_data['probabilities'] hoac trong tuple
        if 'probabilities' in npz_data:
            probs = npz_data['probabilities']
        else:
            # Fallback for some old nnU-Net versions saving a list/tuple
            try:
                content = npz_data['arr_0']
                probs = content
            except Exception as e:
                # If it's a dict containing dicts
                probs = npz_data.get('softmax') # Try another common key
                if probs is None:
                    # Let's just read the raw keys
                    keys = list(npz_data.keys())
                    probs = npz_data[keys[0]]

        gt_img = nib.load(r["gt_path"])
        gt_mask = gt_img.get_fdata()
        spacing = gt_img.header.get_zooms()
        
        # Tinh GT volume cho LV (class 3)
        gt_binary = (gt_mask == TARGET_CLASS).astype(np.uint8)
        y_true = compute_metric_from_binary_mask(gt_binary, spacing)
        
        # Default prediction without shift
        pred_mask = np.argmax(probs, axis=0)
        pred_binary = (pred_mask == TARGET_CLASS).astype(np.uint8)
        y_pred = compute_metric_from_binary_mask(pred_binary, spacing)
        
        # Convert probs to GPU tensor for COMPASS
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=DEVICE)
        
        # Tao ham metric_fn cho rieng buc anh nay (vi spacing khac nhau)
        def metric_fn_bound(binary_mask, sp=spacing):
            return compute_metric_from_binary_mask(binary_mask, sp)
            
        data.append({
            "patient_id": r["patient_id"],
            "frame": r["frame"],
            "y_true": y_true,
            "y_pred": y_pred,
            "probs": probs_tensor,
            "metric_fn": metric_fn_bound
        })
        
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples.")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alpha = 0.1
    
    results = []
    
    print("Running COMPASS Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"--- Fold {fold+1} ---")
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        # 1. Calibration phase: Tinh COMPASS scores cho train_data
        print(f"Calibrating on {len(train_data)} samples...")
        cal_scores = []
        for _, row in train_data.iterrows():
            score = compass_l_score(
                probs=row["probs"],
                y_true=row["y_true"],
                target_class=TARGET_CLASS,
                metric_fn=row["metric_fn"],
                b_max=10.0,
                steps=50
            )
            cal_scores.append(score)
            
        # Tinh beta_hat
        beta_hat = calibrate_compass(np.array(cal_scores), alpha)
        print(f"Calibrated beta_hat = {beta_hat:.4f}")
        
        # 2. Testing phase
        for _, row in test_data.iterrows():
            lower, upper = predict_interval_compass(
                probs=row["probs"],
                beta_hat=beta_hat,
                target_class=TARGET_CLASS,
                metric_fn=row["metric_fn"]
            )
            
            # Split CP baseline for comparison
            # Split CP tinh error = |y_true - y_pred|
            cal_errors = (train_data["y_true"] - train_data["y_pred"]).abs().values
            from src.conformal.split_conformal import conformal_quantile
            q_hat = conformal_quantile(cal_errors, alpha)
            
            scp_lower = max(0, row["y_pred"] - q_hat)
            scp_upper = row["y_pred"] + q_hat
            
            results.append({
                "patient_id": row["patient_id"],
                "frame": row["frame"],
                "y_true": row["y_true"],
                "y_pred": row["y_pred"],
                
                # COMPASS
                "compass_lower": lower,
                "compass_upper": upper,
                "compass_width": upper - lower,
                "compass_covered": lower <= row["y_true"] <= upper,
                
                # Split CP
                "scp_lower": scp_lower,
                "scp_upper": scp_upper,
                "scp_width": scp_upper - scp_lower,
                "scp_covered": scp_lower <= row["y_true"] <= scp_upper,
            })
            
    res_df = pd.DataFrame(results)
    
    # Hien thi so sanh
    compass_cov = res_df["compass_covered"].mean() * 100
    compass_width = res_df["compass_width"].mean()
    
    scp_cov = res_df["scp_covered"].mean() * 100
    scp_width = res_df["scp_width"].mean()
    
    print("\\n" + "="*50)
    print(f"KET QUA SO SANH (LV_EDV or LV_ESV - Volume theo mL)")
    print("="*50)
    print(f"Target Coverage: {(1-alpha)*100:.1f}%")
    print("-" * 50)
    print(f"SPLIT CP (Baseline):")
    print(f"  Coverage : {scp_cov:.2f}%")
    print(f"  Avg Width: {scp_width:.2f} mL")
    print("-" * 50)
    print(f"COMPASS-L (Feature Perturbation):")
    print(f"  Coverage : {compass_cov:.2f}%")
    print(f"  Avg Width: {compass_width:.2f} mL")
    print("="*50)
    out_path = WORKSPACE_DIR / "results" / "compass_results_lv_volume.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Saved detailed results to {out_path}")

if __name__ == "__main__":
    run_compass_experiment()
