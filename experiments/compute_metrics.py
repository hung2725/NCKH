import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.acdc_dataset import ACDCDataset, load_nifti, compute_volume_ml, compute_ef, LABEL_LV, LABEL_RV, LABEL_MYO

# Cấu hình đường dẫn
WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
DATA_DIR = WORKSPACE_DIR / "data/training"
OUTPUT_DIR = WORKSPACE_DIR / "nnunet_output"
RESULTS_DIR = WORKSPACE_DIR / "results"

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading ACDC ground truth dataset...")
    dataset = ACDCDataset(str(DATA_DIR))
    records = dataset.load_all()
    
    results = []
    missing_count = 0
    
    print("\nComputing clinical metrics from nnU-Net predictions...")
    for idx, r in enumerate(records):
        pid = r['patient_id']
        
        # Lấy thông tin frame từ đường dẫn gt
        # Định dạng: patient001_frame01_gt.nii.gz -> frame01
        ed_frame_str = Path(r['gt_ed_path']).name.split('_')[1]
        es_frame_str = Path(r['gt_es_path']).name.split('_')[1]
        
        pred_ed_file = OUTPUT_DIR / f"{pid}_{ed_frame_str}.nii.gz"
        pred_es_file = OUTPUT_DIR / f"{pid}_{es_frame_str}.nii.gz"
        
        if not pred_ed_file.exists() or not pred_es_file.exists():
            print(f"  [Warning] Missing prediction files for patient {pid}")
            missing_count += 1
            continue
            
        # Load predicted mask và spacing
        mask_ed, sp_ed = load_nifti(pred_ed_file)
        mask_es, sp_es = load_nifti(pred_es_file)
        mask_ed = mask_ed.astype(np.int32)
        mask_es = mask_es.astype(np.int32)
        
        # Tính predicted metrics
        pred_lv_edv = compute_volume_ml(mask_ed, sp_ed, LABEL_LV)
        pred_lv_esv = compute_volume_ml(mask_es, sp_es, LABEL_LV)
        pred_rv_edv = compute_volume_ml(mask_ed, sp_ed, LABEL_RV)
        pred_rv_esv = compute_volume_ml(mask_es, sp_es, LABEL_RV)
        pred_myo_ed = compute_volume_ml(mask_ed, sp_ed, LABEL_MYO)
        pred_myo_es = compute_volume_ml(mask_es, sp_es, LABEL_MYO)
        
        pred_record = {
            'patient_id': pid,
            'group': r['group'],
            # Ground truth
            'LV_EDV_gt': r['LV_EDV'],
            'LV_ESV_gt': r['LV_ESV'],
            'LV_EF_gt': r['LV_EF'],
            'RV_EDV_gt': r['RV_EDV'],
            'RV_ESV_gt': r['RV_ESV'],
            'RV_EF_gt': r['RV_EF'],
            'Myo_mass_gt': r['Myo_mass'],
            # nnU-Net Predictions
            'LV_EDV_pred': pred_lv_edv,
            'LV_ESV_pred': pred_lv_esv,
            'LV_EF_pred': compute_ef(pred_lv_edv, pred_lv_esv),
            'RV_EDV_pred': pred_rv_edv,
            'RV_ESV_pred': pred_rv_esv,
            'RV_EF_pred': compute_ef(pred_rv_edv, pred_rv_esv),
            'Myo_mass_pred': (pred_myo_ed + pred_myo_es) / 2.0 * 1.05
        }
        results.append(pred_record)
        
    print(f"\nCompleted! Processed {len(results)} patients (missing: {missing_count})")
    
    if len(results) > 0:
        df = pd.DataFrame(results)
        csv_path = RESULTS_DIR / "acdc_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved combined metrics to: {csv_path}")
        
        # In một số dòng ví dụ để đối chứng
        print("\nFirst 5 patients comparison (LV_EF):")
        print(df[['patient_id', 'group', 'LV_EF_gt', 'LV_EF_pred']].head().to_string(index=False))

if __name__ == "__main__":
    main()
