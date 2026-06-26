"""
src/data/acdc_dataset.py
Load ACDC dataset va tinh clinical metrics tu ground truth masks.

Label convention:
    0 = Background
    1 = Right Ventricle (RV)
    2 = Myocardium (Myo)
    3 = Left Ventricle (LV)
"""

import numpy as np
import nibabel as nib
from pathlib import Path

LABEL_RV  = 1
LABEL_MYO = 2
LABEL_LV  = 3


def load_nifti(path):
    img    = nib.load(str(path))
    data   = np.array(img.get_fdata(), dtype=np.float32)
    zooms  = img.header.get_zooms()
    return data, tuple(float(z) for z in zooms[:3])


def parse_info_cfg(path):
    info = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return info


def compute_volume_ml(mask, spacing, label):
    """The tich (mL) cua mot label trong mask."""
    voxel_mm3 = spacing[0] * spacing[1] * spacing[2]
    n_voxels  = int(np.sum(mask == label))
    return n_voxels * voxel_mm3 / 1000.0


def compute_ef(edv, esv):
    """Ejection Fraction (%)."""
    if edv <= 0:
        return 0.0
    return float(np.clip((edv - esv) / edv * 100.0, 0, 100))


class ACDCDataset:
    """
    Load ACDC va tra ve list cac record.

    Moi record la 1 dict chua:
        patient_id, group,
        LV_EDV, LV_ESV, LV_EF,
        RV_EDV, RV_ESV, RV_EF,
        Myo_mass,
        gt_ed_path, gt_es_path,
        mri_ed_path, mri_es_path
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith('patient')
        ])
        print(f"[ACDC] Tim thay {len(self.patient_dirs)} benh nhan")

    def load_patient(self, patient_dir):
        pid      = patient_dir.name
        info_cfg = patient_dir / 'Info.cfg'
        if not info_cfg.exists():
            return None

        info     = parse_info_cfg(str(info_cfg))
        ed_frame = int(info.get('ED', 1))
        es_frame = int(info.get('ES', 1))

        gt_ed = patient_dir / f'{pid}_frame{ed_frame:02d}_gt.nii.gz'
        gt_es = patient_dir / f'{pid}_frame{es_frame:02d}_gt.nii.gz'
        if not gt_ed.exists() or not gt_es.exists():
            return None

        mask_ed, sp_ed = load_nifti(gt_ed)
        mask_es, sp_es = load_nifti(gt_es)
        mask_ed = mask_ed.astype(np.int32)
        mask_es = mask_es.astype(np.int32)

        lv_edv = compute_volume_ml(mask_ed, sp_ed, LABEL_LV)
        lv_esv = compute_volume_ml(mask_es, sp_es, LABEL_LV)
        rv_edv = compute_volume_ml(mask_ed, sp_ed, LABEL_RV)
        rv_esv = compute_volume_ml(mask_es, sp_es, LABEL_RV)
        myo_ed = compute_volume_ml(mask_ed, sp_ed, LABEL_MYO)
        myo_es = compute_volume_ml(mask_es, sp_es, LABEL_MYO)

        return {
            'patient_id' : pid,
            'group'      : info.get('Group', 'UNKNOWN'),
            'height'     : float(info.get('Height', 0)),
            'weight'     : float(info.get('Weight', 0)),
            # Ground truth clinical metrics
            'LV_EDV'     : lv_edv,
            'LV_ESV'     : lv_esv,
            'LV_EF'      : compute_ef(lv_edv, lv_esv),
            'RV_EDV'     : rv_edv,
            'RV_ESV'     : rv_esv,
            'RV_EF'      : compute_ef(rv_edv, rv_esv),
            'Myo_mass'   : (myo_ed + myo_es) / 2.0 * 1.05,  # density 1.05 g/mL
            # Paths
            'gt_ed_path' : str(gt_ed),
            'gt_es_path' : str(gt_es),
            'mri_ed_path': str(patient_dir / f'{pid}_frame{ed_frame:02d}.nii.gz'),
            'mri_es_path': str(patient_dir / f'{pid}_frame{es_frame:02d}.nii.gz'),
        }

    def load_all(self):
        records, skipped = [], 0
        for d in self.patient_dirs:
            r = self.load_patient(d)
            if r:
                records.append(r)
            else:
                skipped += 1
        print(f"[ACDC] Load {len(records)} benh nhan (bo qua {skipped})")
        return records