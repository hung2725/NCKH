"""
src/data/lits_dataset.py

LiTS (Liver Tumor Segmentation) Dataset Loader.

Dataset: 131 CT volumes from LiTS Challenge (Kaggle)
- images/volume-{i}.nii  (CT scan)
- masks/segmentation-{i}.nii  (0=bg, 1=liver, 2=tumor)

Clinical metrics:
- Liver volume (mL)
- Tumor volume (mL)
- Tumor burden (%) = tumor_volume / liver_volume * 100
"""

import numpy as np
import nibabel as nib
from pathlib import Path

LABEL_LIVER = 1
LABEL_TUMOR = 2


def load_nifti(path):
    img = nib.load(str(path))
    data = np.array(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()
    return data, tuple(float(z) for z in zooms[:3])


def compute_volume_ml(mask, spacing, label):
    """The tich (mL) cua mot label trong mask."""
    voxel_mm3 = spacing[0] * spacing[1] * spacing[2]
    n_voxels = int(np.sum(mask == label))
    return n_voxels * voxel_mm3 / 1000.0


class LiTSDataset:
    """
    Load LiTS dataset.

    Moi record la dict chua:
        patient_id, liver_volume_ml, tumor_volume_ml, tumor_burden_pct,
        image_path, mask_path
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"

        # Tim tat ca volume files
        self.image_files = sorted(
            self.image_dir.glob("volume-*.nii"),
            key=lambda x: int(x.stem.split("-")[1])
        )

        print(f"[LiTS] Tim thay {len(self.image_files)} volumes")

    def load_patient(self, idx):
        """Load 1 volume theo index."""
        img_path = self.image_files[idx]
        vol_id = img_path.stem.split("-")[1]  # "0", "1", ...
        mask_path = self.mask_dir / f"segmentation-{vol_id}.nii"

        if not mask_path.exists():
            print(f"  [Warning] Missing mask for volume-{vol_id}")
            return None

        image, spacing = load_nifti(img_path)
        mask, _ = load_nifti(mask_path)
        mask = mask.astype(np.int32)

        liver_vol = compute_volume_ml(mask, spacing, LABEL_LIVER)
        tumor_vol = compute_volume_ml(mask, spacing, LABEL_TUMOR)
        tumor_burden = (tumor_vol / liver_vol * 100) if liver_vol > 0 else 0.0

        return {
            "patient_id": f"lits_{vol_id}",
            "vol_id": vol_id,
            "liver_volume_ml": liver_vol,
            "tumor_volume_ml": tumor_vol,
            "tumor_burden_pct": tumor_burden,
            "has_tumor": tumor_vol > 0,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "spacing": spacing,
            "image_shape": image.shape,
        }

    def load_all(self):
        """Load toan bo dataset."""
        records, skipped = [], 0
        for i in range(len(self.image_files)):
            r = self.load_patient(i)
            if r:
                records.append(r)
            else:
                skipped += 1
        n_tumor = sum(1 for r in records if r["has_tumor"])
        print(f"[LiTS] Load {len(records)} patients "
              f"({n_tumor} co tumor, {len(records)-n_tumor} khong tumor, "
              f"bo qua {skipped})")
        return records

if __name__ == "__main__":
    import pandas as pd

    DATA_DIR = "D:/Hoc_Tap/NCKH/data_LiTS"
    dataset = LiTSDataset(DATA_DIR)
    records = dataset.load_all()
    df = pd.DataFrame(records)

    print(f"\nTong: {len(df)} patients")
    print(f"Co tumor: {df['has_tumor'].sum()}")
    print(f"Khong tumor: {(~df['has_tumor']).sum()}")
    print(f"\nClinical metrics:")
    print(df[["liver_volume_ml", "tumor_volume_ml", "tumor_burden_pct"]].describe().round(1))
