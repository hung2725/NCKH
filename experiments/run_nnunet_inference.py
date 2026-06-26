import os
import shutil
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
except Exception:
    pass

import nnunet.utilities.nd_softmax
import torch.nn.functional as F

def softmax_helper(x):
    return F.softmax(x, 1)

softmax_helper.__module__ = "nnunet.utilities.nd_softmax"
softmax_helper.__name__ = "softmax_helper"
nnunet.utilities.nd_softmax.softmax_helper = softmax_helper
# -----------------------------------------------------------------------------

from src.data.acdc_dataset import ACDCDataset



import multiprocessing.pool
import nnunet.inference.predict

# Patch Pool to ThreadPool to avoid multiprocessing pickling on Windows
nnunet.inference.predict.Pool = multiprocessing.pool.ThreadPool

# Patch preprocess_multithreaded to run in the main thread
def preprocess_multithreaded_patched(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)
    
    for i, l in enumerate(list_of_lists):
        output_file = output_files[i]
        print(f"Preprocessing {output_file} (single-threaded patch)...")
        d, _, dct = trainer.preprocess_patient(l)
        yield (output_file, (d, dct))

nnunet.inference.predict.preprocess_multithreaded = preprocess_multithreaded_patched

from nnunet.inference.predict import predict_from_folder

# Cấu hình đường dẫn
WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
DATA_DIR = WORKSPACE_DIR / "data/training"
INPUT_DIR = WORKSPACE_DIR / "nnunet_input"
OUTPUT_DIR = WORKSPACE_DIR / "nnunet_output"
NNUNET_DATA_DIR = WORKSPACE_DIR / "nnunet_data"

# Thiết lập environment variables cho nnU-Net (vẫn cần thiết vì nnU-Net đọc chúng khi import/chạy)
os.environ["nnUNet_raw_data_base"] = str(NNUNET_DATA_DIR / "nnUNet_raw_data_base")
os.environ["nnUNet_preprocessed"] = str(NNUNET_DATA_DIR / "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = str(NNUNET_DATA_DIR / "RESULTS_FOLDER")


def prepare_input_data(limit=None):
    """
    Copy các ảnh ED và ES từ thư mục dataset gốc sang thư mục input của nnU-Net,
    đổi tên thành định dạng patientXXX_frameYY_0000.nii.gz.
    """
    print("Preparing input data for nnU-Net...")
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ACDCDataset(str(DATA_DIR))
    records = dataset.load_all()

    if limit:
        records = records[:limit]
        print(f"Limiting to first {limit} patients for testing.")

    count = 0
    for r in records:
        # ED frame
        src_ed = Path(r["mri_ed_path"])
        dest_ed_name = f"{src_ed.stem.split('.')[0]}_0000.nii.gz"
        dest_ed = INPUT_DIR / dest_ed_name
        shutil.copy2(src_ed, dest_ed)

        # ES frame
        src_es = Path(r["mri_es_path"])
        dest_es_name = f"{src_es.stem.split('.')[0]}_0000.nii.gz"
        dest_es = INPUT_DIR / dest_es_name
        shutil.copy2(src_es, dest_es)

        count += 2

    print(f"Prepared {count} images in {INPUT_DIR}")


def run_inference(model="2d", tta=False):
    """
    Chạy nnUNet_predict để dự đoán phân đoạn.
    """
    print(f"Running nnU-Net inference using {model} model (TTA={tta})...")
    
    # Tìm model folder
    model_folder = NNUNET_DATA_DIR / "RESULTS_FOLDER" / "nnUNet" / model / "Task027_ACDC" / "nnUNetTrainerV2__nnUNetPlansv2.1"
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
        
    print(f"Using model folder: {model_folder}")
    
    predict_from_folder(
        model=str(model_folder),
        input_folder=str(INPUT_DIR),
        output_folder=str(OUTPUT_DIR),
        folds=None,
        save_npz=False,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=tta,
        mixed_precision=True,
        overwrite_existing=True
    )
    print("Inference completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run nnU-Net inference on ACDC")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients to predict")
    parser.add_argument("--model", type=str, default="2d", choices=["2d", "3d_fullres"], help="nnU-Net model mode")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (slower)")
    args = parser.parse_args()

    prepare_input_data(limit=args.limit)
    run_inference(model=args.model, tta=args.tta)
