"""
experiments/run_lits_pipeline.py
Pipeline hoan chinh cho LiTS: prepare -> inference -> metrics -> experiment (6 methods)
Chay: python experiments/run_lits_pipeline.py
"""
import sys, os, argparse, numpy as np, pandas as pd, nibabel as nib, torch
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))

# Config
WORKSPACE = Path("D:/Hoc_Tap/NCKH")
LITS_DATA = WORKSPACE / "data_LiTS"
LITS_INPUT = WORKSPACE / "nnunet_input_lits"
LITS_OUTPUT = WORKSPACE / "nnunet_output_lits"
NNUNET_DATA = WORKSPACE / "nnunet_data"
LABEL_LIVER, LABEL_TUMOR = 1, 2

os.environ["nnUNet_raw_data_base"] = str(NNUNET_DATA / "nnUNet_raw_data_base")
os.environ["nnUNet_preprocessed"] = str(NNUNET_DATA / "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = "C:/Users/T.Hung/nnunet_v1/results"

# ===== Buoc 1: Format input =====
def prepare():
    print("="*55); print("  BUOC 1: Format LiTS input (.nii -> .nii.gz)"); print("="*55)
    if LITS_INPUT.exists():
        os.system(f'rmdir /s /q "{LITS_INPUT}"')
        import time; time.sleep(1)
    LITS_INPUT.mkdir(parents=True, exist_ok=True)
    files = sorted((LITS_DATA/"images").glob("volume-*.nii"), key=lambda x: int(x.stem.split("-")[1]))
    print(f"Nen {len(files)} volumes...")
    for src in tqdm(files):
        vol_id = src.stem.split("-")[1]
        img = nib.load(str(src))
        nib.save(img, str(LITS_INPUT / f"lits_{vol_id}_0000.nii.gz"))
    print(f"Done! {len(list(LITS_INPUT.glob('*.nii.gz')))} files\n")

# ===== Buoc 2: Inference =====
def inference(model="3d_lowres"):
    print("="*55); print(f"  BUOC 2: nnU-Net inference ({model})"); print("="*55)
    LITS_OUTPUT.mkdir(parents=True, exist_ok=True)
    # Patches
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
    try:
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
    except: pass
    import nnunet.utilities.nd_softmax, torch.nn.functional as F
    def _sh(x): return F.softmax(x, 1)
    _sh.__module__ = "nnunet.utilities.nd_softmax"; _sh.__name__ = "softmax_helper"
    nnunet.utilities.nd_softmax.softmax_helper = _sh
    import multiprocessing.pool, nnunet.inference.predict
    nnunet.inference.predict.Pool = multiprocessing.pool.ThreadPool
    def _pp(trainer, lol, of, np2=2, sfps=None):
        if sfps is None: sfps = [None]*len(lol)
        for i, l in enumerate(lol):
            d, _, dct = trainer.preprocess_patient(l)
            yield (of[i], (d, dct))
    nnunet.inference.predict.preprocess_multithreaded = _pp
    from nnunet.inference.predict import predict_from_folder

    mf = Path("C:/Users/T.Hung/nnunet_v1/results/nnUNet") / model / "Task003_Liver"
    trainers = list(mf.glob("nnUNetTrainerV2*")) if mf.exists() else []
    mf = trainers[0] if trainers else mf
    if not mf.exists():
        print(f"ERROR: Model not found. Download Task003_Liver.zip first.")
        sys.exit(1)
    print(f"Using model: {mf}")
    predict_from_folder(model=str(mf), input_folder=str(LITS_INPUT), output_folder=str(LITS_OUTPUT),
                        folds=None, save_npz=True, num_threads_preprocessing=2, num_threads_nifti_save=2,
                        lowres_segmentations=None, part_id=0, num_parts=1, tta=False,
                        mixed_precision=True, overwrite_existing=True)
    print("OK\n")

# ===== Buoc 3: Metrics =====
def metrics():
    print("="*55); print("  BUOC 3: Tinh clinical metrics"); print("="*55)
    results = []
    for npz_path in tqdm(sorted(LITS_OUTPUT.glob("*.npz")), desc="Computing"):
        stem = npz_path.stem; vol_id = stem.split("_")[1]
        gt_path = LITS_DATA / "masks" / f"segmentation-{vol_id}.nii"
        pred_path = LITS_OUTPUT / f"{stem}.nii.gz"
        if not gt_path.exists(): continue
        data = np.load(str(npz_path))
        probs = data['softmax'].astype(np.float32)
        gt = nib.load(str(gt_path))
        sp = gt.header.get_zooms()[:3]; vv = float(sp[0]*sp[1]*sp[2])/1000.0
        gv = gt.dataobj
        gt_liv = sum(int(np.sum(np.asarray(gv[...,z]).astype(np.uint8)==1)) for z in range(gv.shape[2]))*vv
        gt_tum = sum(int(np.sum(np.asarray(gv[...,z]).astype(np.uint8)==2)) for z in range(gv.shape[2]))*vv
        ns = probs.shape[1]
        pr_liv = sum(int(np.sum(np.argmax(probs[:,z,:,:],axis=0)==1)) for z in range(ns))*vv
        pr_tum = sum(int(np.sum(np.argmax(probs[:,z,:,:],axis=0)==2)) for z in range(ns))*vv
        results.append({"patient_id":f"lits_{vol_id}","liver_volume_gt":gt_liv,"tumor_volume_gt":gt_tum,
                        "liver_volume_pred":pr_liv,"tumor_volume_pred":pr_tum,
                        "has_tumor":gt_tum>0})
        del probs, data
    df = pd.DataFrame(results)
    df.to_csv(WORKSPACE/"results"/"lits_metrics.csv", index=False)
    print(f"Saved {len(df)} patients ({df['has_tumor'].sum()} with tumor)")
    for m in ["liver_volume","tumor_volume"]:
        mae = (df[f"{m}_gt"]-df[f"{m}_pred"]).abs().mean()
        print(f"  {m}: MAE={mae:.1f} mL")
    print("OK\n")

# ===== Buoc 4: Experiment =====
def experiment():
    print("="*55); print("  BUOC 4: CRC-FS Experiment"); print("="*55)
    npz_files = sorted(LITS_OUTPUT.glob("*.npz"))
    if not npz_files: print("ERROR: No .npz files"); return
    from src.conformal.adaptive_scores import compute_uncertainty_from_probs
    ALPHA, TL, TT = 0.1, 1, 2

    # Pass 1: scalars only
    samples = []
    for npz_path in tqdm(npz_files, desc="Pass 1 (scalars)"):
        stem = npz_path.stem; vol_id = stem.split("_")[1]
        gt_path = LITS_DATA/"masks"/f"segmentation-{vol_id}.nii"
        if not gt_path.exists(): continue
        data = np.load(str(npz_path)); probs = data['softmax'].astype(np.float32)
        gt = nib.load(str(gt_path)); sp = gt.header.get_zooms()[:3]; vv = float(sp[0]*sp[1]*sp[2])/1000.0
        gv = gt.dataobj; ns = probs.shape[1]
        gt_liv = sum(int(np.sum(np.asarray(gv[...,z]).astype(np.uint8)==1)) for z in range(gv.shape[2]))*vv
        gt_tum = sum(int(np.sum(np.asarray(gv[...,z]).astype(np.uint8)==2)) for z in range(gv.shape[2]))*vv
        pr_liv = sum(int(np.sum(np.argmax(probs[:,z,:,:],axis=0)==1)) for z in range(ns))*vv
        pr_tum = sum(int(np.sum(np.argmax(probs[:,z,:,:],axis=0)==2)) for z in range(ns))*vv
        sl = compute_uncertainty_from_probs(probs, TL, method='entropy')
        st = compute_uncertainty_from_probs(probs, TT, method='entropy')
        samples.append({"patient_id":f"lits_{vol_id}","npz_path":str(npz_path),
                        "y_true_liver":gt_liv,"y_pred_liver":pr_liv,"y_true_tumor":gt_tum,"y_pred_tumor":pr_tum,
                        "sigma_liver":sl,"sigma_tumor":st,"voxel_vol":vv,"has_tumor":gt_tum>0})
        del probs, data
    print(f"Loaded {len(samples)} samples")

    # 4a: SCP/CRC/ASCP/ACRC
    print("\n--- Liver Volume (SCP/CRC/ASCP/ACRC) ---")
    _run_output_space(samples, "liver", ALPHA)
    ts = [s for s in samples if s["has_tumor"]]
    if len(ts) > 10:
        print(f"\n--- Tumor Volume ({len(ts)} patients) ---")
        _run_output_space(ts, "tumor", ALPHA)

    # 4b: COMPASS-L + CRC-FS-L
    print("\n--- Liver Volume (COMPASS-L + CRC-FS-L) ---")
    _run_feature_space(samples, "liver", TL, ALPHA)

def _run_output_space(samples, metric, alpha):
    from sklearn.model_selection import KFold
    from src.conformal.split_conformal import calibrate, predict_interval, evaluate_coverage
    from src.conformal.crc import find_lambda
    from src.conformal.adaptive_scores import calibrate_normalized, predict_interval_normalized, find_lambda_adaptive
    n = len(samples); kf = KFold(n_splits=min(5,n), shuffle=True, random_state=42)
    yt = np.array([s[f"y_true_{metric}"] for s in samples])
    yp = np.array([s[f"y_pred_{metric}"] for s in samples])
    sg = np.array([s[f"sigma_{metric}"] for s in samples])
    res = {m:{"cov":[],"wid":[]} for m in ["scp","crc","ascp","acrc"]}
    for cal, test in kf.split(np.arange(n)):
        ct, cp, cs = yt[cal], yp[cal], sg[cal]
        tt, tp, ts = yt[test], yp[test], sg[test]
        for key, cal_fn, pred_fn, cal_args in [
            ("scp", calibrate, predict_interval, [ct,cp,alpha,'abs']),
            ("crc", find_lambda, predict_interval, [ct,cp,alpha,'abs']),
            ("ascp", calibrate_normalized, predict_interval_normalized, [ct,cp,cs,alpha]),
            ("acrc", find_lambda_adaptive, predict_interval_normalized, [ct,cp,cs,alpha])]:
            if key in ["ascp","acrc"]:
                res_mod = cal_fn(*cal_args)
                lo, up = pred_fn(tp, ts, res_mod['q_hat' if key=="ascp" else 'lambda'])
            elif key == "crc":
                res_mod = cal_fn(*cal_args)
                lo, up = pred_fn(tp, res_mod['lambda'], 'abs')
            else:
                res_mod = cal_fn(*cal_args)
                lo, up = pred_fn(tp, res_mod['q_hat'], 'abs')
            ev = evaluate_coverage(tt, lo, up)
            res[key]["cov"].append(ev["coverage"]); res[key]["wid"].append(ev["mean_width"])
    tgt = f"{(1-alpha)*100:.0f}%"
    print(f"  {'Method':<15} {'Coverage':>10} {'Width':>12}  Target={tgt}")
    print(f"  {'-'*40}")
    for key, name in [("scp","Split CP"),("crc","CRC"),("ascp","Adaptive SCP"),("acrc","Adaptive CRC")]:
        cov = np.mean(res[key]["cov"])*100; wid = np.mean(res[key]["wid"])
        print(f"  {name:<15} {cov:>9.1f}% {wid:>9.1f} mL  {'OK' if cov>=(1-alpha)*100-1 else 'X'}")

def _run_feature_space(samples, metric, target_class, alpha):
    from sklearn.model_selection import KFold
    from src.conformal.compass import compass_l_score_binary, calibrate_compass, predict_interval_compass, compute_metric_for_b
    from src.conformal.split_conformal import conformal_quantile
    yt = np.array([s[f"y_true_{metric}"] for s in samples])
    sg = np.array([s[f"sigma_{metric}"] for s in samples]); n = len(samples)
    kf = KFold(n_splits=min(5,n), shuffle=True, random_state=42)
    ccov, cwid = [], []; fc, fw = [], []
    for fold, (cal, test) in enumerate(kf.split(np.arange(n))):
        print(f"  Fold {fold+1}...", flush=True)
        scores = []
        for i in cal:
            s = samples[i]; d = np.load(s["npz_path"]); p = d['softmax']
            t = torch.tensor(p, dtype=torch.float16, device='cuda' if torch.cuda.is_available() else 'cpu'); vv = s["voxel_vol"]
            mf = lambda mask, vv=vv: (mask==target_class).sum().item()*vv if isinstance(mask, torch.Tensor) else np.sum(mask==target_class)*vv
            scores.append(compass_l_score_binary(t, float(yt[i]), target_class, mf, b_max=5.0, n_iter=10))
            del t, p, d
            torch.cuda.empty_cache()
        scores = np.array(scores)
        beta = calibrate_compass(scores, alpha)
        cs = sg[cal]; ns = scores / np.maximum(cs, 1e-8)
        qh = conformal_quantile(ns, alpha); ms = np.median(cs)
        for i in test:
            s = samples[i]; d = np.load(s["npz_path"]); p = d['softmax']
            t = torch.tensor(p, dtype=torch.float16, device='cuda' if torch.cuda.is_available() else 'cpu'); vv = s["voxel_vol"]
            mf = lambda mask, vv=vv: (mask==target_class).sum().item()*vv if isinstance(mask, torch.Tensor) else np.sum(mask==target_class)*vv
            lo, up = predict_interval_compass(t, beta, target_class, mf)
            ccov.append(1.0 if lo<=yt[i]<=up else 0.0); cwid.append(up-lo)
            bf = float(qh*sg[i]); mn = compute_metric_for_b(t, -bf, target_class, mf)
            mp = compute_metric_for_b(t, bf, target_class, mf)
            lf, uf = min(mn,mp), max(mn,mp)
            fc.append(1.0 if lf<=yt[i]<=uf else 0.0); fw.append(uf-lf)
            del t, p, d
            torch.cuda.empty_cache()
    print(f"\n  {'Method':<15} {'Coverage':>10} {'Width':>12}")
    print(f"  {'-'*40}")
    print(f"  {'COMPASS-L':<15} {np.mean(ccov)*100:>9.1f}% {np.mean(cwid):>9.1f} mL")
    print(f"  {'CRC-FS-L':<15} {np.mean(fc)*100:>9.1f}% {np.mean(fw):>9.1f} mL")

# ===== Main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true"); parser.add_argument("--inference", action="store_true")
    parser.add_argument("--metrics", action="store_true"); parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--model", type=str, default="3d_lowres")
    args = parser.parse_args()
    if not any([args.prepare, args.inference, args.metrics, args.experiment]):
        args.prepare = args.inference = args.metrics = args.experiment = True
    print("="*55, flush=True); print("  LiTS PIPELINE", flush=True); print("="*55, flush=True)
    if args.prepare: prepare()
    if args.inference: inference(model=args.model)
    if args.metrics: metrics()
    if args.experiment: experiment()
    print("\n===== DONE =====", flush=True)
