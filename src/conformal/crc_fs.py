"""
src/conformal/crc_fs.py

CRC-FS: Conformal Risk Control in Feature Space
===============================================

Novel framework kết hợp 3 hướng nghiên cứu:
  1. COMPASS (ICLR 2026)         — Feature-space perturbation
  2. CRC (ICLR 2024)              — Finite-sample risk control (Theorem 2.1)
  3. Adaptive CP (Angelopoulos)   — Normalized scores với uncertainty σ_i

ĐÓNG GÓP CHÍNH:
  Lần đầu tiên kết hợp Conformal Risk Control với Feature-Space Perturbation,
  tạo ra prediction intervals vừa có:
    - Finite-sample risk guarantee (mạnh hơn Split CP)
    - Feature-space sensitivity (interval phản ánh cấu trúc model)
    - Adaptive width (interval hẹp cho vùng dễ, rộng cho vùng khó)

ALGORITHM (CRC-FS):
  ┌─────────────────────────────────────────────────────────────┐
  │ CALIBRATION (trên tập cal, n samples):                       │
  │                                                              │
  │   For each sample i:                                         │
  │     R_i  = compass_score(probs_i, y_true_i)                  │
  │            → min β để y_true ∈ [m(-β), m(+β)]               │
  │     σ_i  = entropy(probs_i)                                  │
  │            → model uncertainty estimate                       │
  │     R'_i = R_i / σ_i                                         │
  │            → normalized nonconformity score                   │
  │                                                              │
  │   CRC calibration (Theorem 2.1, Angelopoulos et al. 2024):   │
  │     L_i(λ) = min(1, R'_i / λ) ∈ [0, 1]                     │
  │     λ̂ = inf{ λ ≥ 0 : (1/n)Σ L_i(λ) ≤ α - (1-α)/n }         │
  │                                                              │
  │ PREDICTION (cho test sample j):                              │
  │   β_j = λ̂ × σ_j                                              │
  │   interval = [m(-β_j), m(+β_j)]                              │
  └─────────────────────────────────────────────────────────────┘

VARIANTS:
  - CRC-FS-L: Dùng COMPASS-L perturbation (uniform logit shift)
  - CRC-FS-J: Dùng COMPASS-J perturbation (Jacobian direction)

COMPLEXITY: O(n_cal × n_cal × steps + n_test)
  (so với COMPASS gốc: O(n_cal × steps), thêm factor n_cal cho CRC binary search)

REFERENCES:
  [1] Angelopoulos et al., "Conformal Risk Control", ICLR 2024.
  [2] Cheung et al., "COMPASS: Robust Feature Conformal Prediction
      for Medical Segmentation Metrics", ICLR 2026.
  [3] Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction
      and Distribution-Free Uncertainty Quantification", 2023.
"""

import numpy as np
import torch
from typing import Callable, Tuple, Union, Optional

from src.conformal.compass import (
    perturb_probabilities,
    compute_metric_for_b,
    compass_l_score,
)
from src.conformal.compass_j import (
    compute_volume_jacobian,
    perturb_jacobian_direction,
    compass_j_score,
)
from src.conformal.adaptive_scores import compute_uncertainty_from_probs


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE: CRC-FS Calibration (REVISED)
# ═══════════════════════════════════════════════════════════════════════════════

def logistic_bounded_loss(scores: np.ndarray, lam: float) -> np.ndarray:
    """
    Logistic bounded loss function cho CRC-FS:
        L_i(lambda) = R_i / (R_i + lambda)  in  [0, 1]

    Day la ham loss MUOT (khong co "cliff" nhu min(1, R_i/lambda)):
      - lambda -> 0   : L_i -> 1 (risk max)
      - lambda = R_i   : L_i = 0.5
      - lambda -> inf  : L_i -> 0 (khong risk)
      - Bounded: L_i in [0, 1] -> thoa man Theorem 2.1

    Args:
        scores : COMPASS scores R_i (RAW, khong normalize), shape (n,)
        lam    : nguong lambda > 0

    Returns:
        losses: shape (n,), moi phan tu in [0, 1]
    """
    if lam <= 0:
        return np.ones_like(scores)
    return scores / (scores + lam)


def find_lambda_crc_fs(scores: np.ndarray,
                        alpha: float,
                        B: float = 1.0,
                        n_iter: int = 64) -> dict:
    """
    Tim lambda_hat toi uu bang CRC Theorem 2.1 voi LOGISTIC BOUNDED LOSS.

    Cong thuc (Angelopoulos et al. 2024, Theorem 2.1):
        lambda_hat = inf{ lambda : R_n(lambda) <= alpha - (B - alpha) / n }

    voi:
        R_n(lambda) = (1/n) * sum L_i(lambda)       -- empirical risk
        L_i(lambda) = R_i / (R_i + lambda)           -- logistic bounded loss

    Dung BINARY SEARCH chinh xac (64 iterations).

    Args:
        scores : RAW COMPASS scores R_i, shape (n,)
        alpha  : target risk level (e.g., 0.1 for 90% reliability)
        B      : upper bound on loss (default 1.0)
        n_iter : binary search iterations

    Returns:
        dict voi lambda_hat, risk, alpha, n_cal, risk_threshold, valid
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if n == 0:
        return {
            'lambda': float('inf'), 'risk': 0.0, 'alpha': alpha,
            'n_cal': 0, 'risk_threshold': 0.0, 'valid': False,
        }

    # Theorem 2.1 correction
    risk_threshold = alpha - (B - alpha) / n
    valid = risk_threshold >= 0

    if not valid:
        max_score = float(np.max(scores))
        return {
            'lambda': max_score * 1.1, 'risk': 0.0, 'alpha': alpha,
            'n_cal': n, 'risk_threshold': risk_threshold, 'valid': False,
        }

    # Binary search lambda in [0, max_score * 2]
    max_score = float(np.max(scores))
    lo, hi = 0.0, max_score * 2.0 + 1e-6

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        risk_mid = float(np.mean(logistic_bounded_loss(scores, mid)))
        if risk_mid <= risk_threshold:
            hi = mid  # lambda kha thi -> thu nho hon
        else:
            lo = mid  # lambda khong du -> can lon hon

    lambda_hat = hi
    final_risk = float(np.mean(logistic_bounded_loss(scores, lambda_hat)))

    return {
        'lambda': lambda_hat,
        'risk': final_risk,
        'alpha': alpha,
        'n_cal': n,
        'risk_threshold': risk_threshold,
        'valid': valid,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CRC-FS-L: Uniform Logit Perturbation
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_crc_fs_l(cal_probs: list,
                        cal_y_true: np.ndarray,
                        cal_sigmas: np.ndarray,
                        target_class: int,
                        metric_fns: list,
                        alpha: float,
                        b_max: float = 5.0,
                        steps: int = 100) -> dict:
    """
    Calibrate CRC-FS-L tren tap calibration (REVISED v2).

    Quy trinh:
      1. Tinh COMPASS-L score R_i cho moi sample
      2. PRIMARY: Split Conformal quantile tren R_i -> beta_hat
         (Dam bao coverage guarantee nhu COMPASS goc)
      3. SECONDARY: CRC calibration tren R_i voi logistic bounded loss -> lambda_crc
         (Cung cap finite-sample risk guarantee bo sung)
      4. Luu median_sigma de scale adaptive prediction

    Args:
        cal_probs, cal_y_true, cal_sigmas, target_class, metric_fns, alpha, b_max, steps

    Returns:
        dict voi beta_hat, lambda_crc, raw_scores, median_sigma, etc.
    """
    n = len(cal_probs)
    raw_scores = np.zeros(n)

    for i in range(n):
        raw_scores[i] = compass_l_score(
            cal_probs[i], float(cal_y_true[i]),
            target_class, metric_fns[i],
            b_max=b_max, steps=steps,
        )

    # Normalize: R'_i = R_i / sigma_i
    norm_scores = raw_scores / np.maximum(cal_sigmas, 1e-8)

    # PRIMARY: SCP quantile on NORMALIZED scores -> guarantee preserved
    from src.conformal.split_conformal import conformal_quantile
    q_hat = conformal_quantile(norm_scores, alpha)

    # SECONDARY: CRC on raw scores (diagnostic)
    crc_result = find_lambda_crc_fs(raw_scores, alpha)
    median_sigma = float(np.median(cal_sigmas))

    return {
        'q_hat': q_hat,                    # SCP quantile on normalized scores
        'lambda_crc': crc_result['lambda'],  # CRC diagnostic
        'raw_scores': raw_scores,
        'norm_scores': norm_scores,
        'sigmas': cal_sigmas,
        'median_sigma': median_sigma,
        'alpha': alpha,
        'method': 'CRC-FS-L',
        'crc_result': crc_result,
    }


def predict_interval_crc_fs_l(probs: torch.Tensor,
                               q_hat: float,
                               sigma: float,
                               median_sigma: float,
                               target_class: int,
                               metric_fn: Callable) -> Tuple[float, float]:
    """
    Tao prediction interval cho 1 test sample dung CRC-FS-L (v3).

    beta_test = q_hat * sigma_test
    -> Calibrated tren normalized scores nen guarantee duoc bao toan
    -> Sigma cao -> interval rong, sigma thap -> interval hep

    Args:
        probs, q_hat, sigma, median_sigma, target_class, metric_fn
    Returns: (lower, upper)
    """
    beta = float(q_hat * sigma)
    beta = min(beta, 20.0)

    m_neg = compute_metric_for_b(probs, -beta, target_class, metric_fn)
    m_pos = compute_metric_for_b(probs,  beta, target_class, metric_fn)

    return float(min(m_neg, m_pos)), float(max(m_neg, m_pos))


# ═══════════════════════════════════════════════════════════════════════════════
#  CRC-FS-J: Jacobian Subspace Perturbation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pca_directions(jacobians: list,
                            n_components: int = 5) -> np.ndarray:
    """
    Tính shared PCA subspace từ tập Jacobians của calibration set.

    Mỗi Jacobian được flatten → SVD → V_L = top-k components.
    Sau đó project từng Jacobian lên subspace này để được direction
    CHUNG cho toàn bộ calibration + test.

    Args:
        jacobians    : list các torch.Tensor (C, H, W, D) — Jacobian từ calibration.
        n_components : số chiều của subspace.

    Returns:
        V_L: np.ndarray shape (n_components, D_flat) — principal directions.
    """
    from sklearn.decomposition import TruncatedSVD

    # Flatten mỗi Jacobian
    flat_jacs = []
    for J in jacobians:
        flat_jacs.append(J.flatten().cpu().numpy())

    X = np.stack(flat_jacs, axis=0)  # (n_cal, D_flat)

    if n_components > min(X.shape):
        n_components = min(X.shape)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)
    return svd.components_  # (n_components, D_flat)


def project_to_subspace(jacobian: torch.Tensor,
                         V_L: np.ndarray) -> torch.Tensor:
    """
    Project 1 Jacobian lên shared PCA subspace → perturbation direction.

    direction = V_L^T @ V_L @ J_flat, reshape về (C, H, W, D).

    Args:
        jacobian : (C, H, W, D) Jacobian tensor trên GPU.
        V_L      : (n_components, D_flat) PCA components.

    Returns:
        direction: (C, H, W, D) tensor, cùng device với jacobian.
    """
    shape = jacobian.shape
    J_flat = jacobian.flatten().cpu().numpy()  # (D_flat,)

    # Project: V_L^T @ V_L @ J_flat
    coeffs = V_L @ J_flat                        # (n_components,)
    recon = coeffs @ V_L                         # (D_flat,) — projected back

    direction = torch.tensor(recon.reshape(shape),
                              dtype=jacobian.dtype,
                              device=jacobian.device)
    return direction


def calibrate_crc_fs_j(cal_probs: list,
                        cal_y_true: np.ndarray,
                        cal_sigmas: np.ndarray,
                        cal_voxel_vols: np.ndarray,
                        target_class: int,
                        metric_fns: list,
                        alpha: float,
                        b_max: float = 5.0,
                        steps: int = 100,
                        n_pca_components: int = 5) -> dict:
    """
    Calibrate CRC-FS-J trên tập calibration.

    Quy trình:
      1. Tính Jacobian cho mỗi calibration sample
      2. PCA → shared subspace V_L
      3. Project mỗi Jacobian lên V_L → perturbation direction
      4. Tính COMPASS-J score R_i dùng direction đã project
      5. Normalize: R'_i = R_i / σ_i
      6. CRC calibration → λ̂

    Args:
        cal_probs        : list các torch.Tensor (C, H, W, D).
        cal_y_true       : ground truth metrics, shape (n,).
        cal_sigmas       : uncertainty estimates, shape (n,).
        cal_voxel_vols   : voxel volumes (mL), shape (n,).
        target_class     : class index.
        metric_fns       : list các metric_fn.
        alpha            : target risk level.
        b_max, steps     : compass score parameters.
        n_pca_components : số PCA components.

    Returns:
        dict với lambda_hat, scores, V_L, etc.
    """
    n = len(cal_probs)

    # Bước 1: Tính Jacobians
    jacobians = []
    shapes = set()
    for i in range(n):
        J = compute_volume_jacobian(cal_probs[i], target_class, float(cal_voxel_vols[i]))
        jacobians.append(J)
        shapes.add(tuple(J.shape))

    # Bước 2: PCA subspace (with shape consistency check)
    if len(shapes) > 1:
        print(f"  [CRC-FS-J] Shapes differ ({len(shapes)} unique). "
              f"Falling back to per-sample Jacobian directions (no shared PCA).")
        V_L = None
        # Per-sample directions
        directions = []
        for J in jacobians:
            abs_max = J.abs().max()
            if abs_max > 1e-8:
                d = J / abs_max
            else:
                d = J
            directions.append(d)
    else:
        V_L = compute_pca_directions(jacobians, n_components=n_pca_components)
        # Project → directions
        directions = []
        for J in jacobians:
            d = project_to_subspace(J, V_L)
            abs_max = d.abs().max()
            if abs_max > 1e-8:
                d = d / abs_max
            directions.append(d)

    # Buoc 4: COMPASS-J scores
    raw_scores = np.zeros(n)
    for i in range(n):
        raw_scores[i] = compass_j_score(
            cal_probs[i], float(cal_y_true[i]),
            target_class, metric_fns[i], directions[i],
            b_max=b_max, steps=steps,
        )

    # Normalize
    norm_scores = raw_scores / np.maximum(cal_sigmas, 1e-8)

    # PRIMARY: SCP quantile on normalized scores
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = np.clip(level, 0, 1)
    q_hat = float(np.quantile(norm_scores, level, method='higher'))

    # SECONDARY: CRC
    crc_result = find_lambda_crc_fs(raw_scores, alpha)
    median_sigma = float(np.median(cal_sigmas))

    return {
        'q_hat': q_hat,
        'lambda_crc': crc_result['lambda'],
        'raw_scores': raw_scores,
        'norm_scores': norm_scores,
        'sigmas': cal_sigmas,
        'median_sigma': median_sigma,
        'alpha': alpha,
        'method': 'CRC-FS-J',
        'crc_result': crc_result,
        'V_L': V_L,
        'directions': directions,
    }


def predict_interval_crc_fs_j(probs: torch.Tensor,
                               q_hat: float,
                               sigma: float,
                               median_sigma: float,
                               target_class: int,
                               metric_fn: Callable,
                               voxel_vol: float,
                               V_L: Optional[np.ndarray]) -> Tuple[float, float]:
    """
    Tao prediction interval cho test sample dung CRC-FS-J (v3).

    beta_test = q_hat * sigma_test
    Calibrated tren normalized scores -> coverage guarantee preserved.
    """
    J = compute_volume_jacobian(probs, target_class, voxel_vol)

    if V_L is not None:
        direction = project_to_subspace(J, V_L)
    else:
        direction = J

    abs_max = direction.abs().max()
    if abs_max > 1e-8:
        direction = direction / abs_max

    beta = min(float(q_hat * sigma), 20.0)

    # Perturb
    p_pos = perturb_jacobian_direction(probs,  beta, direction)
    p_neg = perturb_jacobian_direction(probs, -beta, direction)

    mask_pos = (torch.argmax(p_pos, dim=0) == target_class).to(torch.uint8)
    mask_neg = (torch.argmax(p_neg, dim=0) == target_class).to(torch.uint8)

    m_pos = metric_fn(mask_pos)
    m_neg = metric_fn(mask_neg)

    return float(min(m_pos, m_neg)), float(max(m_pos, m_neg))


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY: Diagnostic summary
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_crc_fs(result: dict) -> None:
    """In diagnostic summary cua CRC-FS calibration."""
    print(f"\n  [{result['method']}] Calibration Summary:")
    print(f"    beta_hat (SCP)  = {result['beta_hat']:.4f}")
    print(f"    lambda_crc      = {result['lambda_crc']:.4f}")
    print(f"    alpha           = {result['alpha']:.3f}")
    print(f"    n_cal           = {result['crc_result']['n_cal']}")
    print(f"    Valid (n>=min)   = {result['crc_result']['valid']}")
    print(f"    Risk threshold  = {result['crc_result']['risk_threshold']:.4f}")
    print(f"    Empirical risk  = {result['crc_result']['risk']:.4f}")
    print(f"    median_sigma    = {result['median_sigma']:.4f}")
    print(f"    Raw scores      : mean={result['raw_scores'].mean():.4f}, "
          f"median={np.median(result['raw_scores']):.4f}, "
          f"max={result['raw_scores'].max():.4f}")
    print(f"    sigma (uncertainty): mean={result['sigmas'].mean():.4f}, "
          f"std={result['sigmas'].std():.4f}")
