"""
src/conformal/compass_j.py

COMPASS-J: Jacobian-based Feature Perturbation
(Cheung et al., arXiv:2509.22240, ICLR 2026 - Section 2.2)

Y tuong cot loi:
    COMPASS-L cong deu hang so beta vao toan bo logit → khong hieu qua,
    lam interval phinh to (over-coverage).

    COMPASS-J dung dao ham (Jacobian) cua metric Volume m
    theo logit truoc softmax:

        dV/d_logit[target, x] = p_target(x) * (1 - p_target(x)) * voxel_vol

    Day chinh la variance cua Bernoulli distribution tai moi voxel.
    Cac voxel gan ranh gioi quyet dinh (p_target ≈ 0.5) co gradient
    lon nhat → thuong keo theo su thay doi lon nhat trong Volume.

    Perturbation chi tap trung vao vung ranh gioi (boundary) thay
    vi toan bo anh → cung mot beta nhung Width hep hon nhieu.

    Formula:
        logit'(x) = logit(x) + beta * J_normalized(x)
        p'(x) = softmax(logit'(x))

Complexity: O(n_cal * steps) per fold
"""

import numpy as np
import torch
from typing import Callable, Tuple, Optional
from sklearn.decomposition import TruncatedSVD


# ─────────────────────────────────────────────
#  1.  Tinh Jacobian dV/d_logit (Boundary-weighted direction)
# ─────────────────────────────────────────────

def compute_volume_jacobian(
    probs: torch.Tensor,
    target_class: int,
    voxel_vol: float,
) -> torch.Tensor:
    """
    Tinh dao ham cua Volume theo LOGIT (truoc softmax).

    Dung soft-argmax approximation:
        V(p) ≈ voxel_vol * sum_x p_target(x)

    Dao ham theo logit[target, x] (chain rule qua softmax):
        dV / d_logit[target, x] = p_target(x) * (1 - p_target(x)) * voxel_vol

    Day la phuong sai Bernoulli tai moi voxel:
        - p ≈ 0.5 (ranh gioi quyet dinh) → gradient LON → thay doi Volume nhieu
        - p ≈ 0   hoac p ≈ 1 (vung chac chan) → gradient nho → it anh huong

    Ket qua: huong perturbation chi tap trung vao RANH GIOI can organ,
    giup interval hep hon so voi COMPASS-L (cong deu toan bo).

    Args:
        probs        : (C, H, W, D) softmax probabilities tren GPU.
        target_class : class index cua co quan muc tieu (e.g., 3 = LV).
        voxel_vol    : the tich mot voxel (mL).

    Returns:
        J : Jacobian tensor, shape = (C, H, W, D), cung device voi probs.
            Chi khac 0 o channel target_class.
    """
    eps = 1e-7
    p_target = torch.clamp(probs[target_class], eps, 1.0 - eps)  # (H, W, D)

    # Variance of Bernoulli: p * (1 - p) - day la Jacobian dV/d_logit[target]
    jacobian_target = p_target * (1.0 - p_target) * voxel_vol    # (H, W, D)

    J = torch.zeros_like(probs)  # (C, H, W, D)
    J[target_class] = jacobian_target
    return J


# ─────────────────────────────────────────────
#  2.  PCA de tim dominant subspace V_L
# ─────────────────────────────────────────────

def compute_pca_subspace(
    jacobians: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """
    Tinh ma tran khong gian con (dominant subspace) tu tap Jacobians
    cua tap calibration.

    Moi Jacobian duoc flatten thanh vector roi dua vao SVD.
    Ket qua la top-k left singular vectors (principal components).

    Args:
        jacobians    : shape (n_cal, D_flat) - moi hang la 1 Jacobian da flatten.
        n_components : so chieu cua subspace.

    Returns:
        V_L : shape (n_components, D_flat) - cac principal directions.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(jacobians)
    return svd.components_   # (n_components, D_flat)


def project_jacobian_to_subspace(
    jacobian: torch.Tensor,
    V_L: np.ndarray,
) -> torch.Tensor:
    """
    Project 1 Jacobian len shared PCA subspace → perturbation direction.

    direction = V_L^T @ V_L @ J_flat
    (project len subspace roi reconstruct)

    Args:
        jacobian : (C, H, W, D) Jacobian tensor tren GPU.
        V_L      : (n_components, D_flat) PCA components tu calibration.

    Returns:
        direction: (C, H, W, D) tensor, cung device voi jacobian.
    """
    shape = jacobian.shape
    J_flat = jacobian.flatten().cpu().numpy()

    # Project: coeffs @ V_L = (V_L @ J_flat) @ V_L
    coeffs = V_L @ J_flat                       # (n_components,)
    recon = coeffs @ V_L                        # (D_flat,) — projected

    direction = torch.tensor(
        recon.reshape(shape),
        dtype=jacobian.dtype,
        device=jacobian.device,
    )
    return direction


def compute_shared_directions(
    cal_probs: list,
    target_class: int,
    voxel_vols: list,
    n_components: int = 5,
) -> Tuple[Optional[np.ndarray], list]:
    """
    Tinh shared PCA subspace tu calibration Jacobians
    VA project tung sample len subspace do.

    Neu cac sample co shape khac nhau → fallback ve per-sample direction.

    Args:
        cal_probs    : list cac torch.Tensor (C, H, W, D).
        target_class : class index.
        voxel_vols   : list voxel volumes (mL).
        n_components : so PCA components.

    Returns:
        V_L        : (n_components, D_flat) shared subspace, hoac None neu fallback.
        directions : list cac torch.Tensor (C, H, W, D) da project + normalize.
    """
    # Buoc 1: Tinh Jacobians
    jacobians = []
    shapes = set()
    for i, probs in enumerate(cal_probs):
        J = compute_volume_jacobian(probs, target_class, float(voxel_vols[i]))
        jacobians.append(J)
        shapes.add(tuple(J.shape))

    # Kiem tra shape consistency
    if len(shapes) > 1:
        print(f"  [COMPASS-J] Shapes differ across samples ({len(shapes)} unique). "
              f"Falling back to per-sample Jacobian directions (no shared PCA).")
        # Fallback: per-sample directions (normalized)
        directions = []
        for J in jacobians:
            abs_max = J.abs().max()
            if abs_max > 1e-8:
                d = J / abs_max
            else:
                d = J
            directions.append(d)
        return None, directions

    shape0 = jacobians[0].shape

    # Buoc 2: Flatten + stack
    flat_jacs = [J.flatten().cpu().numpy() for J in jacobians]
    X = np.stack(flat_jacs, axis=0)  # (n_cal, D_flat)

    # Buoc 3: PCA
    V_L = compute_pca_subspace(X, n_components=n_components)

    # Buoc 4: Project + normalize
    directions = []
    for J in jacobians:
        d = project_jacobian_to_subspace(J, V_L)
        abs_max = d.abs().max()
        if abs_max > 1e-8:
            d = d / abs_max
        directions.append(d)

    return V_L, directions


# ─────────────────────────────────────────────
#  3.  Perturb theo huong Jacobian (COMPASS-J)
# ─────────────────────────────────────────────

def perturb_jacobian_direction(
    probs: torch.Tensor,
    beta: float,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Perturb LOGITS theo huong Jacobian roi chuyen lai qua softmax:
        logit'(x) = logit(p(x)) + beta * direction(x)
        p'(x) = softmax(logit'(x))

    Vi direction tap trung o vung ranh gioi (p ≈ 0.5),
    nen perturbation co hieu qua cao nhat o nhung voxel nay.

    Args:
        probs     : (C, H, W, D) softmax probabilities.
        beta      : Perturbation magnitude (scalar).
        direction : (C, H, W, D) Jacobian direction (normalized).

    Returns:
        Perturbed probabilities, shape (C, H, W, D).
    """
    eps = 1e-7
    probs_safe = torch.clamp(probs, eps, 1.0 - eps)

    # Chuyen sang logit-space: log(p_k) de tinh tuong duong logit
    log_p = torch.log(probs_safe)  # log-probabilities, shape (C,H,W,D)

    # Cong perturbation vao log-probability (tuong duong logit shift)
    log_p_perturbed = log_p + beta * direction

    # Chuyen lai qua softmax
    p_new = torch.softmax(log_p_perturbed, dim=0)

    return p_new


# ─────────────────────────────────────────────
#  4.  COMPASS-J Score cho mot sample
# ─────────────────────────────────────────────

def compass_j_score(
    probs: torch.Tensor,
    y_true: float,
    target_class: int,
    metric_fn: Callable,
    direction: torch.Tensor,
    b_max: float = 10.0,
    steps: int = 100,
) -> float:
    """
    Tinh COMPASS-J nonconformity score cho 1 sample.

    R_i = min{beta >= 0 : y_true in [m(-beta*v), m(+beta*v)]}

    Khac COMPASS-L: perturb chi vao vung ranh gioi (direction = Jacobian),
    nen cung mot beta se tao ra dai rong hon → can beta nho hon de bao phu
    → Width nho hon toan cuoc.

    Args:
        probs        : (C, H, W, D) softmax probabilities tren GPU.
        y_true       : Ground truth metric.
        target_class : Class index.
        metric_fn    : Ham tinh metric tu binary mask (torch.Tensor).
        direction    : (C, H, W, D) Jacobian direction (da normalized).
        b_max        : Pham vi tim kiem beta toi da.
        steps        : So buoc tim kiem.

    Returns:
        float: Nonconformity score R_i.
    """
    # Normalize direction by L-inf (max absolute value)
    # So that beta = maximum logit shift at the most sensitive voxel
    # This makes beta comparable to COMPASS-L (which shifts logit by beta uniformly)
    abs_max = direction.abs().max()
    if abs_max > 1e-8:
        direction_n = direction / abs_max
    else:
        direction_n = direction

    betas = np.linspace(0, b_max, steps)

    for beta in betas:
        # Perturb theo ca 2 huong +-beta
        p_pos = perturb_jacobian_direction(probs,  beta, direction_n)
        p_neg = perturb_jacobian_direction(probs, -beta, direction_n)

        mask_pos = (torch.argmax(p_pos, dim=0) == target_class).to(torch.uint8)
        mask_neg = (torch.argmax(p_neg, dim=0) == target_class).to(torch.uint8)

        m_pos = metric_fn(mask_pos)
        m_neg = metric_fn(mask_neg)

        lower = min(m_pos, m_neg)
        upper = max(m_pos, m_neg)

        if lower <= y_true <= upper:
            return float(beta)

    return float(b_max)


# ─────────────────────────────────────────────
#  5.  Calibrate COMPASS-J
# ─────────────────────────────────────────────

def calibrate_compass_j(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Calibrate COMPASS-J dung split conformal quantile.
    Uses the conservative quantile from split_conformal.py.
    """
    from src.conformal.split_conformal import conformal_quantile
    scores = np.asarray(scores, dtype=float)
    return conformal_quantile(scores, alpha)


# ─────────────────────────────────────────────
#  6.  Predict interval với COMPASS-J
# ─────────────────────────────────────────────

def predict_interval_compass_j(
    probs: torch.Tensor,
    beta_hat: float,
    target_class: int,
    metric_fn: Callable,
    direction: torch.Tensor,
) -> Tuple[float, float]:
    """
    Tao prediction interval cho 1 test sample dung COMPASS-J.

    Args:
        probs        : Softmax probabilities.
        beta_hat     : Calibrated beta.
        target_class : Class index.
        metric_fn    : Metric function.
        direction    : Jacobian perturbation direction (se tu normalize).

    Returns:
        (lower, upper): bounds of the prediction interval.
    """
    norm = direction.norm()
    if norm > 1e-8:
        direction_n = direction / norm
    else:
        direction_n = direction

    # Use L-inf normalization for predict too
    abs_max = direction.abs().max()
    if abs_max > 1e-8:
        direction_n = direction / abs_max
    else:
        direction_n = direction

    p_pos = perturb_jacobian_direction(probs,  beta_hat, direction_n)
    p_neg = perturb_jacobian_direction(probs, -beta_hat, direction_n)

    mask_pos = (torch.argmax(p_pos, dim=0) == target_class).to(torch.uint8)
    mask_neg = (torch.argmax(p_neg, dim=0) == target_class).to(torch.uint8)

    m_pos = metric_fn(mask_pos)
    m_neg = metric_fn(mask_neg)

    return float(min(m_pos, m_neg)), float(max(m_pos, m_neg))
