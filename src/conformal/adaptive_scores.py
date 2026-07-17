"""
src/conformal/adaptive_scores.py

Normalized / Adaptive Nonconformity Scores để tối ưu Width
trong khi vẫn đảm bảo Coverage >= (1 - alpha).

Ý tưởng cốt lõi (từ bài báo 1 + 2):
    Thay vì cộng một constant q_hat cho mọi bệnh nhân,
    ta chuẩn hóa sai số theo "độ khó" cục bộ:

        Normalized score: R_i = |y_i - ŷ_i| / σ_i

    Trong đó σ_i là ước lượng độ không chắc chắn của mô hình
    cho bệnh nhân thứ i (computed from softmax entropy or
    standard deviation của probabilities).

    Khi dự đoán:
        Interval cho bệnh nhân j: [ŷ_j - q̂·σ_j, ŷ_j + q̂·σ_j]

    Điều này tạo ra interval THÍCH ỨNG (adaptive):
    - Bệnh nhân dễ (mô hình chắc chắn, σ nhỏ) → interval hẹp
    - Bệnh nhân khó (mô hình không chắc, σ lớn) → interval rộng
    → Coverage được phân bổ thông minh hơn, trung bình Width nhỏ hơn.

References:
    - Angelopoulos & Bates (2023), Section 3.2: Normalized scores
    - Romano et al. (2019), CQR: Conformalized Quantile Regression
"""

import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────
#  1.  Ước lượng σ_i từ softmax probabilities
# ─────────────────────────────────────────────

def compute_uncertainty_from_probs(
    probs: np.ndarray,
    target_class: int,
    method: str = "entropy"
) -> float:
    """
    Tính độ không chắc chắn σ_i cho một sample từ softmax probabilities.

    Args:
        probs        : (C, H, W, D) softmax probabilities.
        target_class : class index của cơ quan mục tiêu (e.g., 3 = LV).
        method       : 'entropy' | 'std' | 'margin'
                       - entropy: Shannon entropy trung bình trên toàn ảnh
                       - std    : std của xác suất target class trên toàn ảnh
                       - margin : 1 - (p_max - p_second_max) (trung bình)

    Returns:
        sigma : scalar, giá trị không âm đại diện cho độ không chắc chắn.
    """
    eps = 1e-8
    probs = np.clip(probs, eps, 1.0)           # (C, H, W, D)

    if method == "entropy":
        # Shannon entropy trung bình trên mọi voxel
        H = -np.sum(probs * np.log(probs), axis=0)  # (H, W, D)
        sigma = float(np.mean(H))

    elif method == "std":
        # Std của xác suất class mục tiêu (đo độ phân tán prediction)
        p_target = probs[target_class]              # (H, W, D)
        sigma = float(np.std(p_target))

    elif method == "margin":
        # Margin = 1 - (p1 - p2): confidence margin giữa top-2 classes
        sorted_p = np.sort(probs, axis=0)[::-1]    # (C, H, W, D) desc
        margin = 1.0 - (sorted_p[0] - sorted_p[1]) # (H, W, D)
        sigma = float(np.mean(margin))

    else:
        raise ValueError(f"method phải là 'entropy', 'std', hoặc 'margin'")

    # Đảm bảo sigma không bằng 0 (tránh chia 0)
    return max(sigma, eps)


# ─────────────────────────────────────────────
#  2.  Normalized nonconformity score
# ─────────────────────────────────────────────

def normalized_score(
    y_true: float,
    y_pred: float,
    sigma: float,
) -> float:
    """
    Tính normalized nonconformity score:
        R_i = |y_true - y_pred| / sigma_i

    Args:
        y_true : Ground truth metric (e.g., volume mL).
        y_pred : Predicted metric.
        sigma  : Ước lượng uncertainty của mô hình cho sample này.

    Returns:
        float: Normalized score (non-negative).
    """
    return abs(y_true - y_pred) / max(sigma, 1e-8)


# ─────────────────────────────────────────────
#  3.  Calibration với Normalized scores
# ─────────────────────────────────────────────

def calibrate_normalized(
    cal_true: np.ndarray,
    cal_pred: np.ndarray,
    cal_sigmas: np.ndarray,
    alpha: float,
) -> dict:
    """
    Calibrate Adaptive Conformal Prediction dùng Normalized scores.

    Công thức (Angelopoulos & Bates, Eq. 3):
        q̂ = (ceil((n+1)(1-α)) / n) - quantile của {R_i}

    Args:
        cal_true   : Ground truth, shape (n,).
        cal_pred   : Predictions, shape (n,).
        cal_sigmas : Uncertainty estimates, shape (n,).
        alpha      : Target miscoverage rate (e.g., 0.1 for 90% coverage).

    Returns:
        dict với 'q_hat', 'scores', 'alpha', 'n_cal'.
    """
    cal_true   = np.asarray(cal_true, dtype=float)
    cal_pred   = np.asarray(cal_pred, dtype=float)
    cal_sigmas = np.asarray(cal_sigmas, dtype=float)

    scores = np.abs(cal_true - cal_pred) / np.maximum(cal_sigmas, 1e-8)
    n      = len(scores)

    from src.conformal.split_conformal import conformal_quantile
    q_hat = conformal_quantile(scores, alpha)

    return {
        'q_hat'   : q_hat,
        'scores'  : scores,
        'alpha'   : alpha,
        'n_cal'   : n,
        'method'  : 'normalized',
    }


# ─────────────────────────────────────────────
#  4.  Prediction interval với Adaptive Width
# ─────────────────────────────────────────────

def predict_interval_normalized(
    test_pred: np.ndarray,
    test_sigmas: np.ndarray,
    q_hat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo Adaptive Prediction Intervals:
        lower_j = ŷ_j - q̂ · σ_j
        upper_j = ŷ_j + q̂ · σ_j

    Args:
        test_pred   : Predicted metrics, shape (m,).
        test_sigmas : Uncertainty estimates, shape (m,).
        q_hat       : Calibrated threshold.

    Returns:
        (lower, upper): mỗi cái shape (m,).
    """
    test_pred   = np.asarray(test_pred, dtype=float)
    test_sigmas = np.asarray(test_sigmas, dtype=float)

    half_width = q_hat * test_sigmas
    lower = test_pred - half_width
    upper = test_pred + half_width

    return np.maximum(lower, 0), upper  # metric >= 0


# ─────────────────────────────────────────────
#  5.  Adaptive CRC với normalized loss
# ─────────────────────────────────────────────

def find_lambda_adaptive(
    cal_true: np.ndarray,
    cal_pred: np.ndarray,
    cal_sigmas: np.ndarray,
    alpha: float,
) -> dict:
    """
    Conformal Risk Control với normalized loss (Theorem 2.1, Angelopoulos et al. 2024).

    Loss function: L_i(λ) = max(0, R_i - λ) / R_i_max  ∈ [0, 1]
    (bounded loss, phù hợp với Theorem 2.1)

    Tìm λ nhỏ nhất sao cho:
        R̂_n(λ) = (1/n) Σ L_i(λ) ≤ α - (B - α)/n

    Với B = 1 (max loss).

    Args:
        cal_true   : Ground truth, shape (n,).
        cal_pred   : Predictions, shape (n,).
        cal_sigmas : Uncertainty estimates, shape (n,).
        alpha      : Target risk level.

    Returns:
        dict với 'lambda', 'risk', 'alpha', 'n_cal'.
    """
    cal_true   = np.asarray(cal_true, dtype=float)
    cal_pred   = np.asarray(cal_pred, dtype=float)
    cal_sigmas = np.asarray(cal_sigmas, dtype=float)

    # Normalized scores
    scores = np.abs(cal_true - cal_pred) / np.maximum(cal_sigmas, 1e-8)

    n = len(scores)
    B = 1.0  # bound on loss (loss_i = 1 nếu missed, 0 nếu covered)

    # Ngưỡng risk cho phép (Theorem 2.1)
    risk_threshold = alpha - (B - alpha) / n
    valid = risk_threshold >= 0

    max_lam = float(np.max(scores)) * 1.01

    if not valid:
        best_lam = max_lam
    else:
        # Binary search chính xác (thay vì grid search thô)
        lo, hi = 0.0, max_lam
        for _ in range(64):   # 64 iterations → sai số < max_lam / 2^64 ≈ 0
            mid = (lo + hi) / 2.0
            # Miscoverage rate tại mid
            risk_mid = float(np.mean(scores > mid))
            if risk_mid <= risk_threshold:
                hi = mid
            else:
                lo = mid
        best_lam = hi

    final_risk = float(np.mean(scores > best_lam))

    return {
        'lambda'    : best_lam,
        'risk'      : final_risk,
        'alpha'     : alpha,
        'n_cal'     : n,
        'valid'     : valid,
        'method'    : 'adaptive_crc',
    }
