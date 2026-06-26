"""
src/conformal/split_conformal.py

Split Conformal Prediction cho clinical metrics.

Theory (Vovk et al. 2005, Angelopoulos & Bates 2022):
    Given calibration scores s_1, ..., s_n and alpha in (0,1):

    q = ceil((n+1)(1-alpha)) / n  quantile of scores

    Prediction interval for new sample: [y_hat - q, y_hat + q]

    Guarantee: P(y_new in interval) >= 1 - alpha
    (distribution-free, valid for any exchangeable data)
"""

import numpy as np
from typing import Tuple


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Tinh conformal quantile tu calibration scores.

    Args:
        scores : nonconformity scores tren calibration set, shape (n,)
        alpha  : miscoverage level (vi du 0.1 cho 90% coverage)

    Returns:
        q_hat  : conformal quantile (scalar)
    """
    n = len(scores)
    if n == 0:
        return float('inf')
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = np.clip(level, 0, 1)
    return float(np.quantile(scores, level, method='higher'))


def calibrate(cal_true: np.ndarray,
              cal_pred: np.ndarray,
              alpha: float,
              mode: str = 'abs') -> dict:
    """
    Calibrate Split Conformal Prediction.

    Args:
        cal_true : true metrics tren calibration set, shape (n,)
        cal_pred : predicted metrics tren calibration set, shape (n,)
        alpha    : miscoverage level
        mode     : 'abs' hoac 'rel'

    Returns:
        result dict voi:
            'q_hat'   : conformal quantile
            'scores'  : calibration nonconformity scores
            'alpha'   : alpha duoc su dung
            'n_cal'   : so luong calibration samples
    """
    from src.metrics.clinical_metrics import nonconformity_score

    scores = nonconformity_score(cal_pred, cal_true, mode=mode)
    q_hat  = conformal_quantile(scores, alpha)

    return {
        'q_hat'  : q_hat,
        'scores' : scores,
        'alpha'  : alpha,
        'n_cal'  : len(scores),
        'mode'   : mode,
    }


def predict_interval(test_pred: np.ndarray,
                     q_hat: float,
                     mode: str = 'abs') -> Tuple[np.ndarray, np.ndarray]:
    """
    Tao prediction intervals cho test set.

    Args:
        test_pred : predicted metrics tren test set, shape (m,)
        q_hat     : conformal quantile tu buoc calibrate()
        mode      : 'abs' hoac 'rel'

    Returns:
        (lower, upper) : moi cai shape (m,)
    """
    test_pred = np.asarray(test_pred, dtype=float)

    if mode == 'abs':
        lower = test_pred - q_hat
        upper = test_pred + q_hat
    elif mode == 'rel':
        lower = test_pred * (1 - q_hat)
        upper = test_pred * (1 + q_hat)
    else:
        raise ValueError(f"mode phai la 'abs' hoac 'rel'")

    return np.maximum(lower, 0), upper  # metric >= 0


def evaluate_coverage(test_true: np.ndarray,
                      lower: np.ndarray,
                      upper: np.ndarray) -> dict:
    """
    Danh gia empirical coverage va interval width.

    Args:
        test_true : true metrics tren test set, shape (m,)
        lower     : lower bounds, shape (m,)
        upper     : upper bounds, shape (m,)

    Returns:
        dict voi:
            'coverage'       : empirical coverage (0-1)
            'mean_width'     : trung binh do rong interval
            'median_width'   : trung vi do rong interval
            'n_test'         : so luong test samples
    """
    covered    = (test_true >= lower) & (test_true <= upper)
    widths     = upper - lower

    return {
        'coverage'     : float(np.mean(covered)),
        'mean_width'   : float(np.mean(widths)),
        'median_width' : float(np.median(widths)),
        'n_covered'    : int(np.sum(covered)),
        'n_test'       : len(test_true),
    }
