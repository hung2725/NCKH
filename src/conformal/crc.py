"""
src/conformal/crc.py

Conformal Risk Control (CRC) cho clinical metrics.

Theory (Angelopoulos et al., ICLR 2024):
    Thay vi control coverage (nhu Split CP),
    CRC control mot ham loss tong quat L(lambda):

        E[L(lambda)] <= alpha

    Bang cach tim lambda nho nhat thoa man dieu kien tren
    tren calibration set.

    Voi clinical metrics:
        lambda = do rong interval
        L(lambda) = 1{y_true NOT in [y_pred - lambda, y_pred + lambda]}
                  = miscoverage indicator

    => CRC voi loss nay tuong duong Split CP nhung framework tong quat hon.
"""

import numpy as np
from typing import Callable


def risk_fn_miscoverage(true: np.ndarray,
                        pred: np.ndarray,
                        lam: float,
                        mode: str = 'abs') -> float:
    """
    Loss function: ty le miscoverage (khong nam trong interval).

    L(lambda) = P(|true - pred| > lambda)

    Args:
        true : true metrics, shape (n,)
        pred : predicted metrics, shape (n,)
        lam  : interval half-width (lambda)
        mode : 'abs' hoac 'rel'

    Returns:
        risk: ty le miscoverage (0-1)
    """
    from src.metrics.clinical_metrics import nonconformity_score
    scores = nonconformity_score(pred, true, mode=mode)
    return float(np.mean(scores > lam))


def find_lambda(cal_true: np.ndarray,
                cal_pred: np.ndarray,
                alpha: float,
                mode: str = 'abs',
                n_grid: int = 1000) -> dict:
    """
    Tim lambda nho nhat sao cho risk(lambda) <= alpha.

    Cong thuc dung (Angelopoulos et al. 2024, Theorem 1):
        R_hat(lambda) = (sum L_i(lambda) + 1) / (n + 1) <= alpha
        <=> sum L_i(lambda) + 1 <= alpha * (n + 1)
        <=> n_miscov + 1 <= alpha * (n + 1)
        <=> n_miscov <= alpha * (n + 1) - 1

    Chu y: neu n < ceil(1/alpha) - 1 = 9 (voi alpha=0.1),
    thi alpha*(n+1) - 1 < 0 => khong co lambda nao thoa man
    => tra ve max_lam (conservative fallback).

    Args:
        cal_true : true metrics, shape (n,)
        cal_pred : predicted metrics, shape (n,)
        alpha    : risk level (vi du 0.1 cho target 90%)
        mode     : 'abs' hoac 'rel'
        n_grid   : so diem trong grid tim kiem

    Returns:
        dict voi:
            'lambda'        : gia tri lambda tim duoc
            'risk'          : empirical risk tai lambda nay
            'alpha'         : alpha duoc su dung
            'n_cal'         : so luong calibration samples
            'max_allowed'   : so miscoverage toi da duoc phep
            'valid'         : True neu n du lon de CRC co the hoat dong
    """
    from src.metrics.clinical_metrics import nonconformity_score

    scores  = nonconformity_score(cal_pred, cal_true, mode=mode)
    n       = len(scores)
    max_lam = float(np.max(scores)) * 1.1

    # Grid cac gia tri lambda
    lambdas = np.linspace(0, max_lam, n_grid)

    # DUNG formula tu Angelopoulos et al. (2024) Theorem 1:
    #   R_hat(lambda) = (n_miscov + 1) / (n + 1) <= alpha
    #   => n_miscov <= alpha * (n + 1) - 1
    max_allowed = alpha * (n + 1) - 1   # so miscoverage toi da cho phep
    valid = max_allowed >= 0            # can n >= ceil(1/alpha) - 1

    if not valid:
        # Nhom qua nho, khong the dam bao guarantee -> tra ve max_lam (conservative)
        best_lam = max_lam
    else:
        best_lam = max_lam  # default conservative
        for lam in lambdas:
            n_miscov = int(np.sum(scores > lam))
            if n_miscov <= max_allowed:
                best_lam = lam
                break  # Tim thay lambda nho nhat thoa man

    final_risk = float(np.mean(scores > best_lam))

    return {
        'lambda'      : best_lam,
        'risk'        : final_risk,
        'alpha'       : alpha,
        'n_cal'       : n,
        'mode'        : mode,
        'max_allowed' : max_allowed,
        'valid'       : valid,
    }


def predict_interval_crc(test_pred: np.ndarray,
                          lam: float,
                          mode: str = 'abs'):
    """
    Tao prediction intervals dung lambda tu CRC.
    Interface tuong tu split_conformal.predict_interval().
    """
    from src.conformal.split_conformal import predict_interval
    return predict_interval(test_pred, lam, mode=mode)
