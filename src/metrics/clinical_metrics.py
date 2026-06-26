"""
src/metrics/clinical_metrics.py

Tinh nonconformity scores tu predicted va true clinical metrics.

Non-conformity score la hat nhan cua conformal prediction:
    score_i = |predicted_metric_i - true_metric_i|        (absolute)
    score_i = |predicted_metric_i - true_metric_i| / true (relative)
"""

import numpy as np
from typing import Literal


MetricName = Literal['LV_EDV', 'LV_ESV', 'LV_EF', 'RV_EDV', 'RV_ESV', 'RV_EF', 'Myo_mass']
ErrorMode  = Literal['abs', 'rel']


def nonconformity_score(pred: np.ndarray,
                        true: np.ndarray,
                        mode: ErrorMode = 'abs') -> np.ndarray:
    """
    Tinh nonconformity score cho mang predictions.

    Args:
        pred : predicted metrics, shape (N,)
        true : ground truth metrics, shape (N,)
        mode : 'abs' -> |pred - true|
               'rel' -> |pred - true| / true

    Returns:
        scores: shape (N,), non-negative
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    assert pred.shape == true.shape

    if mode == 'abs':
        return np.abs(pred - true)
    elif mode == 'rel':
        # Tranh chia cho 0
        denom = np.where(np.abs(true) > 1e-6, np.abs(true), 1e-6)
        return np.abs(pred - true) / denom
    else:
        raise ValueError(f"mode phai la 'abs' hoac 'rel', nhan duoc: {mode}")


def simulate_predictions(true_metrics: np.ndarray,
                         rel_noise: float = 0.10,
                         seed: int = 42) -> np.ndarray:
    """
    Gia lap model predictions bang cach them Gaussian noise.

    Dung de test pipeline truoc khi co real model.
    rel_noise=0.10 tuong duong sai so 10% (phu hop voi nnU-Net tren ACDC).

    Args:
        true_metrics : ground truth values, shape (N,)
        rel_noise    : do lech chuan tuong doi
        seed         : random seed

    Returns:
        pred_metrics : shape (N,)
    """
    rng  = np.random.default_rng(seed)
    noise = rng.normal(0, rel_noise, size=len(true_metrics))
    pred  = true_metrics * (1 + noise)
    return np.clip(pred, 0, None)  # metric luon >= 0
