import numpy as np
import torch
from typing import Tuple, Callable, Union

def perturb_probabilities(probs: torch.Tensor, b: float, target_class: int = 3) -> torch.Tensor:
    """
    Applies a uniform logit shift (COMPASS-L) to the predicted probabilities.
    
    Args:
        probs (torch.Tensor): Tensor of shape (C, H, W, D) containing softmax probabilities.
        b (float): Perturbation magnitude.
        target_class (int): The class index to perturb (default 3 = LV).
        
    Returns:
        torch.Tensor: Perturbed probabilities, same shape as probs.
    """
    eps = 1e-7
    probs_safe = torch.clamp(probs, eps, 1.0 - eps)
    
    exp_b = float(np.exp(b))
    
    new_probs = probs_safe.clone()
    new_probs[target_class] *= exp_b
    
    # Re-normalize over the class dimension (dim 0)
    sum_probs = torch.sum(new_probs, dim=0, keepdim=True)
    new_probs /= sum_probs
    
    return new_probs

def compute_metric_for_b(probs: torch.Tensor, b: float, target_class: int, metric_fn: Callable) -> float:
    """
    Applies logit shift b, thresholding, and computes the clinical metric.
    """
    p_perturbed = perturb_probabilities(probs, b, target_class)
    # Argmax over class dimension to get the mask
    mask = torch.argmax(p_perturbed, dim=0)
    
    # metric_fn expects a binary mask for the target structure
    binary_mask = (mask == target_class).to(torch.uint8)
    return metric_fn(binary_mask)

def compass_l_score(probs: torch.Tensor, y_true: float, target_class: int, metric_fn: Callable, b_max: float = 10.0, steps: int = 50) -> float:
    """
    Computes the COMPASS nonconformity score R_i for a single sample using GPU.
    R_i = min { beta >= 0 : y_true in [m(-beta), m(+beta)] }
    """
    # We will search over beta in [0, b_max].
    betas = np.linspace(0, b_max, steps)
    
    for beta in betas:
        m_neg = compute_metric_for_b(probs, -beta, target_class, metric_fn)
        m_pos = compute_metric_for_b(probs, beta, target_class, metric_fn)
        
        lower = min(m_neg, m_pos)
        upper = max(m_neg, m_pos)
        
        if lower <= y_true <= upper:
            return float(beta)
            
    return float(b_max)

def calibrate_compass(scores: Union[np.ndarray, list], alpha: float) -> float:
    """
    Calibrate COMPASS nonconformity scores using split conformal prediction.
    Uses the conservative quantile from split_conformal.py.
    """
    from src.conformal.split_conformal import conformal_quantile
    scores_arr = np.array(scores)
    return conformal_quantile(scores_arr, alpha)

def predict_interval_compass(probs: torch.Tensor, beta_hat: float, target_class: int, metric_fn: Callable) -> Tuple[float, float]:
    """
    Constructs the prediction interval for a test sample using calibrated beta_hat.
    """
    m_neg = compute_metric_for_b(probs, -beta_hat, target_class, metric_fn)
    m_pos = compute_metric_for_b(probs, beta_hat, target_class, metric_fn)
    
    return float(min(m_neg, m_pos)), float(max(m_neg, m_pos))
