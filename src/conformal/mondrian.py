import numpy as np
from typing import Tuple
from src.conformal.split_conformal import calibrate, predict_interval
from src.conformal.crc import find_lambda, predict_interval_crc

def calibrate_mondrian(cal_true: np.ndarray,
                       cal_pred: np.ndarray,
                       cal_groups: np.ndarray,
                       alpha: float,
                       mode: str = 'abs') -> dict:
    """
    Calibrate Mondrian Split Conformal Prediction.
    
    Args:
        cal_true   : true metrics, shape (n,)
        cal_pred   : predicted metrics, shape (n,)
        cal_groups : group labels, shape (n,)
        alpha      : miscoverage level
        mode       : 'abs' or 'rel'
        
    Returns:
        dict containing group-specific quantiles and models.
    """
    unique_groups = np.unique(cal_groups)
    group_models = {}
    
    for g in unique_groups:
        idx = (cal_groups == g)
        group_models[g] = calibrate(cal_true[idx], cal_pred[idx], alpha, mode=mode)
        
    return {
        'group_models': group_models,
        'alpha': alpha,
        'mode': mode
    }

def predict_interval_mondrian(test_pred: np.ndarray,
                             test_groups: np.ndarray,
                             mondrian_result: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions intervals using Mondrian Split Conformal.
    """
    lower = np.zeros_like(test_pred)
    upper = np.zeros_like(test_pred)
    group_models = mondrian_result['group_models']
    mode = mondrian_result['mode']
    
    # Pre-calculate fallback q_hat (mean of all group-specific q_hats)
    fallback_q = np.mean([m['q_hat'] for m in group_models.values()])
    
    for i, (p, g) in enumerate(zip(test_pred, test_groups)):
        q_hat = group_models[g]['q_hat'] if g in group_models else fallback_q
        lo, up = predict_interval(np.array([p]), q_hat, mode=mode)
        lower[i] = lo[0]
        upper[i] = up[0]
        
    return lower, upper

def calibrate_mondrian_crc(cal_true: np.ndarray,
                           cal_pred: np.ndarray,
                           cal_groups: np.ndarray,
                           alpha: float,
                           mode: str = 'abs') -> dict:
    """
    Calibrate Mondrian Conformal Risk Control.
    """
    unique_groups = np.unique(cal_groups)
    group_models = {}
    
    for g in unique_groups:
        idx = (cal_groups == g)
        group_models[g] = find_lambda(cal_true[idx], cal_pred[idx], alpha, mode=mode)
        
    return {
        'group_models': group_models,
        'alpha': alpha,
        'mode': mode
    }

def predict_interval_mondrian_crc(test_pred: np.ndarray,
                                 test_groups: np.ndarray,
                                 mondrian_result: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions intervals using Mondrian Conformal Risk Control.
    """
    lower = np.zeros_like(test_pred)
    upper = np.zeros_like(test_pred)
    group_models = mondrian_result['group_models']
    mode = mondrian_result['mode']
    
    # Pre-calculate fallback lambda (mean of all group-specific lambdas)
    fallback_lam = np.mean([m['lambda'] for m in group_models.values()])
    
    for i, (p, g) in enumerate(zip(test_pred, test_groups)):
        lam = group_models[g]['lambda'] if g in group_models else fallback_lam
        lo, up = predict_interval_crc(np.array([p]), lam, mode=mode)
        lower[i] = lo[0]
        upper[i] = up[0]
        
    return lower, upper
