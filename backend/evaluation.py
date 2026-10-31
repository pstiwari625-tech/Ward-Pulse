"""
Ward Pulse: Backend Evaluation Layer
Calculates Spatial Out-of-Fold (OOF) metrics: MAE, RMSE, R2, and Poisson Deviance.
"""

import math
from typing import List, Dict, Any

def calculate_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Computes MAE, RMSE, and R2 on count predictions."""
    n = len(y_true)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)

    mean_y = sum(y_true) / n
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "oof_mae": round(mae, 4),
        "oof_rmse": round(rmse, 4),
        "oof_r2": round(r2, 4),
    }
