"""
Ward Pulse: Backend Inference Layer
Generates production forecasts for the 40 unobserved target wards over 18 months (4,320 rows).
Uses the trained Ward-Total + Pulse-Shape + Kernel Kriging architecture.
"""

import math
from typing import Dict, List, Any, Tuple
import csv
import config
from backend.features import extract_ward_feature_vector
from backend.training import solve_ridge_linear_system

def generate_production_predictions(
    train_wards: List[int],
    target_wards: List[int],
    feats_dict: Dict[int, Dict[str, Any]],
    counts_dict: Dict[Tuple[int, str, str], int],
    target_months: List[str] = config.TARGET_MONTHS,
    categories: List[str] = config.CATEGORIES,
    kernel_ell: float = 3.5,
    ridge_alpha: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Fits on all 156 training wards and predicts the 4,320 grid cells for target wards.
    """
    # 1. Compute temporal pulse shape across all training wards
    pulse_shape = {}
    for c in categories:
        tot_cat = sum(counts_dict.get((w, m, c), 0) for w in train_wards for m in target_months)
        for m in target_months:
            tot_m_c = sum(counts_dict.get((w, m, c), 0) for w in train_wards)
            pulse_shape[(m, c)] = tot_m_c / max(tot_cat, 1.0)

    # 2. Fit Ridge regression on 18-month totals for each category
    models_beta = {}
    residuals_train = {}

    for c in categories:
        X_tr = [[1.0] + feats_dict[w]["features"] for w in train_wards]
        y_tr_totals = [sum(counts_dict.get((w, m, c), 0) for m in target_months) for w in train_wards]
        log_y_tr = [math.log(y + 1.0) for y in y_tr_totals]

        beta = solve_ridge_linear_system(X_tr, log_y_tr, alpha=ridge_alpha)
        models_beta[c] = beta

        # Residuals
        log_preds_tr = [beta[0] + sum(beta[k + 1] * f for k, f in enumerate(feats_dict[w]["features"])) for w in train_wards]
        residuals_train[c] = [y_act - y_p for y_act, y_p in zip(log_y_tr, log_preds_tr)]

    # 3. Predict for the 40 target wards
    predictions = []

    for tw in target_wards:
        x_tw, y_tw = feats_dict[tw]["x"], feats_dict[tw]["y"]
        feats_tw = feats_dict[tw]["features"]

        # Compute Gaussian distance weights to all 156 training wards
        dists = [math.sqrt((x_tw - feats_dict[w]["x"])**2 + (y_tw - feats_dict[w]["y"])**2) for w in train_wards]
        weights = [math.exp(- (d**2) / (2.0 * kernel_ell * kernel_ell)) for d in dists]
        sum_w = sum(weights) if sum(weights) > 0 else 1.0

        for c in categories:
            beta = models_beta[c]
            log_pred_static = beta[0] + sum(beta[k + 1] * f for k, f in enumerate(feats_tw))
            pred_total_static = max(0.5, math.exp(log_pred_static) - 1.0)

            # Spatial residual kriging
            krig_adj = sum(w * r for w, r in zip(weights, residuals_train[c])) / sum_w
            pred_total_kriged = max(0.5, pred_total_static * math.exp(krig_adj))

            # Apply MAE shrinkage
            pred_total_final = pred_total_kriged * 0.95

            for m in target_months:
                pred_count = max(0.0, pred_total_final * pulse_shape[(m, c)])
                predictions.append({
                    "ward_id": tw,
                    "month": m,
                    "category_group": c,
                    "predicted_count": round(pred_count, 2)
                })

    return predictions

def export_predictions_csv(predictions: List[Dict[str, Any]], output_path = config.WARD_PREDICTIONS_FILE):
    """Writes primary contract CSV."""
    fieldnames = ["ward_id", "month", "category_group", "predicted_count"]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
