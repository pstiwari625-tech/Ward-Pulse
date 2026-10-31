"""
Ward Pulse: Backend ML Training Layer
Implements the Ward-Total + Pulse-Shape + Kernel Kriging architecture.
Evaluates strictly via Leave-One-Ward-Out (LOWO) cross-validation on the 18-month target window.
"""

import math
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import config

def solve_ridge_linear_system(X: List[List[float]], y: List[float], alpha: float = 1.0) -> List[float]:
    """
    Solves (X^T X + alpha * I) beta = X^T y using Gaussian elimination with partial pivoting.
    Guaranteed zero-dependency implementation for 14 features on N=156 points.
    """
    n_samples = len(X)
    n_features = len(X[0])

    # Compute X^T X + alpha * I
    XtX = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            s = sum(X[k][i] * X[k][j] for k in range(n_samples))
            XtX[i][j] = s + (alpha if i == j and i > 0 else 0.0) # Do not penalize bias (i=0)

    # Compute X^T y
    Xty = [sum(X[k][i] * y[k] for k in range(n_samples)) for i in range(n_features)]

    # Augmented matrix [A | b]
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]

    # Forward elimination with partial pivoting
    for i in range(n_features):
        max_row = max(range(i, n_features), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]

        pivot = A[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12

        for r in range(i + 1, n_features):
            factor = A[r][i] / pivot
            for c in range(i, n_features + 1):
                A[r][c] -= factor * A[i][c]

    # Back-substitution
    beta = [0.0] * n_features
    for i in range(n_features - 1, -1, -1):
        s = A[i][n_features] - sum(A[i][j] * beta[j] for j in range(i + 1, n_features))
        beta[i] = s / (A[i][i] if abs(A[i][i]) > 1e-12 else 1e-12)

    return beta

def train_and_evaluate_lowo(
    train_wards: List[int],
    feats_dict: Dict[int, Dict[str, Any]],
    counts_dict: Dict[Tuple[int, str, str], int],
    target_months: List[str] = config.TARGET_MONTHS,
    categories: List[str] = config.CATEGORIES,
    kernel_ell: float = 3.5,
    ridge_alpha: float = 5.0
) -> Dict[str, Any]:
    """
    Runs Leave-One-Ward-Out (LOWO) spatial CV on 18-month target window.
    Measures Out-of-Fold (OOF) errors and returns trained model components.
    """
    # Precompute pairwise distances
    dist_mat = {}
    for w1 in train_wards:
        for w2 in train_wards:
            dx = feats_dict[w1]["x"] - feats_dict[w2]["x"]
            dy = feats_dict[w1]["y"] - feats_dict[w2]["y"]
            dist_mat[(w1, w2)] = math.sqrt(dx * dx + dy * dy)

    # Metrics accumulators
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_actual_vol = 0.0
    n_points = 0

    cat_abs_err = defaultdict(float)
    cat_sq_err = defaultdict(float)
    cat_actual = defaultdict(float)
    cat_points = defaultdict(int)

    ward_abs_err = defaultdict(list)
    ward_actual = defaultdict(list)

    # LOWO Loop
    for val_w in train_wards:
        tr_wards = [w for w in train_wards if w != val_w]

        # 1. Temporal pulse shape per category
        pulse_shape = {}
        for c in categories:
            tot_cat = sum(counts_dict.get((w, m, c), 0) for w in tr_wards for m in target_months)
            for m in target_months:
                tot_m_c = sum(counts_dict.get((w, m, c), 0) for w in tr_wards)
                pulse_shape[(m, c)] = tot_m_c / max(tot_cat, 1.0)

        # 2. Kernel weights
        w_weights = [math.exp(- (dist_mat[(val_w, j)] ** 2) / (2.0 * kernel_ell * kernel_ell)) for j in tr_wards]
        sum_w = sum(w_weights) if sum(w_weights) > 0 else 1.0

        for c in categories:
            # Construct training 18-month totals and design matrix
            X_tr = [[1.0] + feats_dict[w]["features"] for w in tr_wards]
            y_tr_totals = [sum(counts_dict.get((w, m, c), 0) for m in target_months) for w in tr_wards]
            log_y_tr = [math.log(y + 1.0) for y in y_tr_totals]

            # Fit Ridge Regression on log totals
            beta = solve_ridge_linear_system(X_tr, log_y_tr, alpha=ridge_alpha)

            # In-fold residuals on training wards
            log_preds_tr = [beta[0] + sum(beta[k + 1] * f for k, f in enumerate(feats_dict[w]["features"])) for w in tr_wards]
            residuals_tr = [y_act - y_p for y_act, y_p in zip(log_y_tr, log_preds_tr)]

            # Static prediction for validation ward
            x_val = feats_dict[val_w]["features"]
            log_pred_val = beta[0] + sum(beta[k + 1] * f for k, f in enumerate(x_val))
            pred_total_static = max(0.5, math.exp(log_pred_val) - 1.0)

            # Spatial residual adjustment via Gaussian kernel
            krig_adj = sum(w * r for w, r in zip(w_weights, residuals_tr)) / sum_w
            pred_total_kriged = max(0.5, pred_total_static * math.exp(krig_adj))

            # Shrinkage for MAE alignment
            pred_total_final = pred_total_kriged * 0.95

            # Allocate across target months using pulse shape
            for m in target_months:
                y_true = counts_dict.get((val_w, m, c), 0)
                y_pred = max(0.0, pred_total_final * pulse_shape[(m, c)])

                err = y_true - y_pred
                abs_e = abs(err)

                total_abs_err += abs_e
                total_sq_err += err * err
                total_actual_vol += y_true
                n_points += 1

                cat_abs_err[c] += abs_e
                cat_sq_err[c] += err * err
                cat_actual[c] += y_true
                cat_points[c] += 1

                ward_abs_err[val_w].append(abs_e)
                ward_actual[val_w].append(y_true)

    # Compute final metrics
    overall_mae = total_abs_err / n_points
    overall_rmse = math.sqrt(total_sq_err / n_points)
    overall_wape = (total_abs_err / total_actual_vol) * 100.0

    cat_metrics = {}
    for c in categories:
        cat_metrics[c] = {
            "mae": round(cat_abs_err[c] / cat_points[c], 3),
            "rmse": round(math.sqrt(cat_sq_err[c] / cat_points[c]), 3),
            "wape_pct": round((cat_abs_err[c] / cat_actual[c]) * 100.0, 2),
            "mean_volume": round(cat_actual[c] / cat_points[c], 2)
        }

    # Ward-level MAE distribution
    ward_mae_list = sorted([sum(errs) / len(errs) for errs in ward_abs_err.values()])
    n_w = len(ward_mae_list)
    ward_dist = {
        "min_mae": round(ward_mae_list[0], 2),
        "q10_mae": round(ward_mae_list[int(n_w * 0.10)], 2),
        "q25_mae": round(ward_mae_list[int(n_w * 0.25)], 2),
        "median_mae": round(ward_mae_list[int(n_w * 0.50)], 2),
        "q75_mae": round(ward_mae_list[int(n_w * 0.75)], 2),
        "q90_mae": round(ward_mae_list[int(n_w * 0.90)], 2),
        "max_mae": round(ward_mae_list[-1], 2),
    }

    return {
        "oof_mae": round(overall_mae, 3),
        "oof_rmse": round(overall_rmse, 3),
        "oof_wape_pct": round(overall_wape, 2),
        "category_metrics": cat_metrics,
        "ward_distribution": ward_dist,
    }
