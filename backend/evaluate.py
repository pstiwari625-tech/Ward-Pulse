"""
Ward Pulse: Unified Cross-Validation & Metric Evaluation Engine (backend/evaluate.py)
Reconciles all models and benchmarks in ONE script with fixed seeds on the EXACT SAME
LOWO folds (156 folds), the EXACT SAME 18-month evaluation window (2024-01 to 2025-06),
and the EXACT SAME target (complaint counts per ward-month-category, N = 16,848 points).

Evaluates:
 1. Baseline (ratio-of-sums x exposure)
 2. Ridge only (no kriging)
 3. Ridge + Kriging with bandwidth sweep {0.8, 1.5, 2.5, 3.5, 5.0, 7.0}
 4. Production model (bandwidth 3.5 km, ridge alpha 5.0)
 5. Nested CV (bandwidth chosen strictly inside each fold)
 6. Ward-bootstrap 95% CI (1,000 resamples)
 7. Priority sensitivity (1,000 draws of ±20% weights)
"""

import sys
import math
import json
import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from backend.preprocessing import clean_ward_feature_row
from backend.features import extract_ward_feature_vector
from backend.training import solve_ridge_linear_system
from backend.prioritization import (
    percentile_rank, classify_tier, compute_kendall_tau, generate_ward_summary_table
)

def run_evaluation() -> Dict[str, Any]:
    random.seed(config.RANDOM_SEED)

    # 1. Ingest Data
    with open(config.WARD_FEATURES_FILE, mode="r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    feats_dict = {
        int(r["ward_id"]): extract_ward_feature_vector(clean_ward_feature_row(r))
        for r in feat_rows if r["ward_id"]
    }

    with open(config.PREDICTION_GRID_FILE, mode="r", encoding="utf-8") as f:
        target_wards = sorted({int(r["ward_id"]) for r in csv.DictReader(f)})
    train_wards = sorted(list(set(feats_dict.keys()) - set(target_wards) - {3, 65}))

    counts_dict = defaultdict(int)
    target_months_set = set(config.TARGET_MONTHS)
    with open(config.GRIEVANCES_FILE, mode="r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r["grievance_time"][:7]
            if m in target_months_set:
                w = int(r["ward_id"])
                c = r["category_group"]
                counts_dict[(w, m, c)] += 1

    # Precompute pairwise planar distances (km)
    dist_mat = {}
    for w1 in train_wards:
        for w2 in train_wards:
            dx = feats_dict[w1]["x"] - feats_dict[w2]["x"]
            dy = feats_dict[w1]["y"] - feats_dict[w2]["y"]
            dist_mat[(w1, w2)] = math.sqrt(dx * dx + dy * dy)

    # Candidate bandwidths for sweep
    sweep_bandwidths = [0.8, 1.5, 2.5, 3.5, 5.0, 7.0]
    nested_candidates = [1.5, 2.5, 3.5, 5.0]

    # Error tracking structures
    # Models: baseline, ridge_only, prod (3.5), nested, and sweep {ell: ...}
    err_accum = {
        "baseline": {"abs": 0.0, "sq": 0.0, "cat_abs": defaultdict(float), "cat_pts": defaultdict(int), "pre_abs": 0.0, "pre_pts": 0, "post_abs": 0.0, "post_pts": 0},
        "ridge_only": {"abs": 0.0, "sq": 0.0, "cat_abs": defaultdict(float), "cat_pts": defaultdict(int), "pre_abs": 0.0, "pre_pts": 0, "post_abs": 0.0, "post_pts": 0},
        "production": {"abs": 0.0, "sq": 0.0, "cat_abs": defaultdict(float), "cat_pts": defaultdict(int), "pre_abs": 0.0, "pre_pts": 0, "post_abs": 0.0, "post_pts": 0},
        "nested_cv": {"abs": 0.0, "sq": 0.0, "cat_abs": defaultdict(float), "cat_pts": defaultdict(int), "pre_abs": 0.0, "pre_pts": 0, "post_abs": 0.0, "post_pts": 0}
    }
    for ell in sweep_bandwidths:
        err_accum[f"kriging_{ell}"] = {"abs": 0.0, "sq": 0.0}

    ward_err_base = defaultdict(list)
    ward_err_prod = defaultdict(list)
    total_volume = 0.0
    total_points = 0
    nested_chosen_bandwidths = []

    # Residuals storage for Step 3 intervals
    oof_residuals = [] # list of (val_w, m, c, y_true, y_pred, abs_err, std_res)

    print(f"Executing unified LOWO evaluation across {len(train_wards)} folds...")

    for fold_idx, val_w in enumerate(train_wards):
        tr_wards = [w for w in train_wards if w != val_w]

        # 1. Pulse shape and City rates on tr_wards
        pulse_shape = {}
        city_rates = {}
        for c in config.CATEGORIES:
            exp_k = config.CATEGORY_EXPOSURE_MAP[c]
            tot_cat = sum(counts_dict.get((w, m, c), 0) for w in tr_wards for m in config.TARGET_MONTHS)
            tot_exp = sum(feats_dict[w]["raw"][exp_k] for w in tr_wards)
            for m in config.TARGET_MONTHS:
                tot_m = sum(counts_dict.get((w, m, c), 0) for w in tr_wards)
                pulse_shape[(m, c)] = tot_m / max(tot_cat, 1.0)
                city_rates[(m, c)] = tot_m / max(tot_exp, 1.0)

        # 2. Kernel weights for validation ward across candidate bandwidths
        kernel_weights = {}
        sum_weights = {}
        for ell in sweep_bandwidths:
            w_wts = [math.exp(- (dist_mat[(val_w, j)] ** 2) / (2.0 * ell * ell)) for j in tr_wards]
            kernel_weights[ell] = w_wts
            sum_weights[ell] = sum(w_wts) if sum(w_wts) > 0 else 1.0

        # 3. Inner CV for Nested CV tuning inside tr_wards only
        # Use 5 pseudo-folds within tr_wards to select optimal bandwidth
        inner_pseudo_folds = [tr_wards[i::5] for i in range(5)]
        inner_bandwidth_errs = defaultdict(float)
        inner_pts = 0

        for pf_idx, pf_val in enumerate(inner_pseudo_folds):
            pf_tr = [w for w in tr_wards if w not in set(pf_val)]
            for c in ["electrical", "solid_waste", "roads"]: # Representative high-volume categories
                X_ptr = [[1.0] + feats_dict[w]["features"] for w in pf_tr]
                y_ptr_totals = [sum(counts_dict.get((w, m, c), 0) for m in config.TARGET_MONTHS) for w in pf_tr]
                log_y_ptr = [math.log(y + 1.0) for y in y_ptr_totals]
                b_ptr = solve_ridge_linear_system(X_ptr, log_y_ptr, alpha=5.0)

                log_preds_ptr = [b_ptr[0] + sum(b_ptr[k+1] * f for k, f in enumerate(feats_dict[w]["features"])) for w in pf_tr]
                res_ptr = [y_act - y_p for y_act, y_p in zip(log_y_ptr, log_preds_ptr)]

                for pw in pf_val:
                    y_pw_act = sum(counts_dict.get((pw, m, c), 0) for m in config.TARGET_MONTHS)
                    log_pred_pw = b_ptr[0] + sum(b_ptr[k+1] * f for k, f in enumerate(feats_dict[pw]["features"]))
                    static_pw = max(0.5, math.exp(log_pred_pw) - 1.0)

                    for cand_ell in nested_candidates:
                        dists = [dist_mat[(pw, j)] for j in pf_tr]
                        wts = [math.exp(- (d**2) / (2.0 * cand_ell * cand_ell)) for d in dists]
                        sw = sum(wts) if sum(wts) > 0 else 1.0
                        krig = sum(w * r for w, r in zip(wts, res_ptr)) / sw
                        pred_k = max(0.5, static_pw * math.exp(krig)) * 0.95
                        inner_bandwidth_errs[cand_ell] += abs(y_pw_act - pred_k)
                        inner_pts += 1

        best_inner_ell = min(nested_candidates, key=lambda l: inner_bandwidth_errs[l])
        nested_chosen_bandwidths.append(best_inner_ell)

        # 4. Fit Ridge models for all 6 categories on tr_wards
        for c in config.CATEGORIES:
            exp_k = config.CATEGORY_EXPOSURE_MAP[c]
            val_exp = feats_dict[val_w]["raw"][exp_k]

            X_tr = [[1.0] + feats_dict[w]["features"] for w in tr_wards]
            y_tr_totals = [sum(counts_dict.get((w, m, c), 0) for m in config.TARGET_MONTHS) for w in tr_wards]
            log_y_tr = [math.log(y + 1.0) for y in y_tr_totals]
            beta = solve_ridge_linear_system(X_tr, log_y_tr, alpha=5.0)

            # In-fold training residuals
            log_preds_tr = [beta[0] + sum(beta[k+1] * f for k, f in enumerate(feats_dict[w]["features"])) for w in tr_wards]
            residuals_tr = [y_act - y_p for y_act, y_p in zip(log_y_tr, log_preds_tr)]

            # Static prediction
            log_pred_val = beta[0] + sum(beta[k+1] * f for k, f in enumerate(feats_dict[val_w]["features"]))
            pred_static = max(0.5, math.exp(log_pred_val) - 1.0)

            # Kriged predictions across bandwidths
            pred_kriged = {}
            for ell in sweep_bandwidths:
                krig_adj = sum(w * r for w, r in zip(kernel_weights[ell], residuals_tr)) / sum_weights[ell]
                pred_kriged[ell] = max(0.5, pred_static * math.exp(krig_adj)) * 0.95

            # Nested tuned kriged prediction
            nested_krig_adj = sum(w * r for w, r in zip(kernel_weights[best_inner_ell], residuals_tr)) / sum_weights[best_inner_ell]
            pred_nested = max(0.5, pred_static * math.exp(nested_krig_adj)) * 0.95

            pred_ridge_only = pred_static * 0.95
            pred_prod = pred_kriged[3.5] # Production bandwidth is 3.5 km

            # Allocate across target months using pulse shape
            for m in config.TARGET_MONTHS:
                y_true = counts_dict.get((val_w, m, c), 0)
                shape_m = pulse_shape[(m, c)]
                rate_m = city_rates[(m, c)]

                y_base = rate_m * val_exp
                y_ridge = max(0.0, pred_ridge_only * shape_m)
                y_prod = max(0.0, pred_prod * shape_m)
                y_nested = max(0.0, pred_nested * shape_m)

                # Track residuals for intervals
                abs_e_prod = abs(y_true - y_prod)
                std_res = abs_e_prod / math.sqrt(y_prod + 1.0)
                oof_residuals.append({
                    "ward_id": val_w,
                    "month": m,
                    "category_group": c,
                    "y_true": y_true,
                    "y_pred": y_prod,
                    "abs_error": abs_e_prod,
                    "std_residual": std_res
                })

                ward_err_base[val_w].append(abs(y_true - y_base))
                ward_err_prod[val_w].append(abs_e_prod)

                # Accumulate for slice analysis
                total_volume += y_true
                total_points += 1

                for model_key, y_p in [("baseline", y_base), ("ridge_only", y_ridge),
                                       ("production", y_prod), ("nested_cv", y_nested)]:
                    e = abs(y_true - y_p)
                    se = (y_true - y_p) ** 2
                    err_accum[model_key]["abs"] += e
                    err_accum[model_key]["sq"] += se
                    err_accum[model_key]["cat_abs"][c] += e
                    err_accum[model_key]["cat_pts"][c] += 1
                    if m < "2024-05":
                        err_accum[model_key]["pre_abs"] += e
                        err_accum[model_key]["pre_pts"] += 1
                    else:
                        err_accum[model_key]["post_abs"] += e
                        err_accum[model_key]["post_pts"] += 1

                # Accumulate sweep bandwidths
                for ell in sweep_bandwidths:
                    y_krig = max(0.0, pred_kriged[ell] * shape_m)
                    e_k = abs(y_true - y_krig)
                    err_accum[f"kriging_{ell}"]["abs"] += e_k
                    err_accum[f"kriging_{ell}"]["sq"] += (y_true - y_krig) ** 2

    # 5. Compile Global Metrics
    metrics_summary = {}
    for m_key in ["baseline", "ridge_only", "production", "nested_cv"]:
        mae = err_accum[m_key]["abs"] / total_points
        rmse = math.sqrt(err_accum[m_key]["sq"] / total_points)
        wape = (err_accum[m_key]["abs"] / total_volume) * 100.0
        pre_mae = err_accum[m_key]["pre_abs"] / err_accum[m_key]["pre_pts"]
        post_mae = err_accum[m_key]["post_abs"] / err_accum[m_key]["post_pts"]
        cat_maes = {c: round(err_accum[m_key]["cat_abs"][c] / err_accum[m_key]["cat_pts"][c], 3)
                    for c in config.CATEGORIES}

        metrics_summary[m_key] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "wape_pct": round(wape, 2),
            "pre_may_2024_mae": round(pre_mae, 3),
            "post_may_2024_mae": round(post_mae, 3),
            "category_maes": cat_maes
        }

    # Bandwidth sweep compilation
    sweep_results = {}
    for ell in sweep_bandwidths:
        mae_ell = err_accum[f"kriging_{ell}"]["abs"] / total_points
        rmse_ell = math.sqrt(err_accum[f"kriging_{ell}"]["sq"] / total_points)
        sweep_results[str(ell)] = {
            "bandwidth_km": ell,
            "mae": round(mae_ell, 3),
            "rmse": round(rmse_ell, 3)
        }

    # 6. Ward-Bootstrap 95% CI (1,000 resamples of 156 wards)
    deltas = []
    w_list = list(train_wards)
    n_w = len(w_list)

    for _ in range(1000):
        sample = [random.choice(w_list) for _ in range(n_w)]
        m_base = sum(sum(ward_err_base[w]) / len(ward_err_base[w]) for w in sample) / n_w
        m_prod = sum(sum(ward_err_prod[w]) / len(ward_err_prod[w]) for w in sample) / n_w
        deltas.append(m_prod - m_base)

    deltas.sort()
    ci_lower = round(deltas[25], 3)
    ci_upper = round(deltas[975], 3)
    mean_delta = round(sum(deltas) / len(deltas), 3)

    bootstrap_summary = {
        "resamples": 1000,
        "mean_delta_mae": mean_delta,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "strictly_excludes_zero": bool(ci_upper < 0.0)
    }

    # 7. Priority Sensitivity Analysis (1,000 draws)
    with open(config.WARD_PREDICTIONS_FILE, mode="r", encoding="utf-8") as f:
        target_predictions = list(csv.DictReader(f))
    for p in target_predictions:
        p["ward_id"] = int(p["ward_id"])
        p["predicted_count"] = float(p["predicted_count"])

    # Aggregate target predictions
    ward_cat_totals = {w: {c: 0.0 for c in config.CATEGORIES} for w in target_wards}
    for p in target_predictions:
        w = p["ward_id"]
        c = p["category_group"]
        ward_cat_totals[w][c] += p["predicted_count"] / 18.0

    ward_monthly_totals = [sum(ward_cat_totals[w].values()) for w in target_wards]
    ward_densities = [feats_dict[w]["raw"]["population_density_2011_persons_sq_km"] for w in target_wards]
    ward_growths = [feats_dict[w]["raw"]["decadal_popn_growth_rate_2011_2001"] for w in target_wards]
    ward_infras = [feats_dict[w]["raw"]["streetlights"] / max(1.0, feats_dict[w]["raw"]["road_length_kms"]) for w in target_wards]

    rank_demand = percentile_rank(ward_monthly_totals)
    rank_density = percentile_rank(ward_densities)
    rank_growth = percentile_rank(ward_growths)
    rank_infra = percentile_rank(ward_infras)

    base_cpis = []
    for i in range(len(target_wards)):
        s = (
            config.CPI_WEIGHT_PREDICTED_DEMAND * rank_demand[i] +
            config.CPI_WEIGHT_POPULATION_DENSITY * rank_density[i] +
            config.CPI_WEIGHT_GROWTH_RATE * rank_growth[i] +
            config.CPI_WEIGHT_INFRA_DEFICIT * (1.0 - rank_infra[i])
        )
        base_cpis.append(round(s, 3))

    taus = []
    flips_per_ward = {w: 0 for w in target_wards}

    for _ in range(1000):
        w_dem = random.uniform(0.32, 0.48)
        w_den = random.uniform(0.24, 0.36)
        w_gro = random.uniform(0.12, 0.18)
        w_inf = random.uniform(0.12, 0.18)
        w_sum = w_dem + w_den + w_gro + w_inf

        sim_scores = [
            (w_dem * rank_demand[i] + w_den * rank_density[i] + w_gro * rank_growth[i] + w_inf * (1.0 - rank_infra[i])) / w_sum
            for i in range(len(target_wards))
        ]
        taus.append(compute_kendall_tau(base_cpis, sim_scores))
        for i, (b, s) in enumerate(zip(base_cpis, sim_scores)):
            if classify_tier(b) != classify_tier(s):
                flips_per_ward[target_wards[i]] += 1

    mean_tau = round(sum(taus) / len(taus), 3)
    flipping_wards = [w for w, cnt in flips_per_ward.items() if cnt > 0]
    stable_wards = [w for w, cnt in flips_per_ward.items() if cnt == 0]

    sensitivity_payload = {
        "draws": 1000,
        "perturbation_range_pct": 20,
        "mean_kendall_tau": mean_tau,
        "total_wards": len(target_wards),
        "stable_wards_count": len(stable_wards),
        "borderline_wards_count": len(flipping_wards),
        "borderline_ward_ids": flipping_wards,
        "flips_per_ward": flips_per_ward
    }

    # 8. Assemble Full Metrics Payload
    full_payload = {
        "evaluation_name": "Unified_LOWO_CrossValidation_Reconciliation",
        "evaluation_window": f"{config.TARGET_MONTHS[0]} to {config.TARGET_MONTHS[-1]} (18 months)",
        "total_evaluated_folds": len(train_wards),
        "total_evaluated_cells": total_points,
        "models": metrics_summary,
        "bandwidth_sweep": sweep_results,
        "nested_cv_tuning": {
            "tested_candidates": nested_candidates,
            "chosen_bandwidth_frequencies": {
                str(l): nested_chosen_bandwidths.count(l) for l in nested_candidates
            },
            "nested_mae": metrics_summary["nested_cv"]["mae"],
            "global_prod_mae": metrics_summary["production"]["mae"],
            "score_difference": round(abs(metrics_summary["nested_cv"]["mae"] - metrics_summary["production"]["mae"]), 3)
        },
        "ward_bootstrap_95ci": bootstrap_summary
    }

    # Write metrics to data/metrics/metrics_latest.json
    metrics_path = config.DATA_DIR / "metrics" / "metrics_latest.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    # Write sensitivity to data/metrics/priority_sensitivity.json
    sens_path = config.DATA_DIR / "metrics" / "priority_sensitivity.json"
    with open(sens_path, mode="w", encoding="utf-8") as f:
        json.dump(sensitivity_payload, f, indent=2)

    # Also save OOF residuals for Step 3 intervals
    oof_res_path = config.DATA_DIR / "metrics" / "lowo_oof_residuals.json"
    with open(oof_res_path, mode="w", encoding="utf-8") as f:
        json.dump(oof_residuals, f)

    print("\n[SUCCESS] Unified evaluation complete!")
    print(f"Baseline MAE:   {metrics_summary['baseline']['mae']:.3f} (RMSE: {metrics_summary['baseline']['rmse']:.2f})")
    print(f"Ridge-only MAE: {metrics_summary['ridge_only']['mae']:.3f}")
    print(f"Production MAE: {metrics_summary['production']['mae']:.3f} (RMSE: {metrics_summary['production']['rmse']:.2f}, WAPE: {metrics_summary['production']['wape_pct']:.1f}%)")
    print(f"Nested CV MAE:  {metrics_summary['nested_cv']['mae']:.3f}")
    print(f"Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], Mean delta: {mean_delta:.3f}")
    print(f"Mean Kendall Tau: {mean_tau:.3f}, Flips: {len(flipping_wards)}/40 wards")
    print(f"Exported: {metrics_path} & {sens_path}")

    return full_payload

if __name__ == "__main__":
    run_evaluation()
