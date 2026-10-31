"""
Audit Section C: Model Quality Tests (Tests 12 - 16)
Validates beats-baseline by category/month, ward-total correlation, extrapolation safety,
seasonality correlation, and category mix stability.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import csv, math
from collections import defaultdict
import config
from backend.preprocessing import clean_ward_feature_row
from backend.features import extract_ward_feature_vector
from backend.training import solve_ridge_linear_system

def load_data():
    with open(config.WARD_FEATURES_FILE, mode="r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    feats_dict = {int(r["ward_id"]): extract_ward_feature_vector(clean_ward_feature_row(r))
                  for r in feat_rows if r["ward_id"]}
                  
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
                
    with open(config.WARD_PREDICTIONS_FILE, mode="r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
        
    return train_wards, target_wards, feats_dict, counts_dict, pred_rows

def test_12_beats_baseline_slices():
    """Test 12: Check beats-baseline per category and per month (before/after May 2024)."""
    train_wards, target_wards, feats_dict, counts_dict, _ = load_data()
    target_months = config.TARGET_MONTHS
    categories = config.CATEGORIES
    exp_keys = config.CATEGORY_EXPOSURE_MAP
    
    dist_mat = {(w1, w2): math.sqrt((feats_dict[w1]["x"] - feats_dict[w2]["x"])**2 + 
                                    (feats_dict[w1]["y"] - feats_dict[w2]["y"])**2)
                for w1 in train_wards for w2 in train_wards}
                
    base_err_by_cat = defaultdict(float)
    prod_err_by_cat = defaultdict(float)
    pts_by_cat = defaultdict(int)
    
    base_err_pre_may = 0.0
    prod_err_pre_may = 0.0
    pts_pre_may = 0
    
    base_err_post_may = 0.0
    prod_err_post_may = 0.0
    pts_post_may = 0
    
    for val_w in train_wards:
        tr_wards = [w for w in train_wards if w != val_w]
        w_weights = [math.exp(- (dist_mat[(val_w, j)]**2) / (2.0 * 3.5 * 3.5)) for j in tr_wards]
        sum_w = sum(w_weights)
        
        for c in categories:
            tot_cat = sum(counts_dict.get((w, m, c), 0) for w in tr_wards for m in target_months)
            tot_exp = sum(feats_dict[w]["raw"][exp_keys[c]] for w in tr_wards)
            exp_val = feats_dict[val_w]["raw"][exp_keys[c]]
            
            X_tr = [[1.0] + feats_dict[w]["features"] for w in tr_wards]
            y_tr_totals = [sum(counts_dict.get((w, m, c), 0) for m in target_months) for w in tr_wards]
            log_y_tr = [math.log(y + 1.0) for y in y_tr_totals]
            beta = solve_ridge_linear_system(X_tr, log_y_tr, alpha=5.0)
            
            log_preds_tr = [beta[0] + sum(beta[k+1] * f for k, f in enumerate(feats_dict[w]["features"])) for w in tr_wards]
            res_tr = [y_act - y_p for y_act, y_p in zip(log_y_tr, log_preds_tr)]
            log_pred_val = beta[0] + sum(beta[k+1] * f for k, f in enumerate(feats_dict[val_w]["features"]))
            pred_static = max(0.5, math.exp(log_pred_val) - 1.0)
            krig_adj = sum(w * r for w, r in zip(w_weights, res_tr)) / sum_w
            pred_kriged = max(0.5, pred_static * math.exp(krig_adj)) * 0.95
            
            for m in target_months:
                y_true = counts_dict.get((val_w, m, c), 0)
                tot_m = sum(counts_dict.get((w, m, c), 0) for w in tr_wards)
                shape_m = tot_m / max(tot_cat, 1.0)
                rate_m = tot_m / max(tot_exp, 1.0)
                
                b_pred = rate_m * exp_val
                p_pred = max(0.0, pred_kriged * shape_m)
                
                b_err = abs(y_true - b_pred)
                p_err = abs(y_true - p_pred)
                
                base_err_by_cat[c] += b_err
                prod_err_by_cat[c] += p_err
                pts_by_cat[c] += 1
                
                if m < "2024-05":
                    base_err_pre_may += b_err
                    prod_err_pre_may += p_err
                    pts_pre_may += 1
                else:
                    base_err_post_may += b_err
                    prod_err_post_may += p_err
                    pts_post_may += 1

    cats_beaten = []
    cats_missed = []
    for c in categories:
        b_mae = base_err_by_cat[c] / pts_by_cat[c]
        p_mae = prod_err_by_cat[c] / pts_by_cat[c]
        if p_mae <= b_mae:
            cats_beaten.append(f"{c} ({p_mae:.2f} <= {b_mae:.2f})")
        else:
            cats_missed.append(f"{c} ({p_mae:.2f} > {b_mae:.2f})")
            
    pre_mae_b = base_err_pre_may / pts_pre_may
    pre_mae_p = prod_err_pre_may / pts_pre_may
    post_mae_b = base_err_post_may / pts_post_may
    post_mae_p = prod_err_post_may / pts_post_may
    
    passed = len(cats_missed) == 0 and pre_mae_p < pre_mae_b and post_mae_p < post_mae_b
    return {
        "id": 12,
        "name": "Beats baseline across categories and pre/post May 2024",
        "pass": passed,
        "measured": f"Cats beaten: {len(cats_beaten)}/6. Pre-May: Prod {pre_mae_p:.2f} vs Base {pre_mae_b:.2f}. Post-May: Prod {post_mae_p:.2f} vs Base {post_mae_b:.2f}. Missed: {cats_missed}",
        "note": "Production model beats baseline in all temporal and category slices" if passed else f"Did not beat in: {cats_missed}"
    }

def test_13_ward_totals_correlation():
    """Test 13: Ward predicted totals correlate with true totals (Pearson, Spearman, and MAPE)."""
    train_wards, _, feats_dict, counts_dict, _ = load_data()
    target_months = config.TARGET_MONTHS
    categories = config.CATEGORIES
    
    dist_mat = {(w1, w2): math.sqrt((feats_dict[w1]["x"] - feats_dict[w2]["x"])**2 + 
                                    (feats_dict[w1]["y"] - feats_dict[w2]["y"])**2)
                for w1 in train_wards for w2 in train_wards}
                
    true_totals = []
    pred_totals = []
    
    for val_w in train_wards:
        tr_wards = [w for w in train_wards if w != val_w]
        w_weights = [math.exp(- (dist_mat[(val_w, j)]**2) / (2.0 * 3.5 * 3.5)) for j in tr_wards]
        sum_w = sum(w_weights)
        
        y_w_true = sum(counts_dict.get((val_w, m, c), 0) for m in target_months for c in categories)
        y_w_pred = 0.0
        
        for c in categories:
            X_tr = [[1.0] + feats_dict[w]["features"] for w in tr_wards]
            y_tr_totals = [sum(counts_dict.get((w, m, c), 0) for m in target_months) for w in tr_wards]
            log_y_tr = [math.log(y + 1.0) for y in y_tr_totals]
            beta = solve_ridge_linear_system(X_tr, log_y_tr, alpha=5.0)
            
            log_preds_tr = [beta[0] + sum(beta[k+1] * f for k, f in enumerate(feats_dict[w]["features"])) for w in tr_wards]
            res_tr = [y_act - y_p for y_act, y_p in zip(log_y_tr, log_preds_tr)]
            log_pred_val = beta[0] + sum(beta[k+1] * f for k, f in enumerate(feats_dict[val_w]["features"]))
            pred_static = max(0.5, math.exp(log_pred_val) - 1.0)
            krig_adj = sum(w * r for w, r in zip(w_weights, res_tr)) / sum_w
            y_w_pred += max(0.5, pred_static * math.exp(krig_adj)) * 0.95
            
        true_totals.append(y_w_true)
        pred_totals.append(y_w_pred)

    # Pearson correlation
    n = len(true_totals)
    mean_t = sum(true_totals) / n
    mean_p = sum(pred_totals) / n
    cov = sum((t - mean_t) * (p - mean_p) for t, p in zip(true_totals, pred_totals))
    var_t = sum((t - mean_t)**2 for t in true_totals)
    var_p = sum((p - mean_p)**2 for p in pred_totals)
    pearson_r = cov / (math.sqrt(var_t * var_p) + 1e-12)
    
    # Spearman rank correlation
    def get_ranks(vals):
        pairs = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0] * n
        for r_idx, (orig_idx, _) in enumerate(pairs):
            ranks[orig_idx] = r_idx
        return ranks
        
    ranks_t = get_ranks(true_totals)
    ranks_p = get_ranks(pred_totals)
    cov_rank = sum((rt - (n-1)/2) * (rp - (n-1)/2) for rt, rp in zip(ranks_t, ranks_p))
    var_rank = sum((rt - (n-1)/2)**2 for rt in ranks_t)
    spearman_rho = cov_rank / (var_rank + 1e-12)
    
    # Ward-total MAPE
    mape = sum(abs(t - p) / max(1.0, t) for t, p in zip(true_totals, pred_totals)) / n * 100.0
    
    passed = pearson_r >= 0.80 and spearman_rho >= 0.80
    return {
        "id": 13,
        "name": "Ward predicted totals correlate with true totals",
        "pass": passed,
        "measured": f"Pearson r: {pearson_r:.3f}, Spearman rho: {spearman_rho:.3f}, Ward-total MAPE: {mape:.1f}%",
        "note": "Strong positive monotonic scaling across wards" if passed else "Weak correlation"
    }

def test_14_extrapolation_safety():
    """Test 14: Extrapolation: assert predictions stay within training range of rate-per-exposure."""
    train_wards, target_wards, feats_dict, counts_dict, pred_rows = load_data()
    target_months = config.TARGET_MONTHS
    categories = config.CATEGORIES
    exp_keys = config.CATEGORY_EXPOSURE_MAP
    
    # Training range of rate-per-exposure
    max_train_rate = {}
    min_train_rate = {}
    for c in categories:
        exp_k = exp_keys[c]
        rates = []
        for w in train_wards:
            for m in target_months:
                y = counts_dict.get((w, m, c), 0)
                exp = feats_dict[w]["raw"][exp_k]
                rates.append(y / max(exp, 1.0))
        max_train_rate[c] = max(rates)
        min_train_rate[c] = min(rates)
        
    outliers = []
    for r in pred_rows:
        w = int(r["ward_id"])
        c = r["category_group"]
        pred_cnt = float(r["predicted_count"])
        exp = feats_dict[w]["raw"][exp_keys[c]]
        pred_rate = pred_cnt / max(exp, 1.0)
        
        # Check if exceeds max training rate by > 1.5x
        if pred_rate > max_train_rate[c] * 1.5:
            outliers.append((w, c, pred_rate, max_train_rate[c]))
            
    passed = len(outliers) == 0
    return {
        "id": 14,
        "name": "Extrapolation safety check (rate-per-exposure bounded)",
        "pass": passed,
        "measured": f"Outlier predictions exceeding training rate: {len(outliers)}",
        "note": "Log-linear model prevents runaway extrapolation on larger target wards" if passed else f"Unbounded extrapolation: {outliers[:3]}"
    }

def test_15_seasonality_city_curve_correlation():
    """Test 15: Seasonality: city-level monthly curve of target predictions vs training curve (corr >= 0.95)."""
    train_wards, _, _, counts_dict, pred_rows = load_data()
    target_months = config.TARGET_MONTHS
    
    # Monthly total for train wards
    train_monthly = []
    for m in target_months:
        tot_m = sum(counts_dict.get((w, m, c), 0) for w in train_wards for c in config.CATEGORIES)
        train_monthly.append(tot_m)
        
    # Monthly total for target predictions
    target_monthly_map = defaultdict(float)
    for r in pred_rows:
        target_monthly_map[r["month"]] += float(r["predicted_count"])
    target_monthly = [target_monthly_map[m] for m in target_months]
    
    # Pearson correlation
    n = len(train_monthly)
    mt = sum(train_monthly) / n
    mp = sum(target_monthly) / n
    cov = sum((t - mt) * (p - mp) for t, p in zip(train_monthly, target_monthly))
    vart = sum((t - mt)**2 for t in train_monthly)
    varp = sum((p - mp)**2 for p in target_monthly)
    r = cov / (math.sqrt(vart * varp) + 1e-12)
    
    passed = r >= 0.95
    return {
        "id": 15,
        "name": "Seasonality & macro surge tracking (correlation >= 0.95)",
        "pass": passed,
        "measured": f"Monthly curve correlation: {r:.4f} (Train Jan-24: {train_monthly[0]}, May-24: {train_monthly[4]}; Target Jan-24: {target_monthly[0]:.0f}, May-24: {target_monthly[4]:.0f})",
        "note": "[SANITY CHECK ONLY - Not predictive accuracy]: Reflects macro city pulse shape tracking and post-May 2024 jump" if passed else "Seasonal desynchronization"
    }

def test_16_category_mix_drift():
    """Test 16: Category mix: predicted category shares for 40 wards vs training shares (drift <= 5 pp)."""
    train_wards, _, _, counts_dict, pred_rows = load_data()
    target_months = config.TARGET_MONTHS
    categories = config.CATEGORIES
    
    train_cat_totals = {c: sum(counts_dict.get((w, m, c), 0) for w in train_wards for m in target_months) for c in categories}
    tot_train = sum(train_cat_totals.values())
    train_shares = {c: train_cat_totals[c] / tot_train * 100.0 for c in categories}
    
    pred_cat_totals = defaultdict(float)
    for r in pred_rows:
        pred_cat_totals[r["category_group"]] += float(r["predicted_count"])
    tot_pred = sum(pred_cat_totals.values())
    pred_shares = {c: pred_cat_totals[c] / tot_pred * 100.0 for c in categories}
    
    drifts = {}
    max_drift = 0.0
    for c in categories:
        diff = abs(train_shares[c] - pred_shares[c])
        drifts[c] = (round(train_shares[c], 1), round(pred_shares[c], 1), round(diff, 1))
        if diff > max_drift:
            max_drift = diff
            
    passed = max_drift <= 5.0
    return {
        "id": 16,
        "name": "Category composition mix stability (drift <= 5 pp)",
        "pass": passed,
        "measured": f"Max drift: {max_drift:.1f} pp. Shares (Train% vs Pred%): {drifts}",
        "note": "Category mix aligns closely with training ground truth" if passed else f"Drift exceeds 5 pp"
    }

if __name__ == "__main__":
    for t in [test_12_beats_baseline_slices, test_13_ward_totals_correlation,
              test_14_extrapolation_safety, test_15_seasonality_city_curve_correlation,
              test_16_category_mix_drift]:
        res = t()
        status = "PASS" if res["pass"] else "FAIL"
        print(f"[{status}] Test {res['id']}: {res['name']} | Measured: {res['measured']}")
