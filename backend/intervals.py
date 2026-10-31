"""
Ward Pulse: Conformal Prediction Intervals Engine (backend/intervals.py)
Calibrates scale-aware Mondrian conformal prediction intervals using Leave-One-Ward-Out
(LOWO) out-of-fold residuals on the 18-month evaluation window.

Features:
 1. Scale-aware standardization: |y - yhat| / sqrt(yhat + 1)
 2. Mondrian volume tertiles per category (Low, Med, High) for uniform coverage
 3. Ward-total intervals calibrated separately across the 156 training wards
 4. Strict bounding: lower >= 0, lower <= pred <= upper
 5. Exports:
    - data/predictions/ward_predictions_with_intervals.csv (4,320 rows)
    - data/predictions/ward_summary_intervals.csv (40 wards)
 6. Empirical coverage reporting across all slices (target 85-95%)
"""

import sys
import math
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def compute_percentile(values: List[float], p: float) -> float:
    """Computes empirical percentile p (0.0 to 1.0) with linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1

def calibrate_and_generate_intervals() -> Dict[str, Any]:
    oof_res_path = config.DATA_DIR / "metrics" / "lowo_oof_residuals.json"
    if not oof_res_path.exists():
        from backend.evaluate import run_evaluation
        run_evaluation()

    with open(oof_res_path, mode="r", encoding="utf-8") as f:
        oof_residuals = json.load(f)

    # 1. Calibrate Mondrian Volume Bins per Category
    # Determine volume tertile cutoffs per category
    cat_preds = defaultdict(list)
    for r in oof_residuals:
        cat_preds[r["category_group"]].append(r["y_pred"])

    volume_cutoffs = {}
    for c in config.CATEGORIES:
        preds = cat_preds[c]
        q33 = compute_percentile(preds, 0.333)
        q67 = compute_percentile(preds, 0.667)
        volume_cutoffs[c] = (q33, q67)

    def get_bin(c: str, pred: float) -> str:
        q33, q67 = volume_cutoffs[c]
        if pred <= q33:
            return "low"
        elif pred <= q67:
            return "med"
        return "high"

    # Group standardized residuals by (category, bin)
    mondrian_residuals = defaultdict(list)
    cat_residuals = defaultdict(list)
    bin_residuals = defaultdict(list)
    all_residuals = []

    # Also aggregate per-ward totals
    ward_actual_tot = defaultdict(float)
    ward_pred_tot = defaultdict(float)

    for r in oof_residuals:
        c = r["category_group"]
        b = get_bin(c, r["y_pred"])
        s = r["std_residual"]

        mondrian_residuals[(c, b)].append(s)
        cat_residuals[c].append(s)
        bin_residuals[b].append(s)
        all_residuals.append(s)

        ward_actual_tot[r["ward_id"]] += r["y_true"]
        ward_pred_tot[r["ward_id"]] += r["y_pred"]

    # Nominal 90% conformal quantiles per (category, bin)
    mondrian_q90 = {}
    for c in config.CATEGORIES:
        for b in ["low", "med", "high"]:
            q = compute_percentile(mondrian_residuals[(c, b)], 0.90)
            mondrian_q90[(c, b)] = max(0.5, q)

    # 2. Ward-Total Conformal Calibration (156 wards)
    ward_std_res = []
    for wid in ward_actual_tot:
        y_w = ward_actual_tot[wid]
        p_w = ward_pred_tot[wid]
        ward_std_res.append(abs(y_w - p_w) / math.sqrt(p_w + 1.0))

    q90_ward_total = compute_percentile(ward_std_res, 0.90)

    # 3. Measure Empirical Coverage under LOWO (16,848 cells)
    covered_all = 0
    covered_by_cat = defaultdict(int)
    covered_by_bin = defaultdict(int)
    total_by_cat = defaultdict(int)
    total_by_bin = defaultdict(int)
    total_pts = len(oof_residuals)

    width_sum = 0.0
    pred_sum = 0.0

    for r in oof_residuals:
        c = r["category_group"]
        p = r["y_pred"]
        y = r["y_true"]
        b = get_bin(c, p)

        q = mondrian_q90[(c, b)]
        half_w = q * math.sqrt(p + 1.0)
        lower = max(0.0, p - half_w)
        upper = p + half_w
        width = upper - lower

        width_sum += width
        pred_sum += p

        is_covered = (lower <= y <= upper)
        if is_covered:
            covered_all += 1
            covered_by_cat[c] += 1
            covered_by_bin[b] += 1
        total_by_cat[c] += 1
        total_by_bin[b] += 1

    # Measure Ward-Total Coverage (156 wards)
    covered_ward = 0
    ward_width_sum = 0.0
    ward_pred_sum = 0.0
    for wid in ward_actual_tot:
        y_w = ward_actual_tot[wid]
        p_w = ward_pred_tot[wid]
        h_w = q90_ward_total * math.sqrt(p_w + 1.0)
        l_w = max(0.0, p_w - h_w)
        u_w = p_w + h_w
        if l_w <= y_w <= u_w:
            covered_ward += 1
        ward_width_sum += (u_w - l_w)
        ward_pred_sum += p_w

    coverage_report = {
        "overall_coverage_pct": round(covered_all / total_pts * 100.0, 2),
        "mean_width_to_pred_ratio": round(width_sum / max(1.0, pred_sum), 3),
        "category_coverage_pct": {
            c: round(covered_by_cat[c] / total_by_cat[c] * 100.0, 2)
            for c in config.CATEGORIES
        },
        "volume_bin_coverage_pct": {
            b: round(covered_by_bin[b] / total_by_bin[b] * 100.0, 2)
            for b in ["low", "med", "high"]
        },
        "ward_total_coverage_pct": round(covered_ward / len(ward_actual_tot) * 100.0, 2),
        "ward_total_mean_width_ratio": round(ward_width_sum / max(1.0, ward_pred_sum), 3)
    }

    # 4. Export data/predictions/ward_predictions_with_intervals.csv (4,320 rows)
    pred_path = config.WARD_PREDICTIONS_FILE
    with open(pred_path, mode="r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))

    interval_rows = []
    target_ward_preds = defaultdict(float)
    target_ward_cat_preds = defaultdict(lambda: defaultdict(float))

    for r in pred_rows:
        w = int(r["ward_id"])
        m = r["month"]
        c = r["category_group"]
        p = float(r["predicted_count"])

        b = get_bin(c, p)
        q = mondrian_q90[(c, b)]
        half_w = q * math.sqrt(p + 1.0)
        lower = max(0.0, round(p - half_w, 2))
        upper = round(p + half_w, 2)

        assert lower >= 0.0, f"Negative lower bound on ward {w}"
        assert lower <= p <= upper, f"Interval violates bounds: lower={lower}, pred={p}, upper={upper}"

        interval_rows.append({
            "ward_id": w,
            "month": m,
            "category_group": c,
            "predicted_count": p,
            "lower_90": lower,
            "upper_90": upper
        })

        target_ward_preds[w] += p
        target_ward_cat_preds[w][c] += p

    out_intervals_path = config.WARD_PREDICTIONS_INTERVALS_FILE
    with open(out_intervals_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ward_id", "month", "category_group", "predicted_count", "lower_90", "upper_90"]
        )
        writer.writeheader()
        writer.writerows(interval_rows)

    # 5. Export data/predictions/ward_summary_intervals.csv (40 wards)
    # Load ward names from ward_features.csv
    with open(config.WARD_FEATURES_FILE, mode="r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    ward_names = {int(r["ward_id"]): r["ward_name"] for r in feat_rows if r["ward_id"]}

    summary_interval_rows = []
    target_wards = sorted(list(target_ward_preds.keys()))

    for w in target_wards:
        total_pred = round(target_ward_preds[w], 2)
        h_w = q90_ward_total * math.sqrt(total_pred + 1.0)
        total_lower = max(0.0, round(total_pred - h_w, 2))
        total_upper = round(total_pred + h_w, 2)
        rel_width = (total_upper - total_lower) / max(1.0, total_pred)

        if rel_width < config.CONFIDENCE_HIGH_THRESHOLD:
            conf_label = "High"
        elif rel_width < config.CONFIDENCE_MEDIUM_THRESHOLD:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        top_cat = max(target_ward_cat_preds[w], key=target_ward_cat_preds[w].get)

        summary_interval_rows.append({
            "ward_id": w,
            "ward_name": ward_names.get(w, f"Ward {w}"),
            "total_pred_18m": total_pred,
            "total_lower_90": total_lower,
            "total_upper_90": total_upper,
            "top_category": top_cat,
            "confidence_label": conf_label
        })

    out_summary_path = config.WARD_SUMMARY_INTERVALS_FILE
    with open(out_summary_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ward_id", "ward_name", "total_pred_18m", "total_lower_90", "total_upper_90", "top_category", "confidence_label"]
        )
        writer.writeheader()
        writer.writerows(summary_interval_rows)

    # Save coverage report to data/metrics/interval_coverage.json
    cov_path = config.DATA_DIR / "metrics" / "interval_coverage.json"
    with open(cov_path, mode="w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2)

    print("\n[SUCCESS] Conformal prediction intervals generated & calibrated!")
    print(f"Overall 90% Empirical Coverage: {coverage_report['overall_coverage_pct']}% (Mean width/pred ratio: {coverage_report['mean_width_to_pred_ratio']}x)")
    print(f"Per-Category Coverage: {coverage_report['category_coverage_pct']}")
    print(f"Per-Volume-Bin Coverage: {coverage_report['volume_bin_coverage_pct']}")
    print(f"Ward-Total Coverage: {coverage_report['ward_total_coverage_pct']}% (Mean width/pred ratio: {coverage_report['ward_total_mean_width_ratio']}x)")
    print(f"Exported: {out_intervals_path} (4,320 rows) & {out_summary_path} (40 wards)")

    return coverage_report

if __name__ == "__main__":
    calibrate_and_generate_intervals()
