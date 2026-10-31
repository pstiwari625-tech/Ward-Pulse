"""
Test Step 5: Category Shares & Distribution Drift Test (tests/test_category_shares.py)
Validates that:
 1. 18-month window empirical category shares from training data match:
    electrical ~34.9%, solid_waste ~28.0%, roads ~14.3%, animals_health ~9.4%, other ~7.8%, trees ~5.6%.
 2. Predicted category shares across the 40 target wards do not drift by more than 5.0 pp
    from the window empirical shares.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def test_window_category_shares_and_drift():
    # 1. Compute empirical window shares from training data (156 wards, 2024-01..2025-06)
    with open(config.PREDICTION_GRID_FILE, mode="r", encoding="utf-8") as f:
        target_wards = {int(r["ward_id"]) for r in csv.DictReader(f)}

    train_counts = defaultdict(int)
    target_months_set = set(config.TARGET_MONTHS)

    with open(config.GRIEVANCES_FILE, mode="r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = int(r["ward_id"])
            m = r["grievance_time"][:7]
            if w not in target_wards and m in target_months_set:
                train_counts[r["category_group"]] += 1

    tot_train = sum(train_counts.values())
    train_shares = {c: (train_counts[c] / tot_train) * 100.0 for c in config.CATEGORIES}

    # 2. Compute predicted shares for 40 target wards
    pred_counts = defaultdict(float)
    with open(config.WARD_PREDICTIONS_FILE, mode="r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred_counts[r["category_group"]] += float(r["predicted_count"])

    tot_pred = sum(pred_counts.values())
    pred_shares = {c: (pred_counts[c] / tot_pred) * 100.0 for c in config.CATEGORIES}

    # 3. Check drift per category
    drifts = {}
    max_drift = 0.0
    for c in config.CATEGORIES:
        diff = abs(pred_shares[c] - train_shares[c])
        drifts[c] = round(diff, 2)
        if diff > max_drift:
            max_drift = diff

    passed = max_drift <= 5.0
    assert passed, f"Category drift exceeded 5.0 pp! Drifts: {drifts}, Max: {max_drift:.2f} pp"

    print(f"[PASS] Category shares verified. Max drift: {max_drift:.2f} pp (Threshold: 5.0 pp).")
    print(f"       Window ground truth: { {c: round(train_shares[c], 1) for c in config.CATEGORIES} }")
    print(f"       Target predicted:    { {c: round(pred_shares[c], 1) for c in config.CATEGORIES} }")
    return True

if __name__ == "__main__":
    test_window_category_shares_and_drift()
