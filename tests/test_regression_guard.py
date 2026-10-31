"""
Regression Guard Test (Step 0)
Validates that model inference and the primary predictions contract data/predictions/ward_predictions.csv
remain bit-identical to the golden baseline (tests/golden/preds_v1.csv) within 1e-9 tolerance.
Asserts:
 1. File exists and has exactly 4,320 rows.
 2. Key set equals prediction_grid_template.csv exactly (no missing, extra, or duplicate keys).
 3. No nulls, NaN, or inf values, all values >= 0, float dtype.
 4. Re-running production inference produces predictions identical to golden baseline (diff < 1e-9).
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from backend.preprocessing import clean_ward_feature_row
from backend.features import extract_ward_feature_vector
from backend.inference import generate_production_predictions

def test_regression_guard():
    pred_path = BASE_DIR / "data" / "predictions" / "ward_predictions.csv"
    golden_path = BASE_DIR / "tests" / "golden" / "preds_v1.csv"
    template_path = BASE_DIR / "prediction_grid_template.csv"
    if not template_path.exists():
        template_path = BASE_DIR / "data" / "raw" / "prediction_grid_template.csv"

    assert pred_path.exists(), f"Missing predictions file: {pred_path}"
    assert golden_path.exists(), f"Missing golden predictions file: {golden_path}"
    assert template_path.exists(), f"Missing template file: {template_path}"

    # 1. Read current predictions file
    with open(pred_path, mode="r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))

    # Check 1: 4,320 rows
    assert len(pred_rows) == 4320, f"Expected 4320 rows, got {len(pred_rows)}"

    # Check 2: Schema
    expected_fields = ["ward_id", "month", "category_group", "predicted_count"]
    assert list(pred_rows[0].keys()) == expected_fields, f"Header mismatch: {pred_rows[0].keys()}"

    # Check 3: Key set matches template
    with open(template_path, mode="r", encoding="utf-8") as f:
        template_keys = [(int(r["ward_id"]), r["month"], r["category_group"]) for r in csv.DictReader(f)]

    pred_keys = [(int(r["ward_id"]), r["month"], r["category_group"]) for r in pred_rows]
    assert len(pred_keys) == len(set(pred_keys)), "Duplicate keys found in predictions"
    assert set(pred_keys) == set(template_keys), "Key set does not match prediction_grid_template.csv"

    # Check 4: No nulls, NaNs, all >= 0
    pred_dict = {}
    for r in pred_rows:
        val_str = r["predicted_count"]
        assert val_str not in ("", "NaN", "null", "None", "inf", "-inf"), f"Invalid numeric value: {val_str}"
        val = float(val_str)
        assert val == val and val >= 0.0, f"Negative or NaN prediction: {val}"
        pred_dict[(int(r["ward_id"]), r["month"], r["category_group"])] = val

    # 2. Compare against Golden file
    with open(golden_path, mode="r", encoding="utf-8") as f:
        golden_rows = list(csv.DictReader(f))

    assert len(golden_rows) == 4320, f"Golden file row count mismatch: {len(golden_rows)}"
    max_golden_diff = 0.0
    for r in golden_rows:
        k = (int(r["ward_id"]), r["month"], r["category_group"])
        assert k in pred_dict, f"Key {k} missing from predictions"
        diff = abs(float(r["predicted_count"]) - pred_dict[k])
        if diff > max_golden_diff:
            max_golden_diff = diff
    assert max_golden_diff < 1e-9, f"Predictions drifted from golden baseline! Max diff: {max_golden_diff}"

    # 3. Re-run inference and verify identical reproduction
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

    fresh_preds = generate_production_predictions(
        train_wards=train_wards,
        target_wards=target_wards,
        feats_dict=feats_dict,
        counts_dict=counts_dict,
        target_months=config.TARGET_MONTHS,
        categories=config.CATEGORIES,
        kernel_ell=3.5,
        ridge_alpha=5.0
    )

    max_infer_diff = 0.0
    for p in fresh_preds:
        k = (p["ward_id"], p["month"], p["category_group"])
        diff = abs(p["predicted_count"] - pred_dict[k])
        if diff > max_infer_diff:
            max_infer_diff = diff
    assert max_infer_diff < 1e-9, f"Fresh inference mismatch: {max_infer_diff}"

    print(f"[PASS] Regression Guard: 4,320 rows valid, golden diff={max_golden_diff:.2e}, inference diff={max_infer_diff:.2e}")
    return True

if __name__ == "__main__":
    test_regression_guard()
