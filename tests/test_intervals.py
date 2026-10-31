"""
Test Step 3: Conformal Prediction Intervals Tests (tests/test_intervals.py)
Validates that:
 1. data/predictions/ward_predictions_with_intervals.csv has 4,320 rows, correct schema:
    [ward_id, month, category_group, predicted_count, lower_90, upper_90].
 2. Bounds validity: lower_90 >= 0, and lower_90 <= predicted_count <= upper_90 for all rows.
 3. data/predictions/ward_summary_intervals.csv has 40 wards, correct schema:
    [ward_id, ward_name, total_pred_18m, total_lower_90, total_upper_90, top_category, confidence_label].
 4. Ward-total bounds validity: total_lower_90 >= 0 and total_lower_90 <= total_pred_18m <= total_upper_90.
 5. Empirical coverage under LOWO falls strictly within 85-95% nominal band across:
    - Overall coverage
    - Every category (all 6)
    - Every volume bin (low, med, high)
    - Ward totals
"""

import sys
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def test_cell_intervals_contract_and_bounds():
    pred_path = config.WARD_PREDICTIONS_INTERVALS_FILE
    assert pred_path.exists(), f"Missing file: {pred_path}"

    with open(pred_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    expected_fields = ["ward_id", "month", "category_group", "predicted_count", "lower_90", "upper_90"]
    assert fields == expected_fields, f"Header mismatch: {fields}"
    assert len(rows) == 4320, f"Expected 4,320 rows, got {len(rows)}"

    # Check bounds
    for idx, r in enumerate(rows):
        p = float(r["predicted_count"])
        l = float(r["lower_90"])
        u = float(r["upper_90"])

        assert l >= 0.0, f"Row {idx}: negative lower bound {l}"
        assert l <= p, f"Row {idx}: lower ({l}) > pred ({p})"
        assert p <= u, f"Row {idx}: pred ({p}) > upper ({u})"

    print("[PASS] Cell intervals: 4,320 rows conformant with lower >= 0 and lower <= pred <= upper.")

def test_ward_summary_intervals_contract_and_bounds():
    summary_path = config.WARD_SUMMARY_INTERVALS_FILE
    assert summary_path.exists(), f"Missing file: {summary_path}"

    with open(summary_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    expected_fields = ["ward_id", "ward_name", "total_pred_18m", "total_lower_90", "total_upper_90", "top_category", "confidence_label"]
    assert fields == expected_fields, f"Header mismatch: {fields}"
    assert len(rows) == 40, f"Expected 40 target wards, got {len(rows)}"

    for idx, r in enumerate(rows):
        tot = float(r["total_pred_18m"])
        l = float(r["total_lower_90"])
        u = float(r["total_upper_90"])
        conf = r["confidence_label"]

        assert l >= 0.0, f"Ward {r['ward_id']}: negative total lower bound {l}"
        assert l <= tot, f"Ward {r['ward_id']}: lower ({l}) > tot ({tot})"
        assert tot <= u, f"Ward {r['ward_id']}: tot ({tot}) > upper ({u})"
        assert conf in {"High", "Medium", "Low"}, f"Invalid confidence label: {conf}"
        assert r["top_category"] in config.CATEGORIES, f"Invalid top category: {r['top_category']}"

    print("[PASS] Ward summary intervals: 40 wards conformant with High/Medium/Low confidence labels.")

def test_empirical_coverage_within_85_95_percent():
    cov_path = config.DATA_DIR / "metrics" / "interval_coverage.json"
    assert cov_path.exists(), f"Missing coverage report: {cov_path}"

    with open(cov_path, mode="r", encoding="utf-8") as f:
        cov = json.load(f)

    # 1. Overall coverage
    overall = cov["overall_coverage_pct"]
    assert 85.0 <= overall <= 95.0, f"Overall coverage out of 85-95% band: {overall}%"

    # 2. Per-category coverage
    for c, c_cov in cov["category_coverage_pct"].items():
        assert 85.0 <= c_cov <= 95.0, f"Category {c} coverage out of 85-95% band: {c_cov}%"

    # 3. Per-volume-bin coverage
    for b, b_cov in cov["volume_bin_coverage_pct"].items():
        assert 85.0 <= b_cov <= 95.0, f"Bin {b} coverage out of 85-95% band: {b_cov}%"

    # 4. Ward-total coverage
    ward_cov = cov["ward_total_coverage_pct"]
    assert 85.0 <= ward_cov <= 95.0, f"Ward total coverage out of 85-95% band: {ward_cov}%"

    print(f"[PASS] Empirical coverage strictly within 85-95% nominal band across all slices (Overall: {overall}%).")

if __name__ == "__main__":
    test_cell_intervals_contract_and_bounds()
    test_ward_summary_intervals_contract_and_bounds()
    test_empirical_coverage_within_85_95_percent()
