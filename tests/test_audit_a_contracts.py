"""
Audit Section A: Data Contract Tests (Tests 1 - 6)
Validates prediction grid shape, keys, values, target wards, inductive setting, and absence of fabricated water category.
"""

from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent

def test_1_ward_predictions_shape_and_columns():
    """Test 1: data/predictions/ward_predictions.csv has exactly 4,320 rows and correct columns."""
    pred_path = BASE_DIR / "data" / "predictions" / "ward_predictions.csv"
    assert pred_path.exists(), f"File {pred_path} does not exist"
    
    with open(pred_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    expected_cols = ["ward_id", "month", "category_group", "predicted_count"]
    row_count_match = len(rows) == 4320
    cols_match = fieldnames == expected_cols
    
    return {
        "id": 1,
        "name": "ward_predictions.csv 4320 rows & schema",
        "pass": row_count_match and cols_match,
        "measured": f"Rows: {len(rows)}, Cols: {fieldnames}",
        "note": "Matches sample_submission.csv schema" if (row_count_match and cols_match) else "Row count or columns mismatch"
    }

def test_2_key_set_equals_template():
    """Test 2: Key set equals prediction_grid_template.csv exactly."""
    pred_path = BASE_DIR / "data" / "predictions" / "ward_predictions.csv"
    template_path = BASE_DIR / "prediction_grid_template.csv"
    if not template_path.exists():
        template_path = BASE_DIR / "data" / "raw" / "prediction_grid_template.csv"
        
    with open(pred_path, mode="r", encoding="utf-8") as f:
        pred_keys = [(int(r["ward_id"]), r["month"], r["category_group"]) for r in csv.DictReader(f)]
        
    with open(template_path, mode="r", encoding="utf-8") as f:
        template_keys = [(int(r["ward_id"]), r["month"], r["category_group"]) for r in csv.DictReader(f)]
        
    pred_set = set(pred_keys)
    template_set = set(template_keys)
    
    no_duplicates = len(pred_keys) == len(pred_set)
    sets_match = pred_set == template_set
    
    return {
        "id": 2,
        "name": "Key set equals prediction_grid_template.csv",
        "pass": sets_match and no_duplicates,
        "measured": f"Pred unique keys: {len(pred_set)}, Template keys: {len(template_set)}, Duplicates: {len(pred_keys) - len(pred_set)}",
        "note": "Exact 1:1 key correspondence" if (sets_match and no_duplicates) else "Key mismatch or duplicates detected"
    }

def test_3_no_nulls_nan_inf_non_negative():
    """Test 3: No nulls, no NaN/inf, all values >= 0, dtype float."""
    pred_path = BASE_DIR / "data" / "predictions" / "ward_predictions.csv"
    with open(pred_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    invalid_count = 0
    negative_count = 0
    non_float_count = 0
    
    for r in rows:
        val_str = r.get("predicted_count", "")
        if val_str in ("", "NaN", "null", "None", "inf", "-inf"):
            invalid_count += 1
            continue
        try:
            val = float(val_str)
            if val < 0.0 or val != val: # NaN check
                negative_count += 1
        except ValueError:
            non_float_count += 1
            
    all_valid = (invalid_count == 0 and negative_count == 0 and non_float_count == 0)
    return {
        "id": 3,
        "name": "No null/NaN/inf, all values >= 0, float dtype",
        "pass": all_valid,
        "measured": f"Invalid/Nulls: {invalid_count}, Negative: {negative_count}, Non-float: {non_float_count}",
        "note": "Clean non-negative float predictions" if all_valid else "Violations in numeric contract"
    }

def test_4_cardinalities_40_wards_18_months_6_categories():
    """Test 4: Exactly 40 target wards, 18 months (2024-01..2025-06), 6 categories."""
    pred_path = BASE_DIR / "data" / "predictions" / "ward_predictions.csv"
    with open(pred_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    wards = sorted({int(r["ward_id"]) for r in rows})
    months = sorted({r["month"] for r in rows})
    cats = sorted({r["category_group"] for r in rows})
    
    expected_months = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 7)]
    expected_cats = ["animals_health", "electrical", "other", "roads", "solid_waste", "trees"]
    
    wards_ok = len(wards) == 40
    months_ok = months == expected_months
    cats_ok = cats == expected_cats
    
    passed = wards_ok and months_ok and cats_ok
    return {
        "id": 4,
        "name": "Dimensions: 40 wards, 18 months, 6 categories",
        "pass": passed,
        "measured": f"Wards: {len(wards)}, Months: {len(months)} ({months[0]}..{months[-1]}), Categories: {len(cats)}",
        "note": "Exact dimension match" if passed else f"Mismatch (wards_ok={wards_ok}, months_ok={months_ok}, cats_ok={cats_ok})"
    }

def test_5_inductive_setting_zero_rows_in_grievances():
    """Test 5: Target wards have ZERO rows in grievances file (inductive setting)."""
    griev_path = BASE_DIR / "grievances_public_wards.csv"
    if not griev_path.exists():
        griev_path = BASE_DIR / "data" / "raw" / "grievances_public_wards.csv"
        
    pred_path = BASE_DIR / "data" / "predictions" / "ward_predictions.csv"
    with open(pred_path, mode="r", encoding="utf-8") as f:
        target_wards = {int(r["ward_id"]) for r in csv.DictReader(f)}
        
    with open(griev_path, mode="r", encoding="utf-8") as f:
        train_wards = {int(r["ward_id"]) for r in csv.DictReader(f)}
        
    overlap = target_wards.intersection(train_wards)
    passed = len(overlap) == 0
    return {
        "id": 5,
        "name": "Zero leakage: target wards absent from grievances",
        "pass": passed,
        "measured": f"Overlap ward count: {len(overlap)} (Target: 40, Train: {len(train_wards)})",
        "note": "Strict inductive setting verified" if passed else f"Leaked wards in training file: {overlap}"
    }

def test_6_water_must_not_appear_anywhere():
    """Test 6: 'Water' must NOT appear anywhere: the data has no water category."""
    # Audit prediction files, backend code, and UI components
    files_to_check = [
        BASE_DIR / "data" / "predictions" / "ward_predictions.csv",
        BASE_DIR / "data" / "predictions" / "ward_summary_latest.csv",
        BASE_DIR / "backend" / "features.py",
        BASE_DIR / "backend" / "training.py",
        BASE_DIR / "backend" / "inference.py",
        BASE_DIR / "frontend" / "app.py",
        BASE_DIR / "components" / "charts.py",
        BASE_DIR / "utils" / "data_loader.py",
        BASE_DIR / "README.md",
        BASE_DIR / "data" / "demo_predictions.csv"
    ]
    
    water_occurrences = []
    for fpath in files_to_check:
        if fpath.exists():
            with open(fpath, mode="r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    # Check for "water" used as a civic complaint category
                    if "water" in line.lower():
                        # Exclude natural language notes explicitly saying "no water category in data"
                        if "not present in the loaded dataset schema" in line or "no standalone water category" in line or "has no water category" in line or "handled by bwssb" in line.lower() or "is not modelled" in line.lower():
                            continue
                        water_occurrences.append((fpath.name, line_idx, line.strip()[:60]))
                        
    passed = len(water_occurrences) == 0
    return {
        "id": 6,
        "name": "'Water' category absence check (no fabrication)",
        "pass": passed,
        "measured": f"Water category references found: {len(water_occurrences)} occurrences across repo files",
        "note": "Clean: no fabricated water category" if passed else f"Found water category references in {[o[0] for o in water_occurrences[:3]]}"
    }

if __name__ == "__main__":
    for t in [test_1_ward_predictions_shape_and_columns, test_2_key_set_equals_template,
              test_3_no_nulls_nan_inf_non_negative, test_4_cardinalities_40_wards_18_months_6_categories,
              test_5_inductive_setting_zero_rows_in_grievances, test_6_water_must_not_appear_anywhere]:
        res = t()
        status = "PASS" if res["pass"] else "FAIL"
        print(f"[{status}] Test {res['id']}: {res['name']} | Measured: {res['measured']}")
