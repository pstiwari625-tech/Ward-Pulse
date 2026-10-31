"""
Test Step 1: No Water Category Check (tests/test_no_water.py)
Validates that:
 1. "water" does NOT appear in any tracked repo file (code, docs, frontend, backend, predictions)
    except the explicit README.md domain clarification note:
    "Water supply/sewerage is handled by BWSSB and is not in this dataset, so it is not modelled."
    (and natural raw complaint text descriptions like "water leakage on road" in raw grievance data).
 2. Every prediction category in data/predictions/ward_predictions.csv is strictly
    one of the 6 canonical BBMP categories defined in config.CATEGORIES:
    ['animals_health', 'electrical', 'other', 'roads', 'solid_waste', 'trees'].
"""

import sys
import csv
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def test_no_water_in_tracked_files():
    # 1. Inspect git tracked files
    try:
        tracked_files = subprocess.check_output(
            ["git", "ls-files"], cwd=str(BASE_DIR), text=True
        ).splitlines()
    except Exception:
        tracked_files = [str(p.relative_to(BASE_DIR)) for p in BASE_DIR.rglob("*.py")]

    violations = []
    ALLOWED_RAW_FILES = {"data/raw/grievances_public_wards.csv", "grievances_public_wards.csv"}
    ALLOWED_TEST_FILES = {
        "tests/test_no_water.py",
        "tests/test_audit_a_contracts.py",
        "tests/run_full_audit.py"
    }

    for f_rel in tracked_files:
        f_path = BASE_DIR / f_rel
        if not f_path.is_file() or f_rel in ALLOWED_RAW_FILES or f_rel in ALLOWED_TEST_FILES:
            continue
        try:
            with open(f_path, mode="r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    if "water" in line.lower():
                        # Exclude the explicit README domain clarification note
                        if "handled by bwssb and is not in this dataset" in line.lower():
                            continue
                        violations.append((f_rel, line_idx, line.strip()[:80]))
        except Exception:
            pass

    assert len(violations) == 0, f"Found forbidden 'water' references in tracked files:\n{violations}"
    print(f"[PASS] No forbidden 'water' references found in any tracked files.")


def test_prediction_categories_strictly_six():
    pred_path = config.WARD_PREDICTIONS_FILE
    assert pred_path.exists(), f"Predictions file not found: {pred_path}"

    with open(pred_path, mode="r", encoding="utf-8") as f:
        categories_in_pred = sorted({r["category_group"] for r in csv.DictReader(f)})

    canonical_categories = sorted(config.CATEGORIES)
    assert categories_in_pred == canonical_categories, (
        f"Categories mismatch! Found: {categories_in_pred}, Expected canonical 6: {canonical_categories}"
    )

    # Check summary columns if present
    summary_path = config.WARD_SUMMARY_FILE
    if summary_path.exists():
        with open(summary_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            pred_cols = [c.replace("pred_", "") for c in fields if c.startswith("pred_")]
            if pred_cols:
                assert sorted(pred_cols) == canonical_categories, (
                    f"Summary pred columns mismatch! Found: {sorted(pred_cols)}, Expected: {canonical_categories}"
                )

    print(f"[PASS] Prediction categories strictly match canonical 6: {canonical_categories}")

if __name__ == "__main__":
    test_no_water_in_tracked_files()
    test_prediction_categories_strictly_six()
