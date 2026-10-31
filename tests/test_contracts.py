"""
Ward Pulse: Data Contract Validation Test
Validates format, row counts, non-negativity, and schema invariants of exported predictions.
"""

from pathlib import Path
import csv

def test_prediction_grid_contract():
    pred_path = Path("data/predictions/ward_predictions.csv")
    assert pred_path.exists(), "Prediction contract file does not exist"

    with open(pred_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 4320, f"Expected exactly 4,320 prediction rows, got {len(rows)}"
    expected_fields = {"ward_id", "month", "category_group", "predicted_count"}
    assert set(reader.fieldnames) == expected_fields, f"Fieldnames deviate from contract: {reader.fieldnames}"

    for r in rows:
        val = float(r["predicted_count"])
        assert val >= 0.0, f"Negative prediction encountered: {r}"

def test_summary_contract():
    sum_path = Path("data/predictions/ward_summary_latest.csv")
    assert sum_path.exists(), "Summary contract file does not exist"

    with open(sum_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 40, f"Expected 40 target wards in summary, got {len(rows)}"
    for r in rows:
        cpi = float(r["cpi_score"])
        assert 0.0 <= cpi <= 1.0, f"CPI score out of [0, 1] bounds: {cpi}"
        assert r["priority_tier"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}, f"Invalid tier: {r['priority_tier']}"

if __name__ == "__main__":
    test_prediction_grid_contract()
    test_summary_contract()
    print("[SUCCESS] All contract tests passed.")
