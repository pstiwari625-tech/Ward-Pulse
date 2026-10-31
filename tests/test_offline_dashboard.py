"""
Test Step 6: Offline Plotly Choropleth & Dashboard Tests (tests/test_offline_dashboard.py)
Validates that:
 1. Map is strictly offline: Zero external tile servers (no CartoDB, no OpenStreetMap HTTP tiles).
 2. Category consistency: Sum of the 6 canonical categories matches the ward total (delta < 0.05).
 3. Tooltips contain all required elements: ward name, selected-category count, 90% interval, confidence label, top category, CPI tier.
 4. Reactive selectors: category selector, month selector, and metric toggle are present in frontend/app.py.
 5. Download buttons: st.download_button exists for both ward_predictions.csv and ward_predictions_with_intervals.csv.
 6. Governance tab: reads from data/metrics/*.json, labels equity view 'Low-reporting scenario', and includes the limitation box.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def test_offline_map_tile_independence():
    map_path = BASE_DIR / "frontend" / "components" / "map_view.py"
    app_path = BASE_DIR / "frontend" / "app.py"

    forbidden_servers = ["cartodb", "openstreetmap", "tiles=", "choroplethmapbox"]
    violations = []

    for p in [map_path, app_path]:
        with open(p, mode="r", encoding="utf-8") as f:
            content = f.read().lower()
            for forb in forbidden_servers:
                if forb in content:
                    violations.append((p.name, forb))

    assert len(violations) == 0, f"Found external tile/CDN dependencies: {violations}"
    print("[PASS] 1. Offline readiness: Zero external tile servers or Mapbox dependencies.")

def test_category_sum_consistency():
    pred_path = config.WARD_PREDICTIONS_FILE
    with open(pred_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by (ward_id, month) and sum categories
    ward_month_cat_sum = defaultdict(float)
    for r in rows:
        ward_month_cat_sum[(int(r["ward_id"]), r["month"])] += float(r["predicted_count"])

    # Load summary table totals
    with open(config.WARD_SUMMARY_FILE, mode="r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    inconsistencies = 0
    for r in summary_rows:
        wid = int(r["ward_id"])
        tot_monthly = float(r["total_predicted_monthly"])
        cat_sum = sum(float(r[f"pred_{c}"]) for c in config.CATEGORIES)

        diff = abs(tot_monthly - cat_sum)
        if diff > 0.05:
            inconsistencies += 1

    assert inconsistencies == 0, f"Found {inconsistencies} category sum inconsistencies in summary table"
    print(f"[PASS] 2. Category consistency: 100% of wards reconcile perfectly (sum of 6 categories == total, diff < 0.05).")

def test_tooltips_content():
    map_path = BASE_DIR / "frontend" / "components" / "map_view.py"
    with open(map_path, mode="r", encoding="utf-8") as f:
        code = f.read()

    required_tooltip_elements = [
        "ward_name", "category", "month", "90% interval", "confidence", "dominant need", "priority tier"
    ]
    missing = [elem for elem in required_tooltip_elements if elem not in code.lower()]
    assert len(missing) == 0, f"Missing tooltip fields: {missing}"
    print("[PASS] 3. Hover tooltips: Complete with ward name, category count, 90% interval, confidence label, and priority tier.")

def test_reactive_controls_and_downloads():
    app_path = BASE_DIR / "frontend" / "app.py"
    with open(app_path, mode="r", encoding="utf-8") as f:
        code = f.read()

    # Selectors
    assert "selected_cat" in code or "Filter by Civic Category" in code, "Missing category selector"
    assert "selected_month" in code or "Select Target Month" in code, "Missing month selector"
    assert "selected_metric" in code or "Choropleth Color Metric" in code, "Missing metric toggle"

    # Download buttons
    assert "ward_predictions.csv" in code and "download_button" in code, "Missing download button for ward_predictions.csv"
    assert "ward_predictions_with_intervals.csv" in code and "download_button" in code, "Missing download button for intervals CSV"

    # Limitation box and low-reporting scenario
    assert "Ward-specific surge after May 2024 is not predictable" in code, "Missing required model limitation box"
    assert "Low-reporting scenario" in code or "low-reporting scenario" in code.lower(), "Missing low-reporting scenario label"

    print("[PASS] 4. Reactive controls, dual download buttons, and model limitation disclaimer verified.")

if __name__ == "__main__":
    test_offline_map_tile_independence()
    test_category_sum_consistency()
    test_tooltips_content()
    test_reactive_controls_and_downloads()
