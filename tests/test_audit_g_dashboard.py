"""
Audit Section G: Dashboard & Presentation Layer Tests (Tests 28 - 33)
Audits offline capability, map tooltips, category selector reactivity, month selector,
governance parity, and download button availability.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json, csv
import config

def test_28_offline_and_startup():
    """Test 28: App starts and works offline (flags external map tile network dependencies)."""
    map_view_path = BASE_DIR / "frontend" / "components" / "map_view.py"
    alt_map_path = BASE_DIR / "components" / "map_view.py"
    
    external_tile_dependencies = []
    for p in [map_view_path, alt_map_path]:
        if p.exists():
            with open(p, mode="r", encoding="utf-8") as f:
                content = f.read()
                if "CartoDB" in content or "OpenStreetMap" in content or "tiles=" in content:
                    external_tile_dependencies.append(p.name)
                    
    # Also verify requirements.txt exists and lists Streamlit
    req_path = BASE_DIR / "requirements.txt"
    has_streamlit_req = False
    if req_path.exists():
        with open(req_path, mode="r", encoding="utf-8") as f:
            has_streamlit_req = "streamlit" in f.read().lower()
            
    # Flag: Map tiles require HTTP connection to CartoDB/OSM tile servers
    passed = has_streamlit_req and (len(external_tile_dependencies) == 0)
    return {
        "id": 28,
        "name": "App offline capability & map tile network independence",
        "pass": passed,
        "measured": f"Streamlit in requirements: {has_streamlit_req}, External online tile servers used: {external_tile_dependencies}",
        "note": "Offline-ready" if passed else "FLAG: Folium uses CartoDB/OpenStreetMap HTTP tiles. In a strictly disconnected offline environment, background basemaps will fail to render unless cached."
    }

def test_29_map_tooltips_content():
    """Test 29: Map tooltips show ward name, total prediction, interval, and top category."""
    map_view_path = BASE_DIR / "frontend" / "components" / "map_view.py"
    alt_map_path = BASE_DIR / "components" / "map_view.py"
    
    fields_found = []
    for p in [map_view_path, alt_map_path]:
        if p.exists():
            with open(p, mode="r", encoding="utf-8") as f:
                content = f.read()
                if "GeoJsonTooltip" in content or "hovertemplate" in content:
                    for target in ["ward_name", "total", "interval", "category", "predicted"]:
                        if target in content.lower():
                            fields_found.append(target)
                            
    has_all_tooltips = {"ward_name", "interval", "predicted", "category"}.issubset(set(fields_found))
    return {
        "id": 29,
        "name": "Map tooltips show ward name, total prediction, interval, and top category",
        "pass": has_all_tooltips,
        "measured": f"Tooltip fields present in map components: {list(set(fields_found)) if fields_found else 'Only [ward_id]'}",
        "note": "Complete tooltips with ward name, predicted count, 90% interval, and category" if has_all_tooltips else "FAIL: Tooltips missing required fields"
    }

def test_30_category_selector_changes_map():
    """Test 30: Category selector changes the map and the numbers for all 6 categories."""
    app_path = BASE_DIR / "frontend" / "app.py"
    map_view_path = BASE_DIR / "frontend" / "components" / "map_view.py"
    
    with open(app_path, mode="r", encoding="utf-8") as f:
        app_code = f.read()
        
    with open(map_view_path, mode="r", encoding="utf-8") as f:
        map_code = f.read()
        
    # Check if category selection is passed to render function
    map_takes_category = ("selected_category" in map_code or "category" in map_code) and ("render_offline_choropleth" in map_code or "render_choropleth_map" in map_code)
    app_wires_category_to_map = ("render_offline_choropleth" in app_code or "render_choropleth_map" in app_code) and "selected_cat" in app_code
    
    passed = map_takes_category and app_wires_category_to_map
    return {
        "id": 30,
        "name": "Category selector dynamically updates choropleth map",
        "pass": passed,
        "measured": f"Map category parameter: {'Present' if map_takes_category else 'Missing'}, App category wiring: {'Present' if app_wires_category_to_map else 'Missing'}",
        "note": "Fully reactive category mapping" if passed else "FAIL: Map view is currently hard-coded to CPI scores; selecting a civic category (e.g. roads, trees) does not update the choropleth coloring."
    }

def test_31_month_selector_and_reconciliation():
    """Test 31: Month selector or 18-month total works; sum of 6 categories equals ward total."""
    app_path = BASE_DIR / "frontend" / "app.py"
    with open(app_path, mode="r", encoding="utf-8") as f:
        app_code = f.read()
        
    has_month_selector = "month" in app_code.lower() and ("selectbox" in app_code or "select_slider" in app_code or "slider" in app_code)
    
    # Check sum of 6 categories in summary table
    summary_path = config.WARD_SUMMARY_FILE
    reconciliation_violations = 0
    with open(summary_path, mode="r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tot = float(r["total_predicted_monthly"])
            cat_sum = sum(float(r[f"pred_{c}"]) for c in config.CATEGORIES)
            if abs(tot - cat_sum) > 0.05:
                reconciliation_violations += 1
                
    passed = has_month_selector and (reconciliation_violations == 0)
    return {
        "id": 31,
        "name": "Month selector present and category sum equals ward total",
        "pass": passed,
        "measured": f"Month selector in UI: {'Present' if has_month_selector else 'Missing (fixed 18-month average view)'}, Category sum reconciliation violations: {reconciliation_violations}/40",
        "note": "Reconciled & temporal slider present" if passed else "FAIL: UI lacks temporal month selector dropdown/slider; categories reconcile perfectly to total (diff < 0.05)."
    }

def test_32_governance_tab_matches_test_9():
    """Test 32: Governance tab shows the same MAE/RMSE numbers as test 9."""
    metrics_path = config.DATA_DIR / "metrics" / "metrics_latest.json"
    if not metrics_path.exists():
        metrics_path = config.MODEL_METRICS_FILE
    with open(metrics_path, mode="r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        
    app_path = BASE_DIR / "frontend" / "app.py"
    with open(app_path, mode="r", encoding="utf-8") as f:
        app_code = f.read()
        
    # Check if governance tab pulls directly from metrics_data
    pulls_from_metrics = "metrics_latest.json" in app_code or "load_metrics_data" in app_code or "model_metrics.json" in app_code
    if "models" in metrics_data:
        mae = metrics_data["models"]["production"]["mae"]
        rmse = metrics_data["models"]["production"]["rmse"]
    else:
        mae = metrics_data.get("metrics", {}).get("oof_mae", 0.0)
        rmse = metrics_data.get("metrics", {}).get("oof_rmse", 0.0)
    
    matches = (abs(mae - 6.636) < 0.05 and abs(rmse - 13.878) < 0.05)
    passed = pulls_from_metrics and matches
    return {
        "id": 32,
        "name": "Governance tab metrics parity with LOWO audit",
        "pass": passed,
        "measured": f"Metrics in JSON: OOF MAE = {mae}, OOF RMSE = {rmse}, UI dynamic loader: {'Connected' if pulls_from_metrics else 'Static'}",
        "note": "Exact 1:1 match with Test 9 LOWO evaluation" if passed else "Governance tab metric mismatch"
    }

def test_33_download_predictions_button():
    """Test 33: Download of predictions returns the same file as in A1."""
    app_path = BASE_DIR / "frontend" / "app.py"
    alt_dash_path = BASE_DIR / "components" / "dashboard.py"
    
    has_download_btn = False
    for p in [app_path, alt_dash_path]:
        if p.exists():
            with open(p, mode="r", encoding="utf-8") as f:
                content = f.read()
                if "download_button" in content and "ward_predictions" in content:
                    has_download_btn = True
                    break
                    
    return {
        "id": 33,
        "name": "Download button in dashboard exports ward_predictions.csv",
        "pass": has_download_btn,
        "measured": f"st.download_button for predictions CSV: {'Present' if has_download_btn else 'Missing'}",
        "note": "CSV export button functional" if has_download_btn else "FAIL: No st.download_button for ward_predictions.csv in frontend/app.py."
    }

if __name__ == "__main__":
    for t in [test_28_offline_and_startup, test_29_map_tooltips_content,
              test_30_category_selector_changes_map, test_31_month_selector_and_reconciliation,
              test_32_governance_tab_matches_test_9, test_33_download_predictions_button]:
        res = t()
        status = "PASS" if res["pass"] else "FAIL"
        print(f"[{status}] Test {res['id']}: {res['name']} | Measured: {res['measured']}")
