"""
Ward Pulse: Streamlit Municipal Decision-Support Platform
Interactive executive dashboard for civic complaint forecasts and CPI prioritization.
"""

import sys
from pathlib import Path

# Add root directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import streamlit as st
except ImportError:
    print("[ERROR] Streamlit is not installed. Run: pip install streamlit")
    sys.exit(1)

from frontend.utils import load_summary_data, load_metrics_data, load_metadata_data, load_geojson_data
from frontend.components.metrics_cards import render_kpi_metrics
from frontend.components.charts import render_category_distribution
from frontend.components.map_view import render_choropleth_map

def main():
    st.set_page_config(
        page_title="Ward Pulse — Municipal Civic Demand Intelligence",
        page_icon="🏙️",
        layout="wide"
    )

    st.title("🏙️ Ward Pulse — Civic Demand Intelligence")
    st.markdown("*Municipal decision-support and ward-level demand prediction platform (Lucknow Hackathon)*")

    # Load Contracts
    summary_data = load_summary_data()
    metrics_data = load_metrics_data()
    metadata_data = load_metadata_data()
    geojson_data = load_geojson_data()

    if not summary_data:
        st.warning("⚠️ No predictions found. Run `python -m backend.pipeline` first.")
        st.stop()

    # Metrics computation
    total_wards = len(summary_data)
    critical_wards = sum(1 for r in summary_data if r["priority_tier"] == "CRITICAL")
    mean_monthly = sum(float(r["total_predicted_monthly"]) for r in summary_data) / max(1, total_wards)
    oof_mae = metrics_data.get("metrics", {}).get("oof_mae", 6.84)

    # 1. Executive KPI Section
    render_kpi_metrics(st, total_wards, critical_wards, mean_monthly, oof_mae)
    st.divider()

    # 2. Main Analytics & Map View
    tab_map, tab_drilldown, tab_governance = st.tabs([
        "🗺️ Spatial Priority Map (CPI)",
        "📊 Category Forecasts",
        "⚖️ Model Governance & Explainability"
    ])

    with tab_map:
        st.subheader("Geographical Civic Priority Index (CPI) Overlay")
        st.caption("Color Key: 🔴 Critical (CPI >= 0.70) | 🟠 High (0.45 - 0.70) | 🟡 Medium (0.25 - 0.45) | 🟢 Low (< 0.25)")
        render_choropleth_map(st, summary_data, geojson_data)

    with tab_drilldown:
        st.subheader("Category-Wise Demand Projections")
        cat_sums = {
            "electrical": sum(float(r.get("pred_electrical", 0.0)) for r in summary_data),
            "solid_waste": sum(float(r.get("pred_solid_waste", 0.0)) for r in summary_data),
            "roads": sum(float(r.get("pred_roads", 0.0)) for r in summary_data),
            "animals_health": sum(float(r.get("pred_animals_health", 0.0)) for r in summary_data),
            "other": sum(float(r.get("pred_other", 0.0)) for r in summary_data),
            "trees": sum(float(r.get("pred_trees", 0.0)) for r in summary_data),
        }
        render_category_distribution(st, cat_sums)
        st.dataframe(summary_data, use_container_width=True)

    with tab_governance:
        st.subheader("Model Validation & Spatial Anti-Leakage Audit")
        st.json(metrics_data)
        st.subheader("Top Predictive Features")
        st.json(metadata_data.get("top_feature_importances", {}))

if __name__ == "__main__":
    main()
