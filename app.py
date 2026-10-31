"""
Ward Pulse - Main Application Entry Point
Phase 1: Frontend Foundation & UI Setup
Civic Intelligence Dashboard for Lucknow Municipal Corporation (LMC)
"""

import sys
from pathlib import Path

# Ensure ward-pulse root is in sys.path for robust cross-directory execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.data_loader import load_ward_predictions, load_ward_summary, load_geojson
from components.sidebar import render_sidebar
from components.dashboard import (
    render_header,
    render_overview,
    render_analytics_view,
    render_map_view,
    render_governance_view,
    render_swagger_api_view,
    render_about_view
)


def load_custom_css():
    """Load and inject custom CSS from assets/style.css."""
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom style.css not found. Default Streamlit styles applied.")


def main():
    """Initialize application layout, styling, and navigation routing."""
    # Page configuration
    st.set_page_config(
        page_title="Ward Pulse | Civic Demand Intelligence",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom UI theme
    load_custom_css()

    # Load data status and available prediction datasets
    df_predictions, data_status = load_ward_predictions()
    df_summary = load_ward_summary()
    geojson_data = load_geojson()

    # Render sidebar navigation
    active_view = render_sidebar(data_status)

    # Render persistent header
    render_header(data_status)

    # View Routing
    if active_view == "Overview":
        render_overview(df_predictions, data_status, df_summary, geojson_data)
    elif active_view == "Ward Analytics":
        render_analytics_view(df_predictions, df_summary)
    elif active_view == "Map View":
        render_map_view(df_summary, geojson_data)
    elif active_view == "Model Governance":
        render_governance_view()
    elif active_view == "Swagger API":
        render_swagger_api_view()
    elif active_view == "About Project":
        render_about_view()
    else:
        render_overview(df_predictions, data_status, df_summary, geojson_data)


if __name__ == "__main__":
    main()
