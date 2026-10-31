"""
Ward Pulse - Main Dashboard Component
Clean, modern municipal intelligence dashboard designed for executive presentations.
Strictly eliminates visible raw HTML, development banners, and clutter.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

from components.charts import (
    render_analytics_placeholder,
    plot_top_10_wards,
    plot_category_visualization,
    plot_demand_distribution,
    render_ward_details_section
)
from components.ward_map import render_official_ward_map, render_map_placeholder
from utils.data_loader import get_kpi_metrics, load_model_metrics, load_model_metadata


def render_header(data_status: Dict[str, Any]):
    """
    Renders the persistent top header with title, municipal jurisdiction, and status badge.
    Fully unindented HTML block preventing CommonMark code block rendering.
    """
    badge_class = data_status.get("badge_class", "badge-live")
    badge_label = data_status.get("badge_label", "System Active")
    
    st.markdown(
        f"""<div class="header-container">
<div class="header-title-group">
<h1>🏛️ Ward Pulse</h1>
<p>Municipal Civic Demand Intelligence • Decision-Support Dashboard</p>
</div>
<div class="header-meta">
<span class="badge badge-location">📍 Official Municipal Grid (198 Wards)</span>
<span class="badge {badge_class}"><span class="badge-pill-pulse"></span> {badge_label}</span>
</div>
</div>""",
        unsafe_allow_html=True
    )


def render_kpi_cards(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Feature A: KPI Cards.
    Displays dynamic metrics calculated strictly from the loaded dataset.
    Uses Streamlit columns with self-contained HTML blocks to guarantee zero raw HTML leakage.
    """
    kpis = get_kpi_metrics(df, df_summary)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="kpi-card primary-card">
<div class="kpi-header"><span class="kpi-title">Total Wards</span><span class="kpi-icon">🏙️</span></div>
<div class="kpi-value">{kpis['total_wards']}</div>
<div class="kpi-subtitle"><span>Monitored municipal wards</span></div>
</div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class="kpi-card accent-card">
<div class="kpi-header"><span class="kpi-title">Total Forecast Complaints</span><span class="kpi-icon">📋</span></div>
<div class="kpi-value">{kpis['total_predicted_complaints']}</div>
<div class="kpi-subtitle"><span>Cumulative city-wide demand</span></div>
</div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class="kpi-card">
<div class="kpi-header"><span class="kpi-title">Highest Demand Ward</span><span class="kpi-icon">📍</span></div>
<div class="kpi-value" style="font-size: 1.15rem; color: #1E293B;">{kpis.get('highest_demand_ward', '--')}</div>
<div class="kpi-subtitle"><span>Peak grievance volume</span></div>
</div>""",
            unsafe_allow_html=True
        )

    with col4:
        top_cat = kpis.get("top_demanding_category", "--")
        if top_cat != "--":
            label = "Top Demand Category"
            subtext = "Leading grievance sector"
            val = top_cat
            icon = "🏷️"
            val_style = "font-size: 1.05rem; color: #168C83;"
        else:
            label = "High Priority Wards"
            subtext = "High & Critical priority tiers"
            val = kpis.get('high_demand_wards', '--')
            icon = "🔥"
            val_style = "font-size: 1.25rem; color: #EF4444;"

        st.markdown(
            f"""<div class="kpi-card">
<div class="kpi-header"><span class="kpi-title">{label}</span><span class="kpi-icon">{icon}</span></div>
<div class="kpi-value" style="{val_style}">{val}</div>
<div class="kpi-subtitle"><span>{subtext}</span></div>
</div>""",
            unsafe_allow_html=True
        )


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applies sidebar zone and ward filters to the active dataframe."""
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    # Zone filter
    selected_zone = st.session_state.get("filter_zone", "All Administrative Zones")
    if selected_zone and selected_zone != "All Administrative Zones" and "zone" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["zone"] == selected_zone]

    # Ward filter
    selected_ward = st.session_state.get("filter_ward", "All Municipal Wards")
    if selected_ward and selected_ward != "All Municipal Wards" and "ward_id" in filtered_df.columns:
        try:
            wid = int(selected_ward.split(" - ")[0].replace("Ward", "").strip())
            filtered_df = filtered_df[filtered_df["ward_id"] == wid]
        except (ValueError, IndexError):
            pass

    return filtered_df


def render_overview(
    df: Optional[pd.DataFrame],
    data_status: Dict[str, Any],
    df_summary: Optional[pd.DataFrame] = None,
    geojson_data: Optional[Dict[str, Any]] = None
):
    """
    Renders the primary executive dashboard Overview:
    1. Executive KPI Cards
    2. Side-by-side Top 10 Wards and Department Category Breakdown
    3. Demand Distribution Analysis
    4. Searchable & Sortable Ward Details Registry Table
    """
    raw_df = df if df is not None and not df.empty else df_summary
    active_df = apply_sidebar_filters(raw_df) if raw_df is not None else None

    # Render Executive KPI Section
    render_kpi_cards(active_df, df_summary)

    if active_df is not None and not active_df.empty:
        # 1. Top 10 Wards & Category Breakdown side-by-side
        col1, col2 = st.columns(2)
        with col1:
            plot_top_10_wards(active_df)

        with col2:
            plot_category_visualization(active_df)

        # 2. Demand Distribution Analysis
        plot_demand_distribution(active_df)

        # 3. Interactive Ward Details Table
        render_ward_details_section(active_df)
    else:
        render_analytics_placeholder()


def render_analytics_view(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Renders the dedicated Ward Analytics page.
    """
    raw_df = df if df is not None and not df.empty else df_summary
    active_df = apply_sidebar_filters(raw_df) if raw_df is not None else None

    st.markdown(
        """<div style="margin-bottom: 1.25rem;">
<h2 style="color: var(--primary); margin-bottom: 0.25rem; font-size: 1.5rem;">📈 Ward Demand Analytics</h2>
<p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Comparative demand intelligence, category projections, and ward demand stratification.</p>
</div>""",
        unsafe_allow_html=True
    )

    if active_df is not None and not active_df.empty:
        render_kpi_cards(active_df, df_summary)
        
        col1, col2 = st.columns(2)
        with col1:
            plot_top_10_wards(active_df)

        with col2:
            plot_category_visualization(active_df)

        plot_demand_distribution(active_df)
        render_ward_details_section(active_df)
    else:
        render_analytics_placeholder()


def render_map_view(df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders the dedicated Map View page with official 198 ward boundaries.
    Integrates official hackathon GeoJSON polygons and socio-economic attributes.
    """
    render_official_ward_map(df_summary=df_summary)


def render_about_view():
    """
    Renders the About Project page with architecture details and model audit metrics.
    """
    metrics = load_model_metrics()
    metadata = load_model_metadata()

    st.markdown(
        """<div class="ward-detail-card" style="margin-top: 0; margin-bottom: 1.5rem;">
<div style="border-bottom: 1px solid var(--border-color); padding-bottom: 0.75rem;">
<h3 style="margin: 0; color: var(--primary); font-size: 1.3rem;">ℹ️ About Ward Pulse</h3>
<div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
Civic intelligence and decision-support platform for municipal resource allocation.
</div>
</div>
<div style="font-size: 0.925rem; line-height: 1.7; color: var(--text-primary); margin-top: 1rem;">
<p>
<strong>Ward Pulse</strong> is a geospatial civic demand forecasting platform designed to estimate municipal complaint volume 
across urban wards. By leveraging socio-economic indicators, infrastructure density features, and spatial autocorrelation, 
the system reliably infers grievance volume even in newly incorporated or unobserved wards with limited historical records.
</p>
</div>
</div>""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Performance Audit")
        st.json(metrics)
    with col2:
        st.subheader("Top Predictive Features")
        if metadata:
            st.json(metadata.get("top_feature_importances", {}))
        else:
            st.info("Feature importance metadata loaded from models/model_metadata.json.")


def render_welcome_card():
    """Deprecated: Removed for executive presentation clarity."""
    pass


def render_status_banner(data_status: Optional[Dict[str, Any]] = None):
    """Deprecated: Header badge handles live status cleanly."""
    pass
