"""
Ward Pulse - Main Dashboard Component
Renders the header, status indicators, KPI metric cards, and primary views.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

from components.charts import render_analytics_section, render_analytics_placeholder, render_dataset_preview
from components.ward_map import render_choropleth_map, render_map_placeholder
from utils.data_loader import get_kpi_metrics, load_ward_summary, load_geojson, load_model_metrics, load_model_metadata


def render_header(data_status: Dict[str, Any]):
    """
    Renders the persistent top header with title, location, and data status indicator.
    """
    badge_class = data_status.get("badge_class", "badge-pending")
    badge_label = data_status.get("badge_label", "Official Dataset Pending")
    
    st.markdown(
        f"""
        <div class="header-container">
            <div class="header-title-group">
                <h1>🏛️ Ward Pulse</h1>
                <p>Intelligent Civic Demand Prediction • Municipal Intelligence System</p>
            </div>
            <div class="header-meta">
                <span class="badge badge-location">
                    📍 Lucknow Municipal Corp. / Civic Grid
                </span>
                <span class="badge {badge_class}">
                    <span class="badge-pill-pulse"></span>
                    {badge_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_status_banner(data_status: Dict[str, Any]):
    """
    Renders an alert banner explaining backend connection state.
    """
    if data_status.get("is_official", False):
        st.markdown(
            """
            <div class="status-banner" style="border-left: 4px solid #10B981; background: #ECFDF5;">
                <div class="status-banner-content">
                    <div class="status-banner-icon" style="color: #10B981;">✅</div>
                    <div class="status-banner-text" style="color: #065F46;">
                        <strong>Live ML Pipeline Connected:</strong> Actively streaming official spatio-temporal predictions, 
                        CPI priority scores, and GeoJSON ward boundary overlays from the machine learning pipeline.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="status-banner">
                <div class="status-banner-content">
                    <div class="status-banner-icon">⚡</div>
                    <div class="status-banner-text">
                        <strong>Phase 1 UI Mode:</strong> Displaying modular UI wireframes and contract schema preview. 
                        Predictions load automatically once <code>data/predictions/ward_predictions.csv</code> is generated.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_welcome_card():
    """
    Renders the civic-tech welcome / hero card.
    """
    st.markdown(
        """
        <div class="welcome-card">
            <h2>Civic Intelligence & Zero-History Ward Forecasting</h2>
            <p>
                Ward Pulse empowers municipal governance to forecast ward-level civic grievance demand 
                across solid waste management, road maintenance, water supply, and streetlighting. 
                Using spatial transfer learning and urban density features, our system infers complaint risk even in wards with zero prior historical records.
            </p>
            <div class="welcome-highlights">
                <div class="welcome-chip">🎯 Proactive Resource Allocation</div>
                <div class="welcome-chip">🏙️ 40 Target Wards Monitored</div>
                <div class="welcome-chip">🔍 Zero-History Spatio-Temporal Inference</div>
                <div class="welcome-chip">⚡ Live ML Pipeline Ingestion</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi_cards(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Renders the modular 4-card KPI summary grid.
    """
    kpis = get_kpi_metrics(df, df_summary)

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card primary-card">
                <div class="kpi-header">
                    <span class="kpi-title">Monthly Demand</span>
                    <span class="kpi-icon">📋</span>
                </div>
                <div class="kpi-value">{kpis['total_predicted_complaints']}</div>
                <div class="kpi-subtitle">
                    <span>Forecasted monthly complaints</span>
                </div>
            </div>

            <div class="kpi-card accent-card">
                <div class="kpi-header">
                    <span class="kpi-title">Peak Demand Ward</span>
                    <span class="kpi-icon">🔥</span>
                </div>
                <div class="kpi-value" style="font-size: 1.35rem;">{kpis['highest_demand_ward']}</div>
                <div class="kpi-subtitle">
                    <span>Top resource allocation priority</span>
                </div>
            </div>

            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Critical Wards (CPI &ge; 0.70)</span>
                    <span class="kpi-icon">🚨</span>
                </div>
                <div class="kpi-value" style="color: #EF4444;">{kpis['critical_wards']} Wards</div>
                <div class="kpi-subtitle">
                    <span>Immediate civic intervention required</span>
                </div>
            </div>

            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Spatial Model MAE</span>
                    <span class="kpi-icon">⚙️</span>
                </div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #168C83;">{kpis['model_mae']}</div>
                <div class="kpi-subtitle">
                    <span>5-Fold Spatial GroupKFold Out-of-Fold</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_overview(df: Optional[pd.DataFrame], data_status: Dict[str, Any], df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders the primary dashboard Overview tab.
    """
    render_status_banner(data_status)
    render_welcome_card()
    render_kpi_cards(df, df_summary)

    # Top split: Folium Choropleth Map and Plotly Analytics
    col_map, col_analytics = st.columns([1.1, 0.9])
    with col_map:
        render_choropleth_map(df_summary, geojson_data)
    with col_analytics:
        render_analytics_section(df_summary, df)

    # Bottom contract table preview
    render_dataset_preview(df, df_summary)


def render_analytics_view(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Renders the dedicated Ward Analytics page.
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin-bottom: 0.25rem;">📈 Ward Demand Analytics</h2>
            <p style="color: var(--text-secondary); margin: 0;">In-depth comparative analysis, category distribution, and ward ranking.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_analytics_section(df_summary, df)
    render_dataset_preview(df, df_summary)


def render_map_view(df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders the dedicated Map View page.
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin-bottom: 0.25rem;">🗺️ Geographical Ward Map</h2>
            <p style="color: var(--text-secondary); margin: 0;">Spatial distribution of civic complaint intensity across municipal boundaries.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_choropleth_map(df_summary, geojson_data)


def render_about_view():
    """
    Renders the About Project page with architecture details and phase roadmap.
    """
    metrics = load_model_metrics()
    metadata = load_model_metadata()

    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">ℹ️ About Ward Pulse</div>
                    <div class="section-card-subtitle">Civic intelligence system architecture & validation</div>
                </div>
                <span class="badge badge-location">Ward Pulse Platform</span>
            </div>
            
            <div style="font-size: 0.95rem; line-height: 1.7; color: var(--text-primary);">
                <p>
                    <strong>Ward Pulse</strong> addresses municipal grievance bottlenecks. 
                    In large urban centers, civic bodies often operate reactively, deploying teams only after multiple complaints 
                    are lodged through citizen portals.
                </p>
                <p>
                    Crucially, <em>underserved or newly incorporated wards often register zero historical complaints</em> due to 
                    digital divide or missing logging infrastructure. Ward Pulse solves this by learning an inductive mapping from 
                    static urban infrastructure, demographics, and spatial proximity to predict grievance intensity across 6 civic categories.
                </p>
            </div>
            
            <div style="margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid var(--border-color);">
                <h4 style="color: var(--primary); margin-bottom: 0.75rem;">⚖️ Model Governance & Validation Metrics</h4>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Performance")
        st.json(metrics)
    with col2:
        st.subheader("Top Predictive Features")
        if metadata:
            st.json(metadata.get("top_feature_importances", {}))
        else:
            st.info("Feature importance metadata loaded from models/model_metadata.json.")
