"""
Ward Pulse - Main Dashboard Component
Phase 2: Dashboard and Analytics Implementation
Renders header, status indicators, KPI metric cards, and primary views.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

from components.charts import (
    render_analytics_section,
    render_analytics_placeholder,
    plot_top_10_wards,
    plot_category_visualization,
    plot_demand_distribution,
    render_ward_details_section,
    render_dataset_preview
)
from components.ward_map import render_official_ward_map, render_map_placeholder
from utils.data_loader import get_kpi_metrics, load_model_metrics, load_model_metadata


def render_header(data_status: Dict[str, Any]):
    """
    Renders the persistent top header with title, location, and data status indicator.
    """
    badge_class = data_status.get("badge_class", "badge-pending")
    badge_label = data_status.get("badge_label", "Awaiting Data")
    
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
    Clearly distinguishes official ML dataset from illustrative demo data.
    """
    if data_status.get("is_official", False):
        st.markdown(
            """
            <div class="status-banner" style="border-left: 4px solid #10B981; background: #ECFDF5;">
                <div class="status-banner-content">
                    <div class="status-banner-icon" style="color: #10B981;">✅</div>
                    <div class="status-banner-text" style="color: #065F46;">
                        <strong>Official Prediction Dataset Connected:</strong> Live predictions loaded from 
                        <code>data/predictions/ward_predictions.csv</code>. KPI metrics, top wards, and analytics are dynamically rendered.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="status-banner" style="border-left: 4px solid #F59E0B; background: #FFFBEB;">
                <div class="status-banner-content">
                    <div class="status-banner-icon" style="color: #F59E0B;">⚡</div>
                    <div class="status-banner-text" style="color: #92400E;">
                        <strong>Illustrative Demo Dataset Active:</strong> The organization's actual predictions dataset has not been provided yet.
                        Displaying separate illustrative demo data for Phase 2 validation. <em>Values are illustrative and not actual ML output.</em>
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
            <h2>Phase 2: Civic Demand Intelligence & Analytics</h2>
            <p>
                Ward Pulse forecasts ward-level municipal grievance demand across solid waste management, road maintenance, 
                water supply, and streetlighting. All KPI metrics, top-demand rankings, category distributions, and ward 
                registries are dynamically computed from the ingested prediction dataset.
            </p>
            <div class="welcome-highlights">
                <div class="welcome-chip">🎯 Dynamic KPI Analytics</div>
                <div class="welcome-chip">🏆 Top 10 High-Demand Wards</div>
                <div class="welcome-chip">🏷️ Department Category Breakdown</div>
                <div class="welcome-chip">📊 Demand Distribution Modeling</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi_cards(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Feature A: KPI Cards.
    - Total wards.
    - Total predicted complaints.
    - High-demand wards.
    - Average predicted complaints per ward.
    All values are dynamically calculated from the loaded dataset.
    """
    kpis = get_kpi_metrics(df, df_summary)

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card primary-card">
                <div class="kpi-header">
                    <span class="kpi-title">Total Wards</span>
                    <span class="kpi-icon">🏙️</span>
                </div>
                <div class="kpi-value">{kpis['total_wards']}</div>
                <div class="kpi-subtitle">
                    <span>Monitored municipal wards</span>
                </div>
            </div>

            <div class="kpi-card accent-card">
                <div class="kpi-header">
                    <span class="kpi-title">Total Predicted Complaints</span>
                    <span class="kpi-icon">📋</span>
                </div>
                <div class="kpi-value">{kpis['total_predicted_complaints']}</div>
                <div class="kpi-subtitle">
                    <span>Cumulative city-wide demand</span>
                </div>
            </div>

            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">High-Demand Wards</span>
                    <span class="kpi-icon">🔥</span>
                </div>
                <div class="kpi-value" style="color: #EF4444; font-size: 1.35rem;">{kpis['high_demand_wards']}</div>
                <div class="kpi-subtitle">
                    <span>High & Critical priority wards</span>
                </div>
            </div>

            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Avg Complaints / Ward</span>
                    <span class="kpi-icon">📈</span>
                </div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #168C83;">{kpis['avg_predicted_complaints']}</div>
                <div class="kpi-subtitle">
                    <span>Mean complaints per ward</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_overview(df: Optional[pd.DataFrame], data_status: Dict[str, Any], df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders the primary dashboard Overview tab:
    - Status Banner (Official vs Demo)
    - Welcome Hero Card
    - Dynamic KPI Cards
    - Top 10 Wards (Plotly)
    - Category Visualization (Roads, Garbage, Water, Streetlights)
    - Demand Distribution Charts
    - Interactive Ward Details Table (Search, Filter, Sort)
    """
    render_status_banner(data_status)
    render_welcome_card()
    render_kpi_cards(df, df_summary)

    active_df = df if df is not None and not df.empty else df_summary

    if active_df is not None and not active_df.empty:
        # 1. Top 10 Wards & Category Breakdown side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            plot_top_10_wards(active_df)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            plot_category_visualization(active_df)
            st.markdown('</div>', unsafe_allow_html=True)

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
    active_df = df if df is not None and not df.empty else df_summary

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin-bottom: 0.25rem;">📈 Ward Demand Analytics</h2>
            <p style="color: var(--text-secondary); margin: 0;">Comprehensive comparative analysis, category projections, and ward demand stratification.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if active_df is not None and not active_df.empty:
        render_kpi_cards(active_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            plot_top_10_wards(active_df)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            plot_category_visualization(active_df)
            st.markdown('</div>', unsafe_allow_html=True)

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
    Renders the About Project page with architecture details and roadmap.
    """
    metrics = load_model_metrics()
    metadata = load_model_metadata()

    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">ℹ️ About Ward Pulse</div>
                    <div class="section-card-subtitle">Civic intelligence system architecture & roadmap</div>
                </div>
                <span class="badge badge-location">Ward Pulse Platform</span>
            </div>
            
            <div style="font-size: 0.95rem; line-height: 1.7; color: var(--text-primary);">
                <p>
                    <strong>Ward Pulse</strong> is a civic intelligence platform designed to forecast municipal complaint demand 
                    across Lucknow's municipal wards. By utilizing spatio-temporal indicators and urban density features, 
                    the system infers grievance volume even in newly incorporated wards with zero prior historical records.
                </p>
            </div>
            
            <div style="margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid var(--border-color);">
                <h4 style="color: var(--primary); margin-bottom: 0.75rem;">📋 Project Roadmap Status</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div style="padding: 1rem; background: #ECFDF5; border-radius: 8px; border-left: 3px solid #10B981;">
                        <div style="font-weight: 700; color: #065F46; font-size: 0.85rem;">PHASE 1 (COMPLETED)</div>
                        <div style="font-size: 0.8rem; color: #047857; margin-top: 0.25rem;">
                            Frontend Foundation, UI Design System, Layout Placeholders, Modular Architecture.
                        </div>
                    </div>
                    <div style="padding: 1rem; background: #ECFDF5; border-radius: 8px; border-left: 3px solid #10B981;">
                        <div style="font-weight: 700; color: #065F46; font-size: 0.85rem;">PHASE 2 (COMPLETED)</div>
                        <div style="font-size: 0.8rem; color: #047857; margin-top: 0.25rem;">
                            KPI Cards, Top 10 Wards (Plotly), Category Visualization, Demand Distribution, Ward Details Table.
                        </div>
                    </div>
                    <div style="padding: 1rem; background: #ECFDF5; border-radius: 8px; border-left: 3px solid #10B981;">
                        <div style="font-weight: 700; color: #065F46; font-size: 0.85rem;">PHASE 3 (COMPLETED)</div>
                        <div style="font-size: 0.8rem; color: #047857; margin-top: 0.25rem;">
                            Official 198 GeoJSON Ward Boundaries, Ward Features Join, Interactive Folium Map, Neutral Baseline.
                        </div>
                    </div>
                    <div style="padding: 1rem; background: #F8FAFC; border-radius: 8px; border-left: 3px solid #CBD5E1;">
                        <div style="font-weight: 700; color: #64748B; font-size: 0.85rem;">PHASE 4 (FINAL)</div>
                        <div style="font-size: 0.8rem; color: #475569; margin-top: 0.25rem;">
                            Model Governance, Report Export, Jury Pitch Presentation.
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid var(--border-color);">
                <h4 style="color: var(--primary); margin-bottom: 0.75rem;">⚖️ Model Governance & Spatial Validation</h4>
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
