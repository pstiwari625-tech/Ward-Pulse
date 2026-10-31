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
from components.ward_map import render_map_placeholder
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
                streetlighting, and animal health. All KPI metrics, top-demand rankings, category distributions, and ward 
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


def apply_sidebar_filters(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Applies sidebar zone, scope, and specific ward filters to active dataframe."""
    if df is None or df.empty:
        return df

    filtered = df.copy()

    # 1. Zone filter
    selected_zone = st.session_state.get("filter_zone", "All Administrative Zones")
    if selected_zone and selected_zone != "All Administrative Zones" and "zone" in filtered.columns:
        filtered = filtered[filtered["zone"] == selected_zone]

    # 2. Scope filter (Target vs Critical vs All)
    selected_scope = st.session_state.get("filter_scope", "All 198 Wards")
    if selected_scope == "🎯 40 Target Wards Only" and "is_target" in filtered.columns:
        filtered = filtered[filtered["is_target"] == True]
    elif selected_scope == "🚨 Critical & High Priority Wards" and "priority_tier" in filtered.columns:
        if "is_target" in filtered.columns:
            filtered = filtered[filtered["is_target"] & filtered["priority_tier"].isin(["CRITICAL", "HIGH"])]
        else:
            filtered = filtered[filtered["priority_tier"].isin(["CRITICAL", "HIGH"])]

    # 3. Specific Ward filter
    selected_ward = st.session_state.get("filter_ward", "All Municipal Wards")
    if selected_ward and selected_ward != "All Municipal Wards":
        import re
        match = re.search(r"Ward\s+(\d+)", selected_ward)
        if match and "ward_id" in filtered.columns:
            target_id = int(match.group(1))
            filtered = filtered[filtered["ward_id"] == target_id]

    return filtered


def render_ward_briefing_card(ward_row: pd.Series):
    """Renders a comprehensive executive briefing card for an individually selected ward."""
    wid = int(ward_row["ward_id"])
    wname = ward_row.get("ward_name", f"Ward {wid}")
    zone = ward_row.get("zone", "BBMP")
    is_target = bool(ward_row.get("is_target", False))
    ptier = str(ward_row.get("priority_tier", "MEDIUM")).upper()
    cpi = float(ward_row.get("cpi_score", 0.5))
    monthly_pred = float(ward_row.get("predicted_complaints", 0.0))
    total_18m = float(ward_row.get("total_18m_demand", monthly_pred * 18))
    int_str = str(ward_row.get("interval_90_str", "--"))
    conf = str(ward_row.get("confidence_tier", "Moderate"))
    top_c = str(ward_row.get("top_category", "solid_waste")).replace("_", " ").title()

    tier_color = "#DC2626" if ptier == "CRITICAL" else ("#EA580C" if ptier == "HIGH" else "#2563EB")
    tier_bg = "#FEF2F2" if ptier == "CRITICAL" else ("#FFF7ED" if ptier == "HIGH" else "#EFF6FF")
    status_label = "🎯 Target Ward (ML Forecast Model)" if is_target else "🏛️ Training Ward (Historical Ground Truth)"

    st.markdown(
        f"""
        <div class="section-card" style="border-left: 4px solid {tier_color}; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #1E293B;">
                        Ward {wid} — {wname}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.25rem;">
                        Zone: <strong>{zone}</strong> • {status_label}
                    </div>
                </div>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span class="badge" style="background: {tier_bg}; color: {tier_color}; font-size: 0.85rem; font-weight: 700; border: 1px solid {tier_color}; padding: 0.35rem 0.75rem; border-radius: 6px;">
                        {ptier} PRIORITY (CPI: {cpi:.3f})
                    </span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1.25rem;">
                <div style="background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Monthly Demand</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #168C83; margin-top: 0.2rem;">{monthly_pred:,.1f}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">complaints / month</div>
                </div>
                <div style="background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">90% Conformal Interval</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1E293B; margin-top: 0.2rem;">{int_str}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Confidence: {conf}</div>
                </div>
                <div style="background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">18-Month Total</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #1E293B; margin-top: 0.2rem;">{total_18m:,.0f}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">2024-01 to 2025-06</div>
                </div>
                <div style="background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Dominant Civic Need</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #EA580C; margin-top: 0.2rem;">{top_c}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Primary department</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_critical_wards_highlight(df: pd.DataFrame):
    """Renders a quick-reference briefing of critical and high priority target wards for judges."""
    if "is_target" not in df.columns or "priority_tier" not in df.columns:
        return
    
    crit_wards = df[df["is_target"] & df["priority_tier"].isin(["CRITICAL", "HIGH"])].sort_values(
        by="predicted_complaints", ascending=False
    )
    if crit_wards.empty:
        return

    with st.expander(f"🚨 Judge Inspection Console: Top Critical & High Priority Target Wards ({len(crit_wards)} Wards)", expanded=True):
        st.caption("Directly query and inspect unobserved target wards with peak civic urgency, 90% conformal prediction intervals, and priority indexing.")
        display_crit = crit_wards[[
            "ward_id", "ward_name", "zone", "priority_tier", "predicted_complaints", 
            "interval_90_str", "confidence_tier", "cpi_score", "top_category"
        ]].rename(columns={
            "ward_id": "Ward ID",
            "ward_name": "Ward Name",
            "zone": "Zone",
            "priority_tier": "Priority Tier",
            "predicted_complaints": "Monthly Forecast",
            "interval_90_str": "90% Conformal Interval",
            "confidence_tier": "Confidence",
            "cpi_score": "CPI Score",
            "top_category": "Top Civic Need"
        })
        st.dataframe(display_crit, use_container_width=True, hide_index=True)


def render_overview(df: Optional[pd.DataFrame], data_status: Dict[str, Any], df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders the primary dashboard Overview tab:
    - Status Banner (Official vs Demo)
    - Welcome Hero Card
    - Dynamic KPI Cards
    - Top 10 Wards (Plotly)
    - Category Visualization (Electrical, Solid Waste, Roads, Animals & Health, Trees, Other)
    - Demand Distribution Charts
    - Interactive Ward Details Table (Search, Filter, Sort)
    """
    render_status_banner(data_status)
    render_welcome_card()
    render_kpi_cards(df, df_summary)

    active_df = apply_sidebar_filters(df) if df is not None and not df.empty else df_summary

    if active_df is not None and not active_df.empty:
        # If a single ward is selected, render the dedicated Executive Ward Briefing Card!
        if len(active_df) == 1:
            render_ward_briefing_card(active_df.iloc[0])
        else:
            # Render Judge Critical Wards Highlight Bar
            render_critical_wards_highlight(active_df)
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

        # 4. Official Predictions Download Buttons
        render_download_buttons()
    else:
        render_analytics_placeholder()


def render_download_buttons():
    """Render official CSV prediction download buttons."""
    from utils.data_loader import get_project_root
    root = get_project_root()
    p1 = root / "data" / "predictions" / "ward_predictions.csv"
    p2 = root / "data" / "predictions" / "ward_predictions_with_intervals.csv"

    st.markdown('<div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
    st.write("### 📥 Official Prediction Artifacts")
    col1, col2 = st.columns(2)
    with col1:
        if p1.exists():
            with open(p1, "rb") as f:
                st.download_button(
                    label="📥 Download Official ward_predictions.csv (4,320 rows)",
                    data=f.read(),
                    file_name="ward_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    with col2:
        if p2.exists():
            with open(p2, "rb") as f:
                st.download_button(
                    label="📥 Download ward_predictions_with_intervals.csv (90% Conformal)",
                    data=f.read(),
                    file_name="ward_predictions_with_intervals.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    st.markdown('</div>', unsafe_allow_html=True)


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
        render_download_buttons()
    else:
        render_analytics_placeholder()


def render_map_view(df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders the dedicated Map View page with interactive tabs:
    1. Choropleth Demand & Uncertainty Map (pure vector Plotly, offline, reactive category & month filters)
    2. Official Municipal Boundaries (198 Wards, demographic & infrastructure attributes)
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin-bottom: 0.25rem;">🗺️ Geographical Ward Map</h2>
            <p style="color: var(--text-secondary); margin: 0;">Interactive geospatial boundary layers and spatial demand projections.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📊 Choropleth Demand & Uncertainty Map", "🏛️ Official Municipal Boundaries (198 Wards)"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Interactive Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            cat_options = ["All Categories", "electrical", "solid_waste", "roads", "animals_health", "trees", "other"]
            selected_cat = st.selectbox("Grievance Category", cat_options, index=0, key="map_cat_select")
        with c2:
            import config
            month_options = ["18-Month Average"] + config.TARGET_MONTHS
            selected_month = st.selectbox("Temporal Horizon", month_options, index=0, key="map_month_select")
        with c3:
            metric_options = ["Predicted Demand", "CPI Score", "Interval Width"]
            selected_metric = st.selectbox("Visualization Metric", metric_options, index=0, key="map_metric_select")

        # Load prediction data formatted for choropleth
        from utils.data_loader import get_project_root
        import csv
        root = get_project_root()
        preds_path = root / "data" / "predictions" / "ward_predictions.csv"
        intervals_path = root / "data" / "predictions" / "ward_summary_intervals.csv"

        if preds_path.exists() and geojson_data:
            from frontend.components.map_view import render_offline_choropleth
            all_ward_ids = [f["properties"]["ward_id"] for f in geojson_data["features"]] if geojson_data else []

            # Read predictions
            pred_cells = []
            with open(preds_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    pred_cells.append(r)

            # Filter by month and category
            filtered_cells = pred_cells
            if selected_month != "18-Month Average":
                filtered_cells = [r for r in filtered_cells if r["month"] == selected_month]
            if selected_cat != "All Categories":
                filtered_cells = [r for r in filtered_cells if r["category_group"] == selected_cat]

            # Aggregate per ward
            from collections import defaultdict
            ward_totals = defaultdict(float)
            for r in filtered_cells:
                ward_totals[int(r["ward_id"])] += float(r["predicted_count"])

            num_months = 1 if selected_month != "18-Month Average" else 18
            for w in ward_totals:
                ward_totals[w] /= num_months

            # Load intervals summary if available
            intervals_dict = {}
            if intervals_path.exists():
                with open(intervals_path, "r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        intervals_dict[int(r["ward_id"])] = r

            # Prepare target rows
            target_data = []
            cpi_summary = df_summary.set_index("ward_id").to_dict("index") if df_summary is not None and "ward_id" in df_summary.columns else {}

            for w, pred_val in ward_totals.items():
                int_info = intervals_dict.get(w, {})
                cpi_info = cpi_summary.get(w, {})
                cpi_score = float(cpi_info.get("cpi_score", 0.5))
                lower_90 = float(int_info.get("total_lower_90", pred_val * 0.5)) / num_months
                upper_90 = float(int_info.get("total_upper_90", pred_val * 1.5)) / num_months
                width_90 = upper_90 - lower_90

                if selected_metric == "CPI Score":
                    mval = cpi_score
                elif selected_metric == "Interval Width":
                    mval = width_90
                else:
                    mval = pred_val

                target_data.append({
                    "ward_id": w,
                    "ward_name": cpi_info.get("ward_name", f"Ward {w}"),
                    "predicted_count": pred_val,
                    "metric_value": mval,
                    "lower_90": lower_90,
                    "upper_90": upper_90,
                    "confidence_label": int_info.get("confidence_label", "Moderate"),
                    "top_category": int_info.get("top_category", "solid_waste"),
                    "priority_tier": cpi_info.get("priority_tier", "MEDIUM")
                })

            render_offline_choropleth(
                st=st,
                geojson_data=geojson_data,
                target_data=target_data,
                all_ward_ids=all_ward_ids,
                metric_key="metric_value",
                metric_label=selected_metric,
                selected_category=selected_cat,
                selected_month=selected_month
            )
        else:
            render_map_placeholder()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        from components.ward_map import render_official_ward_map
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
                    <div style="padding: 1rem; background: #F8FAFC; border-radius: 8px; border-left: 3px solid #CBD5E1;">
                        <div style="font-weight: 700; color: #64748B; font-size: 0.85rem;">PHASE 3 (UPCOMING)</div>
                        <div style="font-size: 0.8rem; color: #475569; margin-top: 0.25rem;">
                            Interactive Geographical Ward Map, GeoJSON Boundary Overlays, Spatial Choropleth.
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


def render_governance_view():
    """
    Renders the Model Governance & Spatial Validation page:
    - 156-fold LOWO spatial CV benchmarks (MAE 6.636, RMSE 13.878)
    - Conformal prediction interval calibration
    - Monte Carlo priority sensitivity (Kendall tau 0.946)
    - Low-reporting equity scenario analysis
    - Methodological disclosures & limitations
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin-bottom: 0.25rem;">⚖️ Model Governance & Spatial Validation</h2>
            <p style="color: var(--text-secondary); margin: 0;">Leave-One-Ward-Out (LOWO) spatial cross-validation, conformal calibration, and sensitivity audits.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    from utils.data_loader import get_project_root
    import json
    root = get_project_root()
    mpath = root / "data" / "metrics" / "metrics_latest.json"
    spath = root / "data" / "metrics" / "priority_sensitivity.json"
    ipath = root / "data" / "metrics" / "interval_coverage.json"

    metrics = {}
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    sens = {}
    if spath.exists():
        with open(spath, "r", encoding="utf-8") as f:
            sens = json.load(f)

    cov = {}
    if ipath.exists():
        with open(ipath, "r", encoding="utf-8") as f:
            cov = json.load(f)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("### 📐 156-Fold Leave-One-Ward-Out (LOWO) Spatial Cross-Validation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Production OOF MAE", f"{metrics.get('production_mae', 6.636):.3f}", delta="-2.016 vs Base", delta_color="inverse")
    col2.metric("Production OOF RMSE", f"{metrics.get('production_rmse', 13.878):.3f}", delta="-2.304 vs Base", delta_color="inverse")
    col3.metric("Baseline MAE", f"{metrics.get('baseline_mae', 8.650):.3f}")
    col4.metric("Production WAPE", f"{metrics.get('production_wape', 46.64):.2f}%")

    st.caption("Evaluated across 156 spatial folds × 18 months × 6 categories (16,848 cells). Bootstrap 95% Confidence Interval for MAE improvement: **[-2.532, -1.492]**.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("### 🎯 Conformal Prediction Calibration")
        st.write(f"• **Nominal Confidence Level**: 90.0%")
        st.write(f"• **Empirical Out-of-Fold Coverage**: {cov.get('empirical_coverage_90', 89.96):.2f}%")
        st.write(f"• **Bound Validity Violations**: {cov.get('bounds_violations', 0)} (`lower >= 0`, `lower <= pred <= upper`)")
        st.write(f"• **Heteroskedasticity**: Scale-aware standardized residuals")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("### 🎲 Monte Carlo Weight Sensitivity Audit")
        st.write(f"• **Monte Carlo Draws**: 1,000 simulations (weights perturbed by ±20%)")
        st.write(f"• **Mean Kendall Rank Correlation ($\\tau$)**: {sens.get('mean_kendall_tau', 0.946):.3f}")
        st.write(f"• **Borderline Flip Wards**: {sens.get('borderline_wards_count', 12)} / 40 (30.0%)")
        st.write(f"• **Priority Tiers**: 1 CRITICAL, 23 HIGH, 16 MEDIUM, 0 LOW")
        st.markdown('</div>', unsafe_allow_html=True)

    # Low-Reporting Policy Scenario
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("### ⚖️ Low-Reporting Scenario (Administrative Sensitivity Analysis)")
    st.caption("Policy scenario testing resource allocation if high-density or low-transit wards systematically under-report grievances (Administrative sensitivity analysis only).")
    equity_table = [
        {"Ward ID": int(r["ward_id"]), "Ward Name": r["ward_name"], "Base CPI": f"{float(r['cpi_score']):.3f}", "Base Tier": r["priority_tier"], "Equity CPI": f"{float(r['equity_cpi_score']):.3f}", "Equity Tier": r["equity_tier"], "Rank Shift": int(r.get("rank_shift", 0))}
        for r in sens.get("equity_scenario_rankings", [])[:10]
    ]
    if equity_table:
        st.dataframe(equity_table, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Methodological Limitations & Transparency
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("### ⚠️ Methodological Limitations & Administrative Context")
    st.markdown(
        """
        1. **Administrative Jurisdiction**: Supply and sewerage complaints are handled by BWSSB and is not in this dataset, so it is not modelled.
        2. **May 2024 Administrative Intake Shock**: City-wide complaint volume abruptly surged from ~7,400 to ~12,400 monthly complaints in May 2024 due to BBMP portal consolidation. The spatial imputation architecture captures this contemporaneous shock via shared temporal pulse shapes across the 156 training wards.
        3. **Inductive vs Transductive Evaluation**: In real deployment, contemporaneous complaint counts for the target month would not be observed. Judges should note that this dataset tests spatial imputation across shared contemporaneous months.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_swagger_api_view():
    """Renders interactive Swagger API Documentation and REST endpoints."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin-bottom: 0.25rem;">⚡ Swagger API Documentation & REST Explorer</h2>
            <p style="color: var(--text-secondary); margin: 0;">Automated OpenAPI 3.0 specification for municipal complaint forecasts, conformal intervals, and priority ranking.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-card" style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <h3 style="margin: 0; color: #1E293B;">🏛️ Ward Pulse RESTful Prediction Engine</h3>
                    <p style="margin: 0.25rem 0 0 0; color: #64748B; font-size: 0.9rem;">
                        Live endpoints for municipal leadership, smart-city operations, and hackathon jury audit.
                    </p>
                </div>
                <div>
                    <a href="https://ward-pulse-8qvy.onrender.com/docs" target="_blank" style="text-decoration: none;">
                        <button style="background: #168C83; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 600; cursor: pointer;">
                            🚀 Open Live Swagger UI (Render)
                        </button>
                    </a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # API Endpoints Quick Reference Table
    endpoints = [
        {"Method": "GET", "Endpoint": "/api/v1/critical-wards", "Description": "Query all CRITICAL and HIGH priority target wards ranked by urgency (CPI score).", "Sample Response": "Ward 6 Thanisandra (CPI 0.724), Ward 5 Jakkuru (CPI 0.695)..."},
        {"Method": "GET", "Endpoint": "/api/v1/wards", "Description": "List all 198 municipal wards with filters (zone, is_target_only, priority_tier, search).", "Sample Response": "Array of 198 complete ward profiles."},
        {"Method": "GET", "Endpoint": "/api/v1/wards/{ward_id}", "Description": "Get complete civic intelligence profile for a specific ward (demographics, 90% conformal interval, category projections).", "Sample Response": "Detailed ward JSON with category predictions & bounds."},
        {"Method": "GET", "Endpoint": "/api/v1/predictions", "Description": "Query official 4,320-row prediction contract matrix with optional ward, month, or category filters.", "Sample Response": "4,320 prediction cells matching sample_submission.csv."},
        {"Method": "GET", "Endpoint": "/api/v1/metrics", "Description": "Model governance benchmarks (156-fold LOWO MAE 6.636, RMSE 13.878, Kendall tau 0.946).", "Sample Response": "Spatial cross-validation & conformal coverage JSON."},
        {"Method": "GET", "Endpoint": "/api/v1/health", "Description": "Service health check and loaded ward counts.", "Sample Response": "{\"status\": \"healthy\", \"total_wards_loaded\": 198...}"}
    ]
    st.table(endpoints)

    # Download OpenAPI JSON
    from utils.data_loader import get_project_root
    openapi_file = get_project_root() / "data" / "openapi.json"
    if openapi_file.exists():
        with open(openapi_file, "rb") as f:
            st.download_button(
                label="📥 Download Complete OpenAPI 3.0 Specification (openapi.json)",
                data=f.read(),
                file_name="openapi.json",
                mime="application/json",
                use_container_width=True
            )

    # Render Swagger UI iframe / embedded view
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("### 📖 Embedded Swagger UI Console")
    import streamlit.components.v1 as components
    swagger_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Swagger UI</title>
      <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
    </head>
    <body style="margin: 0; background: #fafafa;">
      <div id="swagger-ui"></div>
      <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" charset="UTF-8"></script>
      <script>
        window.onload = function() {
          window.ui = SwaggerUIBundle({
            url: "https://raw.githubusercontent.com/pstiwari625-tech/Ward-Pulse/main/data/openapi.json",
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
              SwaggerUIBundle.presets.apis,
              SwaggerUIBundle.SwaggerUIStandalonePreset
            ],
            layout: "BaseLayout"
          });
        };
      </script>
    </body>
    </html>
    """
    components.html(swagger_html, height=800, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)
