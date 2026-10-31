"""
Ward Pulse - Charts & Analytics Component
Phase 2: Dashboard and Analytics Implementation
Provides Plotly visualizations for Top 10 wards, category breakdowns,
demand distribution, and interactive ward details.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go


def render_analytics_placeholder():
    """
    Renders the civic analytics section placeholder when datasets are missing or pending.
    """
    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">
                        <span>📈</span> Ward-Wise Civic Analytics
                    </div>
                    <div class="section-card-subtitle">
                        Comparative demand intelligence, complaint category distributions, and ward ranking
                    </div>
                </div>
                <span class="placeholder-badge">Awaiting Data</span>
            </div>
            <div class="placeholder-box" style="min-height: 220px;">
                <div class="placeholder-icon">📊</div>
                <div class="placeholder-title">Civic Complaint Analytics Pending</div>
                <div class="placeholder-desc">
                    Analytics visualizations will render automatically once predictions are loaded.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def plot_top_10_wards(df: pd.DataFrame):
    """
    Feature B: Top 10 Wards with the highest predicted complaints.
    Sorted in descending order using Plotly.
    """
    if df is None or df.empty or "predicted_complaints" not in df.columns:
        st.info("No data available to display Top 10 wards.")
        return

    # Extract top 10 wards sorted descending
    top_10 = df.sort_values(by="predicted_complaints", ascending=False).head(10).copy()

    # For horizontal bar, Plotly renders bottom-to-top, so sort ascending for display
    plot_df = top_10.sort_values(by="predicted_complaints", ascending=True).copy()

    # Create formatted labels
    name_col = "ward_name" if "ward_name" in plot_df.columns else "ward_id"
    plot_df["display_label"] = plot_df.apply(
        lambda r: f"{r[name_col]} (Ward {r['ward_id']})", axis=1
    )
    level_col = "demand_level" if "demand_level" in plot_df.columns else "demand_tier"
    plot_df["level_val"] = plot_df[level_col] if level_col in plot_df.columns else "Standard"

    # Color mapping for demand level
    level_color_map = {
        "Critical": "#EF4444",
        "High": "#F59E0B",
        "Medium": "#3B82F6",
        "Low": "#10B981"
    }

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["predicted_complaints"],
            y=plot_df["display_label"],
            orientation="h",
            marker=dict(
                color=[level_color_map.get(lvl, "#168C83") for lvl in plot_df["level_val"]],
                line=dict(color="#142B45", width=1)
            ),
            text=[f"{v:,.1f}" for v in plot_df["predicted_complaints"]],
            textposition="auto",
            customdata=plot_df[[name_col, "ward_id", "level_val"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (Ward %{customdata[1]})<br>"
                "Predicted Complaints: <b>%{x:,.1f}</b><br>"
                "Demand Level: <b>%{customdata[2]}</b><extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(
            text="🏆 Top 10 Wards by Predicted Complaint Volume",
            font=dict(size=14, color="#142B45", family="Segoe UI, sans-serif")
        ),
        xaxis_title="Predicted Monthly Complaints",
        yaxis_title="",
        margin=dict(l=10, r=20, t=45, b=25),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=11, color="#263445"),
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_category_visualization(df: pd.DataFrame):
    """
    Feature C: Complaint Category Visualization (Electrical, Solid Waste, Roads, Animals & Health, Trees, Other).
    Only displays category predictions when category-level data exists.
    Handles missing category columns gracefully.
    """
    if df is None or df.empty:
        st.info("No data available for category analysis.")
        return

    # Check for target category columns (canonical 6 categories)
    target_cats = {
        "electrical": "Electrical & Streetlights",
        "solid_waste": "Solid Waste & Garbage",
        "roads": "Roads & Pavements",
        "animals_health": "Animals & Health",
        "trees": "Urban Trees & Forestry",
        "other": "Other Civic Complaints"
    }

    # Find available category columns with valid numeric data
    available_cats = {}
    for col_key, display_name in target_cats.items():
        if col_key in df.columns:
            series = pd.to_numeric(df[col_key], errors="coerce").fillna(0.0)
            if series.sum() > 0:
                available_cats[col_key] = display_name

    # Handle missing categories gracefully
    if not available_cats:
        st.markdown(
            """
            <div class="placeholder-box" style="min-height: 280px; padding: 2rem;">
                <div class="placeholder-icon">🏷️</div>
                <div class="placeholder-title">Category-Level Predictions Pending</div>
                <div class="placeholder-desc">
                    Department-level columns (<strong>Electrical, Solid Waste, Roads, Animals & Health, Trees, Other</strong>) are not present in the loaded dataset schema.
                    Category projections will render automatically once department columns are provided in <code>data/predictions/ward_predictions.csv</code>.
                </div>
                <div style="margin-top: 1rem;">
                    <span class="placeholder-badge" style="background: #F1F5F9; color: #64748B;">
                        Awaiting Category Breakdowns
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Calculate category sums
    cat_data = []
    cat_colors = {
        "Electrical & Streetlights": "#EAB308",
        "Solid Waste & Garbage": "#F59E0B",
        "Roads & Pavements": "#168C83",
        "Animals & Health": "#EF4444",
        "Urban Trees & Forestry": "#10B981",
        "Other Civic Complaints": "#64748B"
    }

    for col_key, display_name in available_cats.items():
        total_val = float(pd.to_numeric(df[col_key], errors="coerce").fillna(0.0).sum())
        cat_data.append({
            "Category": display_name,
            "Complaints": total_val,
            "Color": cat_colors.get(display_name, "#142B45")
        })

    cat_df = pd.DataFrame(cat_data)
    total_cat_sum = cat_df["Complaints"].sum()

    # Donut chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=cat_df["Category"],
                values=cat_df["Complaints"],
                hole=0.48,
                marker=dict(colors=[c["Color"] for c in cat_data], line=dict(color="#FFFFFF", width=2)),
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Forecast: <b>%{value:,.1f}</b> complaints<br>Share: <b>%{percent}</b><extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="🏷️ Civic Category Demand Breakdown",
            font=dict(size=14, color="#142B45", family="Segoe UI, sans-serif")
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=11, color="#263445"),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{int(round(total_cat_sum)):,}</b><br><span style='font-size:10px; color:#64748B;'>Category Total</span>",
                x=0.5,
                y=0.5,
                font_size=13,
                showarrow=False
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Note if any expected categories were missing
    missing_cats = [v for k, v in target_cats.items() if k not in available_cats]
    if missing_cats:
        st.caption(f"ℹ️ Displaying {len(available_cats)} available categories. (Pending: {', '.join(missing_cats)})")


def plot_demand_distribution(df: pd.DataFrame):
    """
    Feature D: Demand Distribution.
    Creates meaningful Plotly charts to display complaint demand across wards:
    1. Demand Frequency Histogram with city mean reference line.
    2. Demand Level Risk Stratification (Proportion of Wards per Demand Tier).
    """
    if df is None or df.empty or "predicted_complaints" not in df.columns:
        st.info("No data available to display demand distribution.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Histogram of complaint demand across wards
        mean_val = float(df["predicted_complaints"].mean())
        median_val = float(df["predicted_complaints"].median())

        fig_hist = px.histogram(
            df,
            x="predicted_complaints",
            nbins=12,
            title="📊 Complaint Demand Distribution Across Wards",
            labels={"predicted_complaints": "Predicted Monthly Complaints"},
            color_discrete_sequence=["#142B45"]
        )

        # Add vertical line for city mean
        fig_hist.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Mean: {mean_val:.1f}",
            annotation_position="top right"
        )

        fig_hist.update_layout(
            yaxis_title="Number of Wards",
            margin=dict(l=10, r=10, t=45, b=25),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif", size=11, color="#263445"),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Demand Level Distribution Bar Chart
        level_col = "demand_level" if "demand_level" in df.columns else "demand_tier"
        if level_col in df.columns:
            counts = df[level_col].value_counts().reset_index()
            counts.columns = ["Demand Level", "Ward Count"]

            level_colors = {
                "Critical": "#EF4444",
                "High": "#F59E0B",
                "Medium": "#3B82F6",
                "Low": "#10B981"
            }

            fig_level = px.bar(
                counts,
                x="Demand Level",
                y="Ward Count",
                color="Demand Level",
                color_discrete_map=level_colors,
                title="🎯 Wards by Demand Priority Level",
                text="Ward Count"
            )

            fig_level.update_layout(
                margin=dict(l=10, r=10, t=45, b=25),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(family="sans-serif", size=11, color="#263445"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0")
            )
            st.plotly_chart(fig_level, use_container_width=True)
        else:
            st.info("Demand level column not found in dataset.")
        st.markdown('</div>', unsafe_allow_html=True)


def render_ward_details_section(df: pd.DataFrame):
    """
    Feature E: Ward Details.
    Displays ward ID, ward name, predicted complaints, and demand level.
    Adds useful sorting and filtering.
    """
    if df is None or df.empty:
        st.info("No ward detail records to display.")
        return

    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">
                        <span>📋</span> Ward-Level Details & Demand Intelligence
                    </div>
                    <div class="section-card-subtitle">
                        Filterable and sortable registry of all monitored municipal wards
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # Filter & Sort Controls
    col_search, col_filter, col_sort = st.columns([1.5, 1.2, 1.3])

    with col_search:
        search_query = st.text_input(
            "🔍 Search Ward Name or ID",
            placeholder="Type name (e.g. Hazratganj) or ID...",
            key="ward_details_search"
        )

    # Demand Level filter
    level_col = "demand_level" if "demand_level" in df.columns else "demand_tier"
    available_levels = sorted(list(df[level_col].unique())) if level_col in df.columns else []

    with col_filter:
        selected_levels = st.multiselect(
            "🎯 Filter by Demand Level",
            options=available_levels,
            default=available_levels,
            key="ward_details_level_filter"
        )

    with col_sort:
        sort_choice = st.selectbox(
            "↕️ Sort Wards By",
            options=[
                "Predicted Complaints (Highest First)",
                "Predicted Complaints (Lowest First)",
                "Ward Name (A to Z)",
                "Ward ID (Ascending)"
            ],
            index=0,
            key="ward_details_sort"
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_levels and level_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[level_col].isin(selected_levels)]

    if search_query.strip():
        q = search_query.strip().lower()
        name_col = "ward_name" if "ward_name" in filtered_df.columns else "ward_id"
        filtered_df = filtered_df[
            filtered_df[name_col].astype(str).str.lower().str.contains(q) |
            filtered_df["ward_id"].astype(str).str.contains(q)
        ]

    # Apply sorting
    if sort_choice == "Predicted Complaints (Highest First)":
        filtered_df = filtered_df.sort_values(by="predicted_complaints", ascending=False)
    elif sort_choice == "Predicted Complaints (Lowest First)":
        filtered_df = filtered_df.sort_values(by="predicted_complaints", ascending=True)
    elif sort_choice == "Ward Name (A to Z)":
        name_col = "ward_name" if "ward_name" in filtered_df.columns else "ward_id"
        filtered_df = filtered_df.sort_values(by=name_col, ascending=True)
    elif sort_choice == "Ward ID (Ascending)":
        filtered_df = filtered_df.sort_values(by="ward_id", ascending=True)

    # Determine display columns
    core_cols = ["ward_id", "ward_name", "predicted_complaints", "demand_level"]
    
    # Include 90% confidence intervals, confidence labels, and rank stability if present
    interval_cols = []
    if "total_lower_90" in filtered_df.columns and "total_upper_90" in filtered_df.columns:
        filtered_df["90% Interval"] = filtered_df.apply(
            lambda r: f"[{float(r['total_lower_90']):.1f}, {float(r['total_upper_90']):.1f}]", axis=1
        )
        interval_cols.append("90% Interval")
    if "confidence_label" in filtered_df.columns:
        filtered_df["Confidence"] = filtered_df["confidence_label"].astype(str)
        interval_cols.append("Confidence")
    if "rank_flip_prob" in filtered_df.columns:
        filtered_df["Rank Stability"] = filtered_df["rank_flip_prob"].apply(
            lambda v: f"{float(v):.2f}" if pd.notnull(v) else "--"
        )
        interval_cols.append("Rank Stability")

    category_cols = [c for c in ["electrical", "solid_waste", "roads", "animals_health", "trees", "other"] if c in filtered_df.columns]
    
    # Keep only available relevant columns
    cols_to_show = [c for c in core_cols if c in filtered_df.columns] + interval_cols + category_cols
    display_df = filtered_df[cols_to_show].copy()

    # Rename for professional table headers
    rename_dict = {
        "ward_id": "Ward ID",
        "ward_name": "Ward Name",
        "predicted_complaints": "Predicted Complaints",
        "demand_level": "Demand Level",
        "electrical": "Electrical",
        "solid_waste": "Solid Waste",
        "roads": "Roads",
        "animals_health": "Animals & Health",
        "trees": "Trees",
        "other": "Other"
    }
    display_df = display_df.rename(columns=rename_dict)

    # Format numeric values
    if "Predicted Complaints" in display_df.columns:
        display_df["Predicted Complaints"] = display_df["Predicted Complaints"].round(1)
    for cat in ["Electrical", "Solid Waste", "Roads", "Animals & Health", "Trees", "Other"]:
        if cat in display_df.columns:
            display_df[cat] = display_df[cat].round(1)

    # Record metrics
    st.caption(f"Showing **{len(display_df)}** of **{len(df)}** wards • Total Predicted Complaints in View: **{display_df['Predicted Complaints'].sum():,.1f}**" if "Predicted Complaints" in display_df.columns else f"Showing {len(display_df)} wards")

    # Render interactive DataFrame
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Export button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Ward Predictions (CSV)",
        data=csv_bytes,
        file_name="ward_pulse_predictions_filtered.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_analytics_section(df: Optional[pd.DataFrame] = None, df_summary: Optional[pd.DataFrame] = None):
    """
    Renders the unified Phase 2 Analytics section:
    - Top 10 wards (Plotly)
    - Complaint Category Visualization (Electrical, Solid Waste, Roads, Animals & Health, Trees, Other)
    - Demand Distribution (Histogram + Level Breakdown)
    - Ward Details Table (Search, Filter, Sort)
    """
    active_df = df if df is not None and not df.empty else df_summary

    if active_df is None or active_df.empty:
        render_analytics_placeholder()
        return

    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">
                        <span>📈</span> Ward-Wise Civic Demand Analytics
                    </div>
                    <div class="section-card-subtitle">
                        Top demand centers, category distributions, and predictive demand modeling
                    </div>
                </div>
                <span class="badge badge-live">Phase 2 Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Top 10 Wards & Category Breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        plot_top_10_wards(active_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        plot_category_visualization(active_df)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Demand Distribution
    plot_demand_distribution(active_df)

    # 3. Ward Details Table with sorting and filtering
    render_ward_details_section(active_df)


def render_dataset_preview(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Legacy wrapper delegating to the Phase 2 Ward Details section.
    """
    active_df = df if df is not None and not df.empty else df_summary
    render_ward_details_section(active_df)
