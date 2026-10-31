"""
Ward Pulse - Charts & Analytics Component
Provides layout placeholders, container components, and stub functions for Phase 2 visualizations.
"""

import streamlit as st
import pandas as pd
from typing import Optional


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


def render_analytics_section(df_summary: Optional[pd.DataFrame] = None, df_predictions: Optional[pd.DataFrame] = None):
    """
    Renders live Plotly analytics charts if data is present, otherwise falls back to placeholder.
    """
    if (df_summary is None or df_summary.empty) and (df_predictions is None or df_predictions.empty):
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
                        Comparative demand intelligence, complaint category distributions, and priority rankings
                    </div>
                </div>
                <span class="badge badge-live">Live ML Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if df_summary is not None and not df_summary.empty and "total_predicted_monthly" in df_summary.columns:
            plot_ward_demand_bar(df_summary)
        elif df_predictions is not None and not df_predictions.empty:
            plot_prediction_demand_bar(df_predictions)
        else:
            st.info("No summary data available for ranking chart.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if df_summary is not None and not df_summary.empty:
            plot_category_breakdown_pie(df_summary)
        elif df_predictions is not None and not df_predictions.empty and "category_group" in df_predictions.columns:
            plot_predictions_category_pie(df_predictions)
        else:
            st.info("No category data available for distribution chart.")
        st.markdown('</div>', unsafe_allow_html=True)


def plot_ward_demand_bar(df_summary: pd.DataFrame):
    """Generates horizontal bar chart of top 15 wards by predicted monthly demand."""
    try:
        import plotly.express as px

        top_df = df_summary.sort_values(by="total_predicted_monthly", ascending=True).tail(15)
        name_col = "ward_name" if "ward_name" in top_df.columns else "ward_id"

        fig = px.bar(
            top_df,
            x="total_predicted_monthly",
            y=name_col,
            orientation="h",
            title="Top 15 Wards by Monthly Complaint Demand",
            color="cpi_score" if "cpi_score" in top_df.columns else "total_predicted_monthly",
            color_continuous_scale="Viridis",
            labels={"total_predicted_monthly": "Forecasted Monthly Complaints", name_col: "Ward", "cpi_score": "CPI Score"}
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif", size=11, color="#263445")
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Error rendering ward demand bar: {exc}")


def plot_prediction_demand_bar(df_predictions: pd.DataFrame):
    """Fallback horizontal bar chart when only ward_predictions.csv is available."""
    try:
        import plotly.express as px
        col = "predicted_complaints" if "predicted_complaints" in df_predictions.columns else "predicted_count"
        grouped = df_predictions.groupby("ward_id")[col].sum().reset_index()
        top_df = grouped.sort_values(by=col, ascending=True).tail(15)
        top_df["ward_label"] = top_df["ward_id"].apply(lambda x: f"Ward {x}")

        fig = px.bar(
            top_df,
            x=col,
            y="ward_label",
            orientation="h",
            title="Top Wards by Total Forecasted Complaints",
            color=col,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Error rendering demand bar: {exc}")


def plot_category_breakdown_pie(df_summary: pd.DataFrame):
    """Generates a donut chart of city-wide civic category breakdown."""
    try:
        import plotly.express as px

        cat_cols = {
            "pred_electrical": "Electrical",
            "pred_solid_waste": "Solid Waste",
            "pred_roads": "Roads & Pavements",
            "pred_animals_health": "Animals & Health",
            "pred_trees": "Trees & Greenery",
            "pred_other": "Other Services"
        }

        cat_data = []
        for col, label in cat_cols.items():
            if col in df_summary.columns:
                cat_data.append({"Category": label, "Monthly Complaints": df_summary[col].sum()})

        if cat_data:
            cat_df = pd.DataFrame(cat_data)
            fig = px.pie(
                cat_df,
                names="Category",
                values="Monthly Complaints",
                hole=0.45,
                title="City-Wide Demand by Civic Service Category",
                color_discrete_sequence=["#142B45", "#168C83", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6"]
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="sans-serif", size=11, color="#263445")
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Error rendering category breakdown: {exc}")


def plot_predictions_category_pie(df_predictions: pd.DataFrame):
    """Generates category donut from detailed predictions dataframe."""
    try:
        import plotly.express as px
        col = "predicted_complaints" if "predicted_complaints" in df_predictions.columns else "predicted_count"
        grouped = df_predictions.groupby("category_group")[col].sum().reset_index()
        grouped["Category"] = grouped["category_group"].apply(lambda s: str(s).replace("_", " ").title())

        fig = px.pie(
            grouped,
            names="Category",
            values=col,
            hole=0.45,
            title="City-Wide Category Demand Breakdown",
            color_discrete_sequence=["#142B45", "#168C83", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6"]
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Error rendering category pie: {exc}")


def render_dataset_preview(df: Optional[pd.DataFrame], df_summary: Optional[pd.DataFrame] = None):
    """
    Renders a clean preview table of the prediction dataset.
    """
    display_df = df_summary if (df_summary is not None and not df_summary.empty) else df

    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">
                        <span>📋</span> Ward Demand Intelligence Data Table
                    </div>
                    <div class="section-card-subtitle">
                        Comprehensive ward rankings, CPI scores, priority tiers, and forecasted complaint volumes
                    </div>
                </div>
                <span class="badge badge-live">Export Ready</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    if display_df is not None and not display_df.empty:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Displaying {len(display_df):,} records. Predictions ingested from official ML pipeline.")
    else:
        st.info("No prediction data schema file available to preview.")

    st.markdown("</div>", unsafe_allow_html=True)
