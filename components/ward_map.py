"""
Ward Pulse - Ward Map Component
Provides layout placeholders, geo-spatial metadata, and stub functions for Phase 2 Folium maps.
"""

import streamlit as st
from typing import Optional
import pandas as pd

# Lucknow municipal center coordinates for future map centering
LUCKNOW_COORDINATES = [26.8467, 80.9462]
DEFAULT_ZOOM_LEVEL = 12


import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd


def render_choropleth_map(df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Renders an interactive Folium choropleth map color-coded by CPI priority tiers.
    Falls back gracefully if data is not yet loaded.
    """
    if geojson_data is None or df_summary is None or df_summary.empty:
        render_map_placeholder()
        return

    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">
                        <span>🗺️</span> Spatial Priority Map (CPI Choropleth)
                    </div>
                    <div class="section-card-subtitle">
                        Interactive civic demand density across municipal ward boundaries
                    </div>
                </div>
                <span class="badge badge-live">Live Spatial Model</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        import folium
        from streamlit_folium import st_folium

        # Center map on ward polygon centroids (approx 12.9716, 77.5946)
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB positron")

        # Create quick lookup dictionaries
        cpi_lookup = {}
        tier_lookup = {}
        name_lookup = {}
        demand_lookup = {}

        for _, row in df_summary.iterrows():
            w_id = int(row["ward_id"])
            cpi_lookup[w_id] = float(row.get("cpi_score", 0.0))
            tier_lookup[w_id] = str(row.get("priority_tier", "LOW")).upper()
            name_lookup[w_id] = str(row.get("ward_name", f"Ward {w_id}"))
            demand_lookup[w_id] = round(float(row.get("total_predicted_monthly", 0.0)), 1)

        def get_fill_color(ward_id):
            tier = tier_lookup.get(ward_id, "NONE")
            if tier == "CRITICAL":
                return "#EF4444"  # Red
            elif tier == "HIGH":
                return "#F97316"  # Orange
            elif tier == "MEDIUM":
                return "#FBBF24"  # Amber
            elif tier == "LOW":
                return "#10B981"  # Emerald Green
            return "#E2E8F0"      # Light neutral for training/unobserved

        # Style function
        def style_fn(feature):
            w_id = feature["properties"].get("ward_id")
            is_target = w_id in cpi_lookup
            return {
                "fillColor": get_fill_color(w_id),
                "color": "#334155" if is_target else "#94A3B8",
                "weight": 1.5 if is_target else 0.75,
                "fillOpacity": 0.75 if is_target else 0.15,
            }

        folium.GeoJson(
            geojson_data,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["ward_id"],
                aliases=["Ward ID:"],
                localize=True
            )
        ).add_to(m)

        # Legend indicator
        st.markdown(
            """
            <div style="display: flex; gap: 1rem; margin-bottom: 0.75rem; flex-wrap: wrap; font-size: 0.85rem; font-weight: 500;">
                <span style="display: inline-flex; align-items: center; gap: 0.35rem;">
                    <span style="display: inline-block; width: 14px; height: 14px; background: #EF4444; border-radius: 3px;"></span> Critical (CPI &ge; 0.70)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.35rem;">
                    <span style="display: inline-block; width: 14px; height: 14px; background: #F97316; border-radius: 3px;"></span> High (0.45 &ndash; 0.70)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.35rem;">
                    <span style="display: inline-block; width: 14px; height: 14px; background: #FBBF24; border-radius: 3px;"></span> Medium (0.25 &ndash; 0.45)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.35rem;">
                    <span style="display: inline-block; width: 14px; height: 14px; background: #10B981; border-radius: 3px;"></span> Low (&lt; 0.25)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.35rem; color: #64748B;">
                    <span style="display: inline-block; width: 14px; height: 14px; background: #E2E8F0; border-radius: 3px; border: 1px solid #CBD5E1;"></span> Historical Baseline
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st_folium(m, width=None, height=480, use_container_width=True)

    except Exception as exc:
        st.warning(f"Could not render interactive map: {exc}")
        render_map_placeholder()


def render_map_placeholder():
    """
    Renders the geographical ward map placeholder fallback.
    """
    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-header">
                <div>
                    <div class="section-card-title">
                        <span>🗺️</span> Municipal Ward Geo-Spatial Map
                    </div>
                    <div class="section-card-subtitle">
                        Interactive choropleth layer visualizing predicted civic complaint intensity
                    </div>
                </div>
                <span class="placeholder-badge">Awaiting GeoJSON</span>
            </div>
            
            <div class="placeholder-box" style="min-height: 420px; background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F6 100%);">
                <div class="placeholder-icon">📍</div>
                <div class="placeholder-title">Interactive Ward Demand Choropleth Map</div>
                <div class="placeholder-desc">
                    Folium map with dynamic GeoJSON ward polygon overlays. Color-coded by predicted complaint demand tiers (Critical, High, Medium, Low).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
