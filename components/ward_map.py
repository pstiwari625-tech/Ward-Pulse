"""
Ward Pulse - Ward Map Component
Phase 3 Target: Geographical Ward Map & Spatial Choropleth
Reserved for Phase 3 implementation in strict accordance with project roadmap.
"""

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd

# Centroid coordinates for Lucknow civic center
LUCKNOW_COORDINATES = [26.8467, 80.9462]
DEFAULT_ZOOM_LEVEL = 12


def render_map_placeholder():
    """
    Renders the geographical ward map placeholder container for Phase 3.
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
                <span class="placeholder-badge">Phase 3 Target</span>
            </div>
            
            <div class="placeholder-box" style="min-height: 420px; background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F6 100%);">
                <div class="placeholder-icon">📍</div>
                <div class="placeholder-title">Interactive Ward Demand Choropleth Map</div>
                <div class="placeholder-desc">
                    Spatial choropleth map with GeoJSON ward polygon overlays and heatmaps will be implemented in <strong>Phase 3</strong>.
                    Currently operating in <strong>Phase 2: Dashboard & Analytics</strong>.
                </div>
                
                <div style="display: flex; gap: 0.75rem; margin-top: 1.25rem; flex-wrap: wrap; justify-content: center;">
                    <div class="welcome-chip" style="background: white; color: #142B45; border: 1px solid #CBD5E1;">
                        🗺️ Centroid: 26.8467° N, 80.9462° E
                    </div>
                    <div class="welcome-chip" style="background: white; color: #142B45; border: 1px solid #CBD5E1;">
                        📐 GeoJSON Polygon Overlay: Phase 3
                    </div>
                    <div class="welcome-chip" style="background: white; color: #142B45; border: 1px solid #CBD5E1;">
                        🎨 Heatmap Scale: Scheduled for Phase 3
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_choropleth_map(df_summary: Optional[pd.DataFrame] = None, geojson_data: Optional[Dict[str, Any]] = None):
    """
    Stub for Phase 3 choropleth map. Calls placeholder in Phase 2.
    """
    render_map_placeholder()
