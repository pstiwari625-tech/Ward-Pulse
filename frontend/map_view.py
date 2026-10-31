"""
Ward Pulse - Frontend Map View
Phase 3: Official GeoJSON Map Integration
Provides frontend interface to official municipal ward boundaries and attributes.
"""

from typing import Optional, Dict, Any
import pandas as pd
from components.ward_map import render_official_ward_map


def render_map_view(
    df_summary: Optional[pd.DataFrame] = None,
    geojson_data: Optional[Dict[str, Any]] = None,
    df_predictions: Optional[pd.DataFrame] = None
):
    """
    Renders the official Ward Pulse geographical map view.
    Integrates 198 official ward boundaries with ward socio-economic features.
    """
    render_official_ward_map(df_summary=df_summary, df_predictions=df_predictions)


def render_choropleth_map(
    st_context=None,
    summary_data=None,
    geojson_data=None,
    df_predictions=None
):
    """Backward-compatible wrapper for render_official_ward_map."""
    render_official_ward_map(df_summary=summary_data, df_predictions=df_predictions)
