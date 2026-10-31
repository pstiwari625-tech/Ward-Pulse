"""
Ward Pulse: Frontend Map View Component
Phase 3: Official GeoJSON Map Integration
"""

from typing import Optional, Dict, Any
import pandas as pd
from components.ward_map import render_official_ward_map


def render_choropleth_map(st=None, summary_data=None, geojson_data=None):
    """
    Renders the official interactive Folium map with 198 official ward boundaries.
    """
    render_official_ward_map(df_summary=summary_data)
