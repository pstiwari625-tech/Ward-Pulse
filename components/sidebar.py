"""
Ward Pulse - Sidebar Component
Renders the navigation sidebar, project branding, and municipal filters.
Clean, executive municipal design without developer clutter or phase tags.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


def get_official_zones_and_wards():
    """Load official zones and ward list for dynamic sidebar filtering."""
    try:
        from utils.geo_loader import load_ward_features
        df_feat, _ = load_ward_features()
        if df_feat is not None and not df_feat.empty:
            zones = sorted(df_feat["zone"].dropna().unique().tolist())
            return df_feat, zones
    except Exception:
        pass
    return None, []


def render_sidebar(data_status: Dict[str, Any]) -> str:
    """
    Renders the sidebar navigation, municipal branding, and active filters.
    
    Args:
        data_status: Dictionary containing current pipeline connection state.
        
    Returns:
        str: Currently selected navigation page ('Overview', 'Ward Analytics', 'Map View', 'About Project').
    """
    with st.sidebar:
        # Branding Header
        st.markdown(
            """<div class="sidebar-brand">
<div class="sidebar-logo">
<div class="sidebar-logo-icon">🏛️</div>
<div>
<div class="sidebar-logo-text">Ward Pulse</div>
<div class="sidebar-logo-tagline">Civic Demand Intelligence</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

        # Navigation Menu
        st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
        
        nav_options = [
            "📊 Overview",
            "📈 Ward Analytics",
            "🗺️ Map View",
            "ℹ️ About Project"
        ]
        
        selected_nav = st.radio(
            label="Dashboard Navigation",
            options=nav_options,
            index=0,
            label_visibility="collapsed"
        )
        
        active_view = selected_nav.split(" ", 1)[1]

        # Municipal Filtering Section
        st.markdown('<div class="sidebar-section-title">Municipal Jurisdiction</div>', unsafe_allow_html=True)
        
        df_feat, zones = get_official_zones_and_wards()
        zone_options = ["All Administrative Zones"] + zones
        
        selected_zone = st.selectbox(
            label="Administrative Zone",
            options=zone_options,
            index=0,
            key="sidebar_zone_select",
            help="Filter analytics across specific administrative divisions."
        )
        st.session_state["filter_zone"] = selected_zone

        # Dynamic Ward list based on selected zone
        if df_feat is not None and not df_feat.empty:
            if selected_zone != "All Administrative Zones":
                filtered_wards_df = df_feat[df_feat["zone"] == selected_zone].sort_values(by="ward_id")
            else:
                filtered_wards_df = df_feat.sort_values(by="ward_id")

            ward_labels = ["All Municipal Wards"] + [
                f"Ward {int(r['ward_id'])} - {r['ward_name']}" for _, r in filtered_wards_df.iterrows()
            ]
        else:
            ward_labels = ["All Municipal Wards"]

        selected_ward = st.selectbox(
            label="Municipal Ward",
            options=ward_labels,
            index=0,
            key="sidebar_ward_select",
            help="Select a specific municipal ward for focused demand analysis."
        )
        st.session_state["filter_ward"] = selected_ward

        # Minimal Status Indicator
        is_official = data_status.get("is_official", False)
        badge_class = data_status.get("badge_class", "badge-live" if is_official else "badge-pending")
        badge_label = data_status.get("badge_label", "System Active")
        
        st.markdown(
            f"""<div style="margin-top: 2rem; padding: 0.75rem; background: #F8FAFC; border: 1px solid var(--border-color); border-radius: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.75rem; font-weight: 600; color: #334155;">System Status</span>
<span class="badge {badge_class}" style="font-size: 0.7rem; padding: 0.2rem 0.5rem;">{badge_label}</span>
</div>
<div style="font-size: 0.72rem; color: #64748B; margin-top: 0.35rem;">198 Municipal Wards Monitored</div>
</div>""",
            unsafe_allow_html=True
        )

        # Clean Sidebar Footer
        st.markdown(
            """<div style="margin-top: 1.5rem; padding-top: 0.75rem; border-top: 1px solid var(--border-color); text-align: center;">
<span style="font-size: 0.72rem; color: #94A3B8;">Ward Pulse Platform • Civic Intelligence</span>
</div>""",
            unsafe_allow_html=True
        )

    return active_view
