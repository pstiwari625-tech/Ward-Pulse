"""
Ward Pulse - Sidebar Component
Renders the navigation sidebar, project branding, and pipeline status.
"""

import streamlit as st
from typing import Dict, Any


def render_sidebar(data_status: Dict[str, Any]) -> str:
    """
    Renders the sidebar navigation and branding.
    
    Args:
        data_status: Dictionary containing current pipeline connection state.
        
    Returns:
        str: Currently selected navigation page ('Overview', 'Ward Analytics', 'Map View', 'About Project').
    """
    with st.sidebar:
        # Branding Header
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">
                    <div class="sidebar-logo-icon">WP</div>
                    <div>
                        <div class="sidebar-logo-text">Ward Pulse</div>
                        <div class="sidebar-logo-tagline">Civic Demand Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Municipal Location Indicator
        st.markdown(
            """
            <div style="margin-bottom: 1.25rem;">
                <span class="badge badge-location" style="width: 100%; justify-content: center;">
                    📍 Lucknow Municipal Corp. (LMC)
                </span>
            </div>
            """,
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
        
        # Clean selected option string for routing
        active_view = selected_nav.split(" ", 1)[1]

        # Phase 2 Filter Preview (Disabled / Placeholder)
        st.markdown('<div class="sidebar-section-title">Ward Filters (Phase 2)</div>', unsafe_allow_html=True)
        
        st.selectbox(
            label="Select Zone",
            options=["All Zones (Preview)", "Zone 1 - Central", "Zone 2 - South", "Zone 3 - North", "Zone 4 - East"],
            index=0,
            disabled=True,
            help="Zone and ward level filtering will be fully active in Phase 2 with backend data connection."
        )
        
        st.selectbox(
            label="Select Ward",
            options=["All Wards (110 Wards)", "Ward 101 - Hazratganj", "Ward 102 - Alambagh", "Ward 103 - Gomti Nagar"],
            index=0,
            disabled=True,
            help="Ward selection dropdown enabled in Phase 2."
        )

        # Pipeline Status Card
        st.markdown('<div class="sidebar-section-title">Backend Pipeline</div>', unsafe_allow_html=True)
        
        is_official = data_status.get("is_official", False)
        badge_class = data_status.get("badge_class", "badge-pending")
        badge_label = data_status.get("badge_label", "Dataset Pending")
        
        st.markdown(
            f"""
            <div class="sidebar-pipeline-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4>Data Status</h4>
                    <span class="badge {badge_class}" style="font-size: 0.7rem; padding: 0.2rem 0.5rem;">
                        {badge_label}
                    </span>
                </div>
                <p><strong>Target Integration:</strong><br><code>data/predictions/ward_predictions.csv</code></p>
                <p style="margin-top: 0.4rem;">{data_status.get('description', '')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Sidebar Footer
        st.markdown(
            """
            <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border-color); text-align: center;">
                <span style="font-size: 0.72rem; color: #94A3B8;">
                    Ward Pulse v0.1 • Hackathon Edition<br>
                    Phase 1: Frontend Foundation
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    return active_view
