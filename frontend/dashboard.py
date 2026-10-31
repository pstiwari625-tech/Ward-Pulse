"""
Ward Pulse - Frontend Dashboard Module
Phase 2 & 3: Master Dashboard Interface
Provides header, KPI metric cards, and primary views according to modular structure.
"""

from components.dashboard import (
    render_header,
    render_status_banner,
    render_welcome_card,
    render_kpi_cards,
    render_overview,
    render_analytics_view,
    render_map_view,
    render_about_view
)

__all__ = [
    "render_header",
    "render_status_banner",
    "render_welcome_card",
    "render_kpi_cards",
    "render_overview",
    "render_analytics_view",
    "render_map_view",
    "render_about_view"
]
