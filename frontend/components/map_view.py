"""
Ward Pulse: Frontend Map View Component
Renders interactive Folium choropleths with GeoJSON polygon overlays.
"""

def render_choropleth_map(st, summary_data, geojson_data):
    """Renders Folium map with ward polygons and CPI priority color-coding."""
    try:
        import folium
        from streamlit_folium import st_folium

        # Initialize map centered on Bengaluru coordinates
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB positron")

        # Create quick lookup for CPI scores
        cpi_lookup = {int(r["ward_id"]): float(r["cpi_score"]) for r in summary_data}
        tier_lookup = {int(r["ward_id"]): r["priority_tier"] for r in summary_data}

        def get_color(ward_id):
            tier = tier_lookup.get(ward_id, "NONE")
            if tier == "CRITICAL":
                return "#d73027"
            elif tier == "HIGH":
                return "#fc8d59"
            elif tier == "MEDIUM":
                return "#fee08b"
            elif tier == "LOW":
                return "#91cf60"
            return "#d9d9d9"

        if geojson_data:
            folium.GeoJson(
                geojson_data,
                style_function=lambda feature: {
                    "fillColor": get_color(feature["properties"]["ward_id"]),
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 0.6 if feature["properties"]["ward_id"] in cpi_lookup else 0.15,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["ward_id"],
                    aliases=["Ward ID:"],
                    localize=True
                )
            ).add_to(m)

        st_folium(m, width=800, height=500)
    except ImportError:
        st.info("Folium or streamlit-folium not installed in active environment. Displaying tabular summary.")
        st.dataframe(summary_data)
