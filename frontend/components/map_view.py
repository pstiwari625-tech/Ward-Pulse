"""
Ward Pulse: Offline Plotly Choropleth Map Component (frontend/components/map_view.py)
Renders a pure vector Plotly choropleth using local GeoJSON polygon boundaries.
Strictly offline: Zero external tile servers (pure local vector geometry, no external basemaps).
Renders the 156 training wards in light neutral grey and the 40 target wards colored by the selected metric.
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import config

def render_offline_choropleth(
    st,
    geojson_data: Dict[str, Any],
    target_data: List[Dict[str, Any]],
    all_ward_ids: List[int],
    metric_key: str = "metric_value",
    metric_label: str = "Predicted Demand",
    selected_category: str = "All Categories",
    selected_month: str = "18-Month Average"
):
    """
    Renders pure vector offline Plotly choropleth map with rich hover tooltips.
    """
    if not geojson_data:
        st.warning("GeoJSON boundary data not available. Displaying table view.")
        st.dataframe(target_data)
        return

    target_ward_ids = {int(r["ward_id"]) for r in target_data}
    training_ward_ids = [w for w in all_ward_ids if w not in target_ward_ids]

    fig = go.Figure()

    # Trace 1: 156 Training Wards (Light Neutral Grey Fill, Thin Dark Border)
    fig.add_trace(go.Choropleth(
        geojson=geojson_data,
        featureidkey="properties.ward_id",
        locations=training_ward_ids,
        z=[0] * len(training_ward_ids),
        colorscale=[[0, "#E2E8F0"], [1, "#E2E8F0"]],
        showscale=False,
        marker=dict(line=dict(color="#94A3B8", width=0.8)),
        hoverinfo="text",
        text=[f"<b>Ward {w}</b><br>Historical Training Ward (156 City Wards)" for w in training_ward_ids],
        name="Training Wards"
    ))

    # Trace 2: 40 Target Wards (Colored by metric, rich tooltip)
    target_locations = [int(r["ward_id"]) for r in target_data]
    target_z = [float(r[metric_key]) for r in target_data]

    # Customdata packing for tooltip:
    # 0: ward_name, 1: metric_label, 2: predicted_val, 3: lower_90, 4: upper_90,
    # 5: confidence_label, 6: top_category, 7: priority_tier
    customdata = [
        [
            r.get("ward_name", f"Ward {r['ward_id']}"),
            metric_label,
            float(r.get("predicted_count", r.get("total_pred", 0.0))),
            float(r.get("lower_90", 0.0)),
            float(r.get("upper_90", 0.0)),
            r.get("confidence_label", "Medium"),
            r.get("top_category", "solid_waste"),
            r.get("priority_tier", "MEDIUM")
        ]
        for r in target_data
    ]

    # Colorscale selection based on metric
    if "CPI" in metric_label:
        colorscale = "YlOrRd"
    elif "Width" in metric_label:
        colorscale = "Purples"
    else:
        colorscale = "YlOrRd"

    hovertemplate = (
        "<b>%{customdata[0]} (Ward %{location})</b><br>"
        "------------------------------------<br>"
        f"<b>Category:</b> {selected_category}<br>"
        f"<b>Month:</b> {selected_month}<br>"
        "<b>%{customdata[1]}:</b> %{z:.2f}<br>"
        "<b>90% Interval:</b> [%{customdata[3]:.1f}, %{customdata[4]:.1f}]<br>"
        "<b>Confidence:</b> %{customdata[5]}<br>"
        "<b>Dominant Need:</b> %{customdata[6]}<br>"
        "<b>Priority Tier:</b> %{customdata[7]}"
        "<extra></extra>"
    )

    fig.add_trace(go.Choropleth(
        geojson=geojson_data,
        featureidkey="properties.ward_id",
        locations=target_locations,
        z=target_z,
        colorscale=colorscale,
        marker=dict(line=dict(color="#1E293B", width=1.5)),
        customdata=customdata,
        hovertemplate=hovertemplate,
        colorbar=dict(
            title=dict(text=metric_label, side="right"),
            thickness=15,
            len=0.75,
            x=1.02
        ),
        name="Target Wards"
    ))

    # Strict Offline Geos Configuration (No web tiles, no mapbox, no external CDN)
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator"
    )

    fig.update_layout(
        margin=dict(r=10, t=10, l=10, b=10),
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)")
    )

    st.plotly_chart(fig, use_container_width=True)

# Backward compatibility alias
def render_choropleth_map(st, summary_data, geojson_data, category="All Categories", month="18-Month Average", metric="Predicted Demand"):
    all_ward_ids = [f["properties"]["ward_id"] for f in geojson_data["features"]] if geojson_data else []
    # Ensure metric_value exists
    formatted_data = []
    for r in summary_data:
        rc = dict(r)
        if "metric_value" not in rc:
            rc["metric_value"] = float(rc.get("predicted_count", rc.get("total_predicted_monthly", 0.0)))
        formatted_data.append(rc)
    render_offline_choropleth(
        st=st,
        geojson_data=geojson_data,
        target_data=formatted_data,
        all_ward_ids=all_ward_ids,
        metric_key="metric_value",
        metric_label=metric,
        selected_category=category,
        selected_month=month
    )
