"""
Ward Pulse: Frontend Charts Component
Renders Plotly visualizations for category breakdowns and priority distributions.
"""

def render_category_distribution(st, category_sums):
    """Renders category complaint breakdown as a horizontal bar chart."""
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(list(category_sums.items()), columns=["Category", "Monthly Complaints"])
        df = df.sort_values(by="Monthly Complaints", ascending=True)
        fig = px.bar(
            df,
            x="Monthly Complaints",
            y="Category",
            orientation="h",
            title="City-Wide Monthly Forecast by Civic Category",
            color="Monthly Complaints",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.write("### City-Wide Monthly Forecast by Civic Category")
        for cat, val in category_sums.items():
            st.write(f"- **{cat.replace('_', ' ').title()}**: {round(val, 1)} complaints")
