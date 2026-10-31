"""
Ward Pulse: Frontend Metric Cards Component
Renders top-level KPI metrics for municipal decision makers.
"""

def render_kpi_metrics(st, total_wards, critical_wards, mean_monthly_complaints, oof_mae):
    """Renders 4 executive summary KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Target Wards Monitored", value=total_wards)
    with col2:
        st.metric(label="Critical Priority Wards", value=critical_wards, delta=f"{round(critical_wards/max(1, total_wards)*100, 1)}%")
    with col3:
        st.metric(label="Avg Monthly Grievances / Ward", value=round(mean_monthly_complaints, 1))
    with col4:
        st.metric(label="Spatial Model Out-of-Fold MAE", value=oof_mae)
