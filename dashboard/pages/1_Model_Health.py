
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Model Health | InGen Fleet",
    page_icon="📊",
    layout="wide",
)

st.title("Model Health")
st.caption(
    "Week 8 Capstone | Synthetic data | "
    "Training baseline vs. March 2026 test period"
)

drift_path = ROOT / "reports/W07/W07_feature_drift.csv"

if not drift_path.exists():
    st.error(f"Feature drift file not found: {drift_path}")
    st.stop()

df = pd.read_csv(drift_path)

df = df.rename(columns={
    "drift_status": "status",
    "ks_statistic": "ks",
})

st.subheader("Feature Drift Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Features Monitored", len(df))
col2.metric(
    "Critical Drift Features",
    int((df["status"] == "CRITICAL").sum()),
)
col3.metric(
    "Missingness Review Flags",
    int((df["missingness_flag"] == "REVIEW").sum()),
)

st.subheader("Population Stability Index (PSI)")
st.caption("PSI thresholds: Warning ≥ 0.10 | Critical ≥ 0.25")

fig = px.bar(
    df.sort_values("psi"),
    x="psi",
    y="feature",
    orientation="h",
    color="status",
    title="Feature Drift — PSI",
    labels={
        "psi": "Population Stability Index",
        "feature": "Feature",
    },
)

fig.add_vline(x=0.10, line_dash="dash")
fig.add_vline(x=0.25, line_dash="dash")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Kolmogorov-Smirnov (KS) Statistic")
st.caption("KS thresholds: Warning ≥ 0.10 | Critical ≥ 0.20")

fig_ks = px.bar(
    df.sort_values("ks"),
    x="ks",
    y="feature",
    orientation="h",
    color="status",
    title="Feature Drift — KS",
    labels={
        "ks": "KS Statistic",
        "feature": "Feature",
    },
)

fig_ks.add_vline(x=0.10, line_dash="dash")
fig_ks.add_vline(x=0.20, line_dash="dash")

st.plotly_chart(fig_ks, use_container_width=True)

st.subheader("Detailed Drift Metrics")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Model Performance Reference")

col1, col2 = st.columns(2)

col1.metric("Validation PR-AUC", "0.0467")
col2.metric("Validation Brier Score", "0.0393")

st.caption(
    "These are Week 4 validation reference metrics, "
    "not performance measured during the monitoring period."
)

st.warning(
    "Several telemetry features show substantial drift. "
    "Drift alone does not prove model degradation. "
    "Investigate data quality and evaluate matured labels "
    "before considering retraining."
)

st.info(
    "Monitoring thresholds are illustrative. "
    "This dashboard uses retrospective synthetic data "
    "and is not connected to a live production system."
)
