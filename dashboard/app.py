
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="InGen Fleet Monitoring",
    page_icon="🤖",
    layout="wide",
)

st.title("InGen Fleet Monitoring & Operations")
st.caption(
    "Week 8 Capstone | Synthetic fleet data | "
    "Retrospective demonstration"
)

st.header("Fleet Risk Today")

queue_path = (
    ROOT / "reports/W07/"
    "W07_service_queue_20260308_cost20.csv"
)

if not queue_path.exists():
    st.error(f"Queue file not found: {queue_path}")
    st.stop()

df = pd.read_csv(queue_path)

selected = df[df["selected_for_service"] == True].copy()
eligible = df[df["eligible"] == True].copy()
suppressed = df[df["suppressed"] == True].copy()

capacity = int(df["daily_capacity"].iloc[0])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Robots Scored", len(df))
col2.metric("Above Threshold", int(df["above_threshold"].sum()))
col3.metric("Selected for Service", len(selected))
col4.metric("Capacity Utilization", f"{len(selected) / capacity:.0%}")

st.subheader("Daily Maintenance Priority Queue")

st.dataframe(
    selected[
        [
            "queue_rank",
            "unit_id",
            "site_id",
            "calibrated_risk",
            "days_since_last_maintenance",
            "recommended_action",
        ]
    ].sort_values("queue_rank"),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Highest-Risk Robots")

top = df.nlargest(15, "calibrated_risk").copy()
top["risk_percent"] = top["calibrated_risk"] * 100

fig = px.bar(
    top.sort_values("risk_percent"),
    x="risk_percent",
    y="unit_id",
    orientation="h",
    color="selected_for_service",
    labels={
        "risk_percent": "Calibrated Failure Risk (%)",
        "unit_id": "Robot",
        "selected_for_service": "Selected",
    },
    title="Top 15 Robots by Predicted 7-Day Failure Risk",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Operational Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Eligible Robots", len(eligible))
col2.metric("Suppressed Robots", len(suppressed))
col3.metric(
    "Deferred Eligible Robots",
    max(len(eligible) - len(selected), 0),
)

st.info(
    "Risk scores are based on synthetic data. "
    "This dashboard demonstrates a retrospective "
    "maintenance decision workflow, not live fleet operations."
)
