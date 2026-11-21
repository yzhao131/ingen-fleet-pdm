
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Planning | InGen Fleet",
    layout="wide",
)

st.title("Fleet Maintenance & Resource Planning")
st.caption(
    "Week 8 Capstone | Synthetic fleet data | "
    "Historical analysis and illustrative forecasts"
)


def load_csv(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


# --------------------------------------------------
# 1. Historical Weekly Service Demand
# --------------------------------------------------

st.header("1. Historical Weekly Service Demand")

demand = load_csv(
    "reports/W05/W05_weekly_service_demand.csv"
)

demand["week_start"] = pd.to_datetime(
    demand["week_start"], utc=True
)

fig_demand = px.line(
    demand.sort_values("week_start"),
    x="week_start",
    y="service_events",
    markers=True,
    title="Historical Weekly Maintenance Events",
    labels={
        "week_start": "Week",
        "service_events": "Service Events",
    },
)

st.plotly_chart(fig_demand, use_container_width=True)


# --------------------------------------------------
# 2. Technician Staffing Forecast
# --------------------------------------------------

st.header("2. Technician Staffing Forecast")

staffing = load_csv(
    "reports/W05/W05_technician_staffing_plan.csv"
)

staffing["week_start"] = pd.to_datetime(
    staffing["week_start"], utc=True
)

staffing = staffing.sort_values("week_start")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Forecast Horizon",
    f"{len(staffing)} weeks",
)

col2.metric(
    "Total Forecast Service Events",
    f"{staffing['forecast_service_events'].sum():.1f}",
)

col3.metric(
    "Peak Expected Technicians",
    int(staffing["expected_technicians"].max()),
)

fig_staff = go.Figure()

fig_staff.add_trace(
    go.Scatter(
        x=staffing["week_start"],
        y=staffing["upper_service_events"],
        mode="lines",
        name="Upper Estimate",
        line=dict(dash="dash"),
    )
)

fig_staff.add_trace(
    go.Scatter(
        x=staffing["week_start"],
        y=staffing["lower_service_events"],
        mode="lines",
        name="Lower Estimate",
        fill="tonexty",
        fillcolor="rgba(100, 149, 237, 0.15)",
    )
)

fig_staff.add_trace(
    go.Scatter(
        x=staffing["week_start"],
        y=staffing["forecast_service_events"],
        mode="lines+markers",
        name="Expected Service Events",
    )
)

fig_staff.update_layout(
    title="12-Week Service Demand Forecast",
    xaxis_title="Week",
    yaxis_title="Forecast Service Events",
)

st.plotly_chart(fig_staff, use_container_width=True)

st.subheader("Weekly Staffing Requirements")

st.dataframe(
    staffing[
        [
            "week_start",
            "forecast_service_events",
            "lower_service_events",
            "upper_service_events",
            "expected_technician_hours",
            "expected_technicians",
            "upper_technicians",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 3. Battery Replacement Forecast
# --------------------------------------------------

st.header("3. Battery Replacement Planning")

battery = load_csv(
    "reports/W05/W05_battery_replacement_forecast.csv"
)

threshold = st.selectbox(
    "Battery Replacement Threshold",
    options=sorted(battery["threshold_pct"].unique()),
    format_func=lambda x: f"{x:.0f}% capacity",
)

battery_view = battery[
    battery["threshold_pct"] == threshold
].copy()

battery_view["predicted_crossing_date"] = pd.to_datetime(
    battery_view["predicted_crossing_date"],
    utc=True,
    errors="coerce",
)

valid_forecasts = battery_view.dropna(
    subset=["predicted_crossing_date"]
).copy()

col1, col2 = st.columns(2)

col1.metric(
    "Robots Evaluated",
    len(battery_view),
)

col2.metric(
    "Forecasted Crossing Dates",
    len(valid_forecasts),
)

if not valid_forecasts.empty:
    valid_forecasts["crossing_month"] = (
        valid_forecasts["predicted_crossing_date"]
        .dt.strftime("%Y-%m")
    )

    monthly_crossings = (
        valid_forecasts.groupby("crossing_month")
        .size()
        .reset_index(name="robots")
    )

    fig_battery = px.bar(
        monthly_crossings,
        x="crossing_month",
        y="robots",
        title=(
            f"Predicted Battery Capacity Crossings "
            f"({threshold:.0f}% Threshold)"
        ),
        labels={
            "crossing_month": "Month",
            "robots": "Robots",
        },
    )

    st.plotly_chart(fig_battery, use_container_width=True)

st.dataframe(
    battery_view[
        [
            "unit_id",
            "current_capacity_pct",
            "predicted_crossing_date",
            "model_source",
            "status",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 4. Monthly Battery Procurement
# --------------------------------------------------

st.header("4. Monthly Battery Procurement")

procurement = load_csv(
    "reports/W05/W05_battery_procurement_monthly.csv"
)

procurement_view = procurement[
    procurement["threshold_pct"] == threshold
].copy()

if not procurement_view.empty:
    fig_procurement = px.bar(
        procurement_view.sort_values("order_month"),
        x="order_month",
        y="battery_units_to_order",
        color="scenario",
        barmode="group",
        title="Monthly Battery Procurement by Scenario",
        labels={
            "order_month": "Order Month",
            "battery_units_to_order": "Battery Units",
            "scenario": "Scenario",
        },
    )

    st.plotly_chart(
        fig_procurement,
        use_container_width=True,
    )
else:
    st.info(
        "No procurement records for the selected threshold."
    )


# --------------------------------------------------
# 5. Maintenance Capacity & Cost Analysis
# --------------------------------------------------

st.header("5. Maintenance Capacity Analysis")

impact = load_csv(
    "reports/W07/W07_impact_results.csv"
)

cost_ratio = st.selectbox(
    "False Negative to False Positive Cost Ratio",
    options=sorted(impact["cost_ratio"].unique()),
    index=2,
    format_func=lambda x: f"{int(x)}:1",
)

impact_view = impact[
    impact["cost_ratio"] == cost_ratio
].copy()

capacity_view = impact_view[
    impact_view["strategy"] == "capacity_limited"
].copy()

fig_cost = px.line(
    capacity_view.sort_values("daily_capacity"),
    x="daily_capacity",
    y="total_cost_units",
    markers=True,
    title="Simulated Cost vs. Daily Maintenance Capacity",
    labels={
        "daily_capacity": "Daily Capacity (Robots)",
        "total_cost_units": "Simulated Cost Units",
    },
)

baseline = impact_view[
    impact_view["strategy"] == "no_alert"
]

if not baseline.empty:
    baseline_cost = baseline["total_cost_units"].iloc[0]

    fig_cost.add_hline(
        y=baseline_cost,
        line_dash="dash",
        annotation_text="No-Alert Baseline",
    )

st.plotly_chart(fig_cost, use_container_width=True)

st.subheader("Capacity Sensitivity Results")

st.dataframe(
    impact_view[
        [
            "strategy",
            "daily_capacity",
            "tp",
            "fp",
            "fn",
            "recall",
            "alerts",
            "total_cost_units",
            "cost_difference_vs_no_alert",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Cost differences are calculated relative to the no-alert baseline. "
    "Negative values indicate lower simulated costs. "
    "Results are based on synthetic data and illustrative cost assumptions, "
    "not realized financial savings."
)

st.info(
    "Planning forecasts are based on synthetic data and "
    "model assumptions. Cost comparisons are retrospective "
    "simulations in abstract cost units, not realized savings. "
    "Forecast uncertainty and operational constraints "
    "should be considered before making decisions."
)
