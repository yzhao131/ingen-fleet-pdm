
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Planning assumptions
# ---------------------------------------

HORIZON = 12

HOURS_PER_SERVICE_EVENT = 8
AVAILABLE_HOURS_PER_TECHNICIAN = 30

# Preliminary historical residual interval
LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90

# ---------------------------------------
# Load weekly service demand
# ---------------------------------------

weekly = pd.read_csv(
    REPORT_DIR / "W05_weekly_service_demand.csv"
)

weekly["week_start"] = pd.to_datetime(
    weekly["week_start"],
    utc=True
)

weekly = weekly.sort_values(
    "week_start"
).reset_index(drop=True)

# Exclude incomplete first and last weeks
complete = weekly.iloc[1:-1].copy()

print("=== Training Data ===")
print("Complete weeks:", len(complete))
print(
    "Last complete week:",
    complete["week_start"].iloc[-1]
)

# ---------------------------------------
# Last Value Forecast
# ---------------------------------------

last_value = float(
    complete["service_events"].iloc[-1]
)

# The simulation ends on June 30, 2026.
# Start with the first fully unobserved calendar week.
SIMULATION_END = pd.Timestamp(
    "2026-06-30", tz="UTC"
)

first_future_monday = (
    SIMULATION_END.normalize()
    + pd.offsets.Week(weekday=0)
)

future_dates = pd.date_range(
    start=first_future_monday,
    periods=HORIZON,
    freq="W-MON",
    tz="UTC"
)

point_forecast = np.repeat(
    last_value,
    HORIZON
)

# ---------------------------------------
# Preliminary prediction intervals
# ---------------------------------------


# ---------------------------------------
# Horizon-specific preliminary intervals
# ---------------------------------------

intervals = pd.read_csv(
    REPORT_DIR / "W05_calibrated_interval_summary.csv"
)

intervals = intervals.sort_values("horizon")

expected_horizons = list(range(1, HORIZON + 1))

assert (
    intervals["horizon"].tolist()
    == expected_horizons
), "Interval horizons do not match forecast horizon."

lower_errors = intervals[
    "lower_error"
].to_numpy()

upper_errors = intervals[
    "upper_error"
].to_numpy()

lower_forecast = np.maximum(
    0,
    point_forecast + lower_errors
)

upper_forecast = np.maximum(
    0,
    point_forecast + upper_errors
)

# Service events are nonnegative integer counts.
# Round outward for conservative planning.
lower_forecast = np.floor(lower_forecast)
upper_forecast = np.ceil(upper_forecast)

# ---------------------------------------
# Technician hours and staffing
# ---------------------------------------

expected_hours = (
    point_forecast * HOURS_PER_SERVICE_EVENT
)

upper_hours = (
    upper_forecast * HOURS_PER_SERVICE_EVENT
)

expected_staff = np.ceil(
    expected_hours / AVAILABLE_HOURS_PER_TECHNICIAN
).astype(int)

upper_staff = np.ceil(
    upper_hours / AVAILABLE_HOURS_PER_TECHNICIAN
).astype(int)

planning = pd.DataFrame({
    "week_start": future_dates,
    "forecast_service_events": point_forecast,
    "lower_service_events": lower_forecast,
    "upper_service_events": upper_forecast,
    "expected_technician_hours": expected_hours,
    "upper_technician_hours": upper_hours,
    "expected_technicians": expected_staff,
    "upper_technicians": upper_staff,
    "forecast_horizon": np.arange(1, HORIZON + 1)
})

output = (
    REPORT_DIR / "W05_technician_staffing_plan.csv"
)

planning.to_csv(output, index=False)

print("\n=== 12-Week Technician Staffing Plan ===")
print(planning.to_string(index=False))

print("\n=== Planning Summary ===")
print(
    "Total forecast service events:",
    planning["forecast_service_events"].sum()
)
print(
    "Total expected technician-hours:",
    planning["expected_technician_hours"].sum()
)
print(
    "Maximum upper-bound technicians:",
    planning["upper_technicians"].max()
)

print("\nSaved:", output)
