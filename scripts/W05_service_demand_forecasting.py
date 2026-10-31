
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "causal_fixed"
    / "raw"
    / "maintenance.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "W05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load maintenance records
maintenance = pd.read_csv(
    DATA_PATH,
    parse_dates=["start_time", "end_time"]
)

# Use UTC timestamps consistently
maintenance["start_time"] = pd.to_datetime(
    maintenance["start_time"],
    utc=True
)

# Count each service event in the week it starts.
# Weeks run Monday through Sunday.
maintenance["week_start"] = (
    maintenance["start_time"].dt.normalize()
    - pd.to_timedelta(
        maintenance["start_time"].dt.weekday,
        unit="D"
    )
)

weekly_counts = (
    maintenance.groupby("week_start")
    .size()
    .rename("service_events")
)

# Cover the simulation window, including zero-event weeks.
simulation_start = pd.Timestamp(
    "2025-01-01", tz="UTC"
)
simulation_end = pd.Timestamp(
    "2026-06-30", tz="UTC"
)

first_week = (
    simulation_start
    - pd.Timedelta(days=simulation_start.weekday())
)

last_week = (
    simulation_end
    - pd.Timedelta(days=simulation_end.weekday())
)

all_weeks = pd.date_range(
    first_week,
    last_week,
    freq="W-MON",
    tz="UTC"
)

weekly = (
    weekly_counts
    .reindex(all_weeks, fill_value=0)
    .rename_axis("week_start")
    .reset_index()
)

weekly.to_csv(
    OUTPUT_DIR / "W05_weekly_service_demand.csv",
    index=False
)

print("=== Weekly Service Demand ===")
print("Total weeks:", len(weekly))
print("Total service events:", weekly["service_events"].sum())
print("Mean events per week:", round(weekly["service_events"].mean(), 3))
print("Zero-event weeks:", (weekly["service_events"] == 0).sum())

print("\nFirst 10 weeks:")
print(weekly.head(10).to_string(index=False))

print("\nLast 10 weeks:")
print(weekly.tail(10).to_string(index=False))


# --------------------------------------------------
# Forecasting dataset: complete weeks only
# --------------------------------------------------

weekly_model = weekly.iloc[1:-1].copy().reset_index(drop=True)

print("\n=== Modeling Dataset ===")
print("Complete weeks:", len(weekly_model))
print("Start:", weekly_model["week_start"].min())
print("End:", weekly_model["week_start"].max())

# --------------------------------------------------
# Baseline: 4-week seasonal naive
# --------------------------------------------------

def seasonal_naive_forecast(history, horizon=8, season_length=4):
    """
    Repeat the last observed seasonal cycle.
    """
    if len(history) < season_length:
        raise ValueError("Not enough history for seasonal naive.")

    last_season = list(history[-season_length:])

    return [
        last_season[i % season_length]
        for i in range(horizon)
    ]


history = weekly_model["service_events"].to_numpy()

baseline_forecast = seasonal_naive_forecast(
    history,
    horizon=8,
    season_length=4
)

future_weeks = pd.date_range(
    weekly_model["week_start"].iloc[-1] + pd.Timedelta(weeks=1),
    periods=8,
    freq="W-MON",
    tz="UTC"
)

baseline_results = pd.DataFrame({
    "week_start": future_weeks,
    "forecast_service_events": baseline_forecast
})

print("\n=== 8-Week Seasonal-Naive Forecast ===")
print(baseline_results.to_string(index=False))

baseline_results.to_csv(
    OUTPUT_DIR / "W05_seasonal_naive_forecast.csv",
    index=False
)
