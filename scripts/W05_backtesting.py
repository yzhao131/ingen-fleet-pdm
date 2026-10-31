
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

data = pd.read_csv(
    REPORT_DIR / "W05_weekly_service_demand.csv",
    parse_dates=["week_start"]
)

# Remove incomplete boundary weeks
data = data.iloc[1:-1].reset_index(drop=True)

y = data["service_events"].to_numpy(dtype=float)

HORIZON = 12
MIN_TRAIN = 40
STEP = 4


def forecast_last_value(history, horizon):
    return np.repeat(history[-1], horizon)


def forecast_moving_average(history, horizon):
    return np.repeat(np.mean(history[-4:]), horizon)


def forecast_seasonal_naive(history, horizon):
    return np.resize(history[-4:], horizon)


models = {
    "Last Value": forecast_last_value,
    "Moving Average (4)": forecast_moving_average,
    "Seasonal Naive (4)": forecast_seasonal_naive,
}

results = []

for train_end in range(
    MIN_TRAIN,
    len(y) - HORIZON + 1,
    STEP
):
    history = y[:train_end]
    actual = y[train_end:train_end + HORIZON]

    for model_name, forecast_fn in models.items():
        predictions = forecast_fn(history, HORIZON)

        for h in range(HORIZON):
            results.append({
                "model": model_name,
                "origin_week": str(
                    data["week_start"].iloc[train_end - 1]
                ),
                "horizon": h + 1,
                "actual": actual[h],
                "predicted": predictions[h],
                "absolute_error": abs(
                    actual[h] - predictions[h]
                )
            })


results_df = pd.DataFrame(results)

# Aggregate MAE across all origins and horizons
summary = (
    results_df.groupby("model")
    .agg(
        MAE=("absolute_error", "mean"),
        Forecasts=("absolute_error", "size")
    )
    .sort_values("MAE")
)

# Evaluate accuracy separately for each horizon
horizon_mae = (
    results_df.groupby(["model", "horizon"])
    ["absolute_error"]
    .mean()
    .unstack("model")
)

results_df.to_csv(
    REPORT_DIR / "W05_backtesting_results.csv",
    index=False
)

summary.to_csv(
    REPORT_DIR / "W05_baseline_comparison.csv"
)

horizon_mae.to_csv(
    REPORT_DIR / "W05_horizon_mae.csv"
)

print("=== Baseline Comparison ===")
print(summary.round(3))

print("\n=== MAE by Forecast Horizon ===")
print(horizon_mae.round(3))

print("\nNumber of rolling origins:",
      results_df["origin_week"].nunique())
