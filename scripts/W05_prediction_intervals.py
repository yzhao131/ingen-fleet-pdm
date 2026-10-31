
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

data = pd.read_csv(
    REPORT_DIR / "W05_weekly_service_demand.csv",
    parse_dates=["week_start"]
)

# Complete weeks only
data = data.iloc[1:-1].reset_index(drop=True)

y = data["service_events"].to_numpy(dtype=float)

HORIZON = 12
MIN_TRAIN = 40
STEP = 4
COVERAGE_TARGET = 0.80

results = []

for train_end in range(
    MIN_TRAIN,
    len(y) - HORIZON + 1,
    STEP
):
    history = y[:train_end]
    actual = y[train_end:train_end + HORIZON]

    # Last-value point forecast
    point_forecast = np.repeat(
        history[-1],
        HORIZON
    )

    # Historical one-step naive residuals
    historical_errors = (
        history[1:] - history[:-1]
    )

    # Central 80% empirical residual interval
    lower_error = np.quantile(
        historical_errors, 0.10
    )
    upper_error = np.quantile(
        historical_errors, 0.90
    )

    lower = np.maximum(
        point_forecast + lower_error,
        0
    )
    upper = np.maximum(
        point_forecast + upper_error,
        lower
    )

    for h in range(HORIZON):
        results.append({
            "origin_week": str(
                data["week_start"].iloc[train_end - 1]
            ),
            "horizon": h + 1,
            "actual": actual[h],
            "predicted": point_forecast[h],
            "lower_80": lower[h],
            "upper_80": upper[h],
            "covered": (
                lower[h] <= actual[h] <= upper[h]
            )
        })


results_df = pd.DataFrame(results)

coverage = results_df["covered"].mean()

coverage_by_horizon = (
    results_df.groupby("horizon")["covered"]
    .mean()
    .reset_index(name="empirical_coverage")
)

results_df.to_csv(
    REPORT_DIR / "W05_prediction_intervals.csv",
    index=False
)

coverage_by_horizon.to_csv(
    REPORT_DIR / "W05_interval_coverage.csv",
    index=False
)

print("=== Prediction Interval Evaluation ===")
print("Target coverage:", COVERAGE_TARGET)
print("Empirical coverage:", round(coverage, 3))
print("Total forecasts:", len(results_df))

print("\n=== Coverage by Horizon ===")
print(coverage_by_horizon.to_string(index=False))


# --------------------------------------------------
# Prediction Interval Width Analysis
# --------------------------------------------------

results_df["interval_width"] = (
    results_df["upper_80"] - results_df["lower_80"]
)

average_width = results_df["interval_width"].mean()

width_by_horizon = (
    results_df.groupby("horizon")
    .agg(
        mean_width=("interval_width", "mean"),
        coverage=("covered", "mean")
    )
    .reset_index()
)

width_by_horizon.to_csv(
    REPORT_DIR / "W05_interval_width_analysis.csv",
    index=False
)

print("\n=== Interval Width Analysis ===")
print("Average interval width:", round(average_width, 3))
print("\nWidth and coverage by horizon:")
print(width_by_horizon.round(3).to_string(index=False))
