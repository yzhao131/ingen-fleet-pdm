
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

data = pd.read_csv(
    REPORT_DIR / "W05_weekly_service_demand.csv",
    parse_dates=["week_start"]
)

# Same complete-week dataset as baseline backtesting
data = data.iloc[1:-1].reset_index(drop=True)

y = data["service_events"].to_numpy(dtype=float)

HORIZON = 12
MIN_TRAIN = 40
STEP = 4


def prophet_forecast(history, dates, horizon):
    train_df = pd.DataFrame({
        "ds": pd.to_datetime(dates, utc=True).dt.tz_localize(None),
        "y": history
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80
    )

    model.fit(train_df)

    future = model.make_future_dataframe(
        periods=horizon,
        freq="W-MON"
    )

    forecast = model.predict(future)

    return np.maximum(
        forecast["yhat"].tail(horizon).to_numpy(),
        0
    )


def exponential_smoothing_forecast(history, horizon):
    model = SimpleExpSmoothing(
        history,
        initialization_method="estimated"
    )

    fitted = model.fit(optimized=True)

    return np.maximum(
        np.asarray(fitted.forecast(horizon)),
        0
    )


results = []

for train_end in range(
    MIN_TRAIN,
    len(y) - HORIZON + 1,
    STEP
):
    history = y[:train_end]
    actual = y[train_end:train_end + HORIZON]
    dates = data["week_start"].iloc[:train_end]

    predictions = {
        "Prophet": prophet_forecast(
            history, dates, HORIZON
        ),
        "Exponential Smoothing": (
            exponential_smoothing_forecast(
                history, HORIZON
            )
        )
    }

    for model_name, forecast in predictions.items():
        for h in range(HORIZON):
            results.append({
                "model": model_name,
                "origin_week": str(
                    data["week_start"].iloc[train_end - 1]
                ),
                "horizon": h + 1,
                "actual": actual[h],
                "predicted": forecast[h],
                "absolute_error": abs(
                    actual[h] - forecast[h]
                )
            })


results_df = pd.DataFrame(results)

summary = (
    results_df.groupby("model")
    .agg(
        MAE=("absolute_error", "mean"),
        Forecasts=("absolute_error", "size")
    )
    .sort_values("MAE")
)

results_df.to_csv(
    REPORT_DIR / "W05_advanced_backtesting.csv",
    index=False
)

print("\n=== Advanced Model Comparison ===")
print(summary.round(3))

# Combine with existing baseline results
baseline = pd.read_csv(
    REPORT_DIR / "W05_backtesting_results.csv"
)

combined = pd.concat(
    [baseline, results_df],
    ignore_index=True
)

combined_summary = (
    combined.groupby("model")
    .agg(
        MAE=("absolute_error", "mean"),
        Forecasts=("absolute_error", "size")
    )
    .sort_values("MAE")
)

combined_summary.to_csv(
    REPORT_DIR / "W05_all_model_comparison.csv"
)

print("\n=== All Models Ranked by MAE ===")
print(combined_summary.round(3))
