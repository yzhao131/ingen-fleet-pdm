
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

df = pd.read_csv(
    REPORT_DIR / "W05_battery_lifecycle.csv",
    parse_dates=["timestamp"]
)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values(
    ["unit_id", "timestamp"]
).reset_index(drop=True)

THRESHOLDS = [80.0, 90.0]
N_BOOTSTRAP = 300
RANDOM_SEED = 42
MIN_OBSERVATIONS = 30

# Estimated from the pooled linear model
POOLED_SLOPE = -0.017455

# Require sufficient usage variation for
# reliable per-segment slope estimation
MIN_CYCLE_SPAN = 30
MIN_SEGMENT_DAYS = 30

rng = np.random.default_rng(RANDOM_SEED)

results = []

def crossing_date(
    intercept,
    slope,
    threshold,
    current_cycles,
    cycles_per_day,
    current_date
):
    if slope >= 0 or cycles_per_day <= 0:
        return pd.NaT

    crossing_cycles = (
        threshold - intercept
    ) / slope

    remaining_cycles = crossing_cycles - current_cycles

    if remaining_cycles <= 0:
        return current_date

    days = remaining_cycles / cycles_per_day

    # Prevent invalid / excessively distant dates
    if not np.isfinite(days) or days > 36500:
        return pd.NaT

    return current_date + pd.Timedelta(days=float(days))


for unit_id, unit_data in df.groupby("unit_id"):

    # Use latest observed battery lifecycle
    latest_segment = unit_data["battery_segment"].max()

    group = unit_data[
        unit_data["battery_segment"] == latest_segment
    ].sort_values("timestamp").copy()

    if len(group) < MIN_OBSERVATIONS:
        continue

    current = group.iloc[-1]
    current_date = current["timestamp"]
    current_capacity = float(current["battery_capacity_pct"])
    current_cycles = float(current["cycles_since_repair"])

    # Estimate recent usage rate using last 60 days
    recent_start = current_date - pd.Timedelta(days=60)

    recent = group[
        group["timestamp"] >= recent_start
    ]

    if len(recent) < 2:
        recent = group

    elapsed_days = (
        recent["timestamp"].iloc[-1]
        - recent["timestamp"].iloc[0]
    ).total_seconds() / 86400

    if elapsed_days <= 0:
        continue

    cycles_per_day = (
        recent["charge_cycles"].iloc[-1]
        - recent["charge_cycles"].iloc[0]
    ) / elapsed_days

    X = group[["cycles_since_repair"]].to_numpy()
    y = group["battery_capacity_pct"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    original_slope = float(model.coef_[0])

    cycle_span = (
        group["cycles_since_repair"].max()
        - group["cycles_since_repair"].min()
    )

    segment_days = (
        group["timestamp"].iloc[-1]
        - group["timestamp"].iloc[0]
    ).total_seconds() / 86400

    use_fallback = (
        cycle_span < MIN_CYCLE_SPAN
        or segment_days < MIN_SEGMENT_DAYS
        or original_slope >= 0
    )

    if use_fallback:
        slope = POOLED_SLOPE

    # Re-estimate the intercept while keeping
    # the pooled degradation slope fixed.
        intercept = float(
            np.mean(y - slope * X[:, 0])
        )

        model_source = "Pooled slope fallback"
    else:
        slope = original_slope
        intercept = float(model.intercept_)
        model_source = "Per-unit linear"

    # Residual bootstrap (conditional on fitted slope,
    # intercept and assumed future usage rate).
    fitted = intercept + slope * X[:, 0]
    residuals = y - fitted

    # Reduce autocorrelation by sampling one residual
    # per day rather than every 6-hour observation.
    group["residual"] = residuals
    group["day"] = group["timestamp"].dt.floor("D")

    daily_residuals = (
        group.groupby("day")["residual"]
        .mean()
        .to_numpy()
    )

    for threshold in THRESHOLDS:

        if current_capacity <= threshold:
            status = "Already below threshold"
            point_date = current_date
            p10_date = current_date
            p90_date = current_date

        else:
            point_date = crossing_date(
                intercept,
                slope,
                threshold,
                current_cycles,
                cycles_per_day,
                current_date
            )

            simulated_dates = []

            for _ in range(N_BOOTSTRAP):

                # Bootstrap a capacity offset as a
                # simple uncertainty sensitivity.
                noise = rng.choice(daily_residuals)

                simulated_date = crossing_date(
                    intercept + noise,
                    slope,
                    threshold,
                    current_cycles,
                    cycles_per_day,
                    current_date
                )

                if pd.notna(simulated_date):
                    simulated_dates.append(simulated_date)

            if pd.notna(point_date):
                status = "Forecasted"
            else:
                status = "Not estimable"

            if simulated_dates:
                date_values = pd.Series(simulated_dates)
                p10_date = date_values.quantile(0.10)
                p90_date = date_values.quantile(0.90)
            else:
                p10_date = pd.NaT
                p90_date = pd.NaT

        results.append({
            "unit_id": unit_id,
            "threshold_pct": threshold,
            "current_date": current_date,
            "current_capacity_pct": round(current_capacity, 3),
            "estimated_cycles_per_day": round(cycles_per_day, 4),
            "degradation_slope": round(slope, 6),
            "original_slope": round(original_slope, 6),
            "model_source": model_source,
            "cycle_span": int(cycle_span),
            "segment_days": round(segment_days, 2),
            "predicted_crossing_date": point_date,
            "crossing_date_p10": p10_date,
            "crossing_date_p90": p90_date,
            "status": status
        })


forecast_df = pd.DataFrame(results)

output_path = (
    REPORT_DIR / "W05_battery_replacement_forecast.csv"
)

forecast_df.to_csv(output_path, index=False)

print("=== Battery Replacement Forecast ===")
print("Total predictions:", len(forecast_df))

print("\n=== Status by Threshold ===")
print(
    forecast_df.groupby(
        ["threshold_pct", "status"]
    ).size().to_string()
)

for threshold in THRESHOLDS:
    subset = forecast_df[
        forecast_df["threshold_pct"] == threshold
    ]

    print(f"\n=== {threshold:.0f}% Threshold ===")

    display_columns = [
        "unit_id",
        "current_capacity_pct",
        "estimated_cycles_per_day",
        "degradation_slope",
        "predicted_crossing_date",
        "status"
    ]

    print(
        subset[display_columns]
        .sort_values("predicted_crossing_date")
        .head(10)
        .to_string(index=False)
    )

print("\nSaved:", output_path)
