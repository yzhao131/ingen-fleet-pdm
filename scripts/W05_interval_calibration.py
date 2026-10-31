
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

df = pd.read_csv(
    REPORT_DIR / "W05_backtesting_results.csv"
)

df = df[df["model"] == "Last Value"].copy()

df["origin_week"] = pd.to_datetime(
    df["origin_week"], utc=True
)

df = df.sort_values(
    ["origin_week", "horizon"]
).reset_index(drop=True)

# ------------------------------------------
# Split rolling origins chronologically
# ------------------------------------------

origins = sorted(df["origin_week"].unique())
calibration_origins = origins[:4]
evaluation_origins = origins[4:]

# Check whether calibration and evaluation
# forecast target weeks overlap.

calibration_targets = set()
evaluation_targets = set()

for _, row in df.iterrows():

    target_week = (
        row["origin_week"]
        + pd.Timedelta(weeks=int(row["horizon"]))
    )

    if row["origin_week"] in calibration_origins:
        calibration_targets.add(target_week)

    elif row["origin_week"] in evaluation_origins:
        evaluation_targets.add(target_week)

overlap = calibration_targets.intersection(
    evaluation_targets
)

print("\n=== Target Week Overlap Check ===")
print("Overlapping target weeks:", len(overlap))

if overlap:
    print("First overlapping weeks:")
    for week in sorted(overlap)[:10]:
        print(week)

calibration = df[
    df["origin_week"].isin(calibration_origins)
].copy()

evaluation = df[
    df["origin_week"].isin(evaluation_origins)
].copy()


# ------------------------------------------
# Remove evaluation targets that overlap
# with calibration targets
# ------------------------------------------

evaluation["target_week"] = (
    evaluation["origin_week"]
    + pd.to_timedelta(
        evaluation["horizon"] * 7,
        unit="D"
    )
)

before = len(evaluation)

evaluation = evaluation[
    ~evaluation["target_week"].isin(
        calibration_targets
    )
].copy()

after = len(evaluation)

print("\n=== Non-Overlapping Evaluation ===")
print("Evaluation forecasts before:", before)
print("Evaluation forecasts after:", after)
print("Removed forecasts:", before - after)

print(
    "Remaining horizons:",
    sorted(evaluation["horizon"].unique().tolist())
)


print("=== Calibration Split ===")
print("Calibration origins:", len(calibration_origins))
print("Evaluation origins:", len(evaluation_origins))
print("Calibration forecasts:", len(calibration))
print("Evaluation forecasts:", len(evaluation))

# ------------------------------------------
# Calculate signed forecast errors
# ------------------------------------------

calibration["error"] = (
    calibration["actual"]
    - calibration["predicted"]
)

evaluation["error"] = (
    evaluation["actual"]
    - evaluation["predicted"]
)

# ------------------------------------------
# Horizon-specific interval calibration
# ------------------------------------------

interval_rows = []

for horizon in range(1, 13):

    # Pool neighboring horizons to increase
    # calibration sample size.
    nearby = calibration[
        calibration["horizon"].between(
            max(1, horizon - 2),
            min(12, horizon + 2)
        )
    ]

    errors = nearby["error"].to_numpy()

    lower_error = np.quantile(errors, 0.10)
    upper_error = np.quantile(errors, 0.90)

    horizon_eval = evaluation[
        evaluation["horizon"] == horizon
    ].copy()

    horizon_eval["lower"] = np.maximum(
        0,
        horizon_eval["predicted"] + lower_error
    )

    horizon_eval["upper"] = np.maximum(
        0,
        horizon_eval["predicted"] + upper_error
    )

    horizon_eval["covered"] = (
        (horizon_eval["actual"] >= horizon_eval["lower"])
        & (horizon_eval["actual"] <= horizon_eval["upper"])
    )

    horizon_eval["interval_width"] = (
        horizon_eval["upper"]
        - horizon_eval["lower"]
    )

    interval_rows.append({
        "horizon": horizon,
        "calibration_samples": len(errors),
        "evaluation_samples": len(horizon_eval),
        "lower_error": lower_error,
        "upper_error": upper_error,
        "empirical_coverage": horizon_eval["covered"].mean(),
        "average_interval_width": horizon_eval["interval_width"].mean()
    })

interval_summary = pd.DataFrame(interval_rows)

# ------------------------------------------
# Evaluate overall coverage
# ------------------------------------------

valid = interval_summary.dropna(
    subset=["empirical_coverage"]
)

overall_coverage = np.average(
    valid["empirical_coverage"],
    weights=valid["evaluation_samples"]
)

overall_width = np.average(
    valid["average_interval_width"],
    weights=valid["evaluation_samples"]
)

overall_width = (
    interval_summary["average_interval_width"].mean()
)

print("\n=== Horizon-Specific Intervals ===")
print(interval_summary.round(3).to_string(index=False))

print("\n=== Overall Evaluation ===")
print("Target coverage: 0.80")
print("Empirical coverage:", round(overall_coverage, 3))
print("Average interval width:", round(overall_width, 3))

# ------------------------------------------
# Save results
# ------------------------------------------

interval_summary.to_csv(
    REPORT_DIR / "W05_calibrated_interval_summary.csv",
    index=False
)

print("\nSaved interval calibration summary.")
