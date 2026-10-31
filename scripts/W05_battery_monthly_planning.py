
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

forecast = pd.read_csv(
    REPORT_DIR / "W05_battery_replacement_forecast.csv",
    parse_dates=["predicted_crossing_date"]
)

# Simulation ends June 30, 2026.
# Forecast planning starts July 2026.
START_MONTH = pd.Period("2026-07", freq="M")
END_MONTH = pd.Period("2031-12", freq="M")

months = pd.period_range(
    START_MONTH,
    END_MONTH,
    freq="M"
)

planning_rows = []

for threshold in [80.0, 90.0]:

    subset = forecast[
        (forecast["threshold_pct"] == threshold)
        & (forecast["status"] == "Forecasted")
    ].copy()

    subset["replacement_month"] = (
        pd.to_datetime(
            subset["predicted_crossing_date"],
            utc=True
        )
        .dt.tz_localize(None)
        .dt.to_period("M")
    )

    monthly_counts = (
        subset.groupby("replacement_month")
        ["unit_id"]
        .nunique()
    )

    for month in months:

        count = int(monthly_counts.get(month, 0))

        planning_rows.append({
            "month": str(month),
            "threshold_pct": threshold,
            "expected_replacements": count,
            "battery_units_required": count,
            "scenario": (
                "Primary 80%"
                if threshold == 80.0
                else "Sensitivity 90%"
            )
        })

planning = pd.DataFrame(planning_rows)

output_path = (
    REPORT_DIR / "W05_monthly_battery_planning.csv"
)

planning.to_csv(output_path, index=False)

print("=== Monthly Battery Replacement Planning ===")

for threshold in [80.0, 90.0]:

    subset = planning[
        planning["threshold_pct"] == threshold
    ]

    nonzero = subset[
        subset["expected_replacements"] > 0
    ]

    print(f"\n=== {threshold:.0f}% Threshold ===")

    print(
        "Total replacements within planning horizon:",
        int(subset["expected_replacements"].sum())
    )

    print("First 12 nonzero months:")

    print(
        nonzero[
            [
                "month",
                "expected_replacements",
                "battery_units_required"
            ]
        ].head(12).to_string(index=False)
    )

print("\nSaved:", output_path)
