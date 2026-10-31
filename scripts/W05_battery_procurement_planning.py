
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

# ---------------------------------------
# Planning assumptions
# ---------------------------------------

LEAD_TIME_DAYS = 30
SAFETY_BUFFER_DAYS = 14

TOTAL_ADVANCE_DAYS = (
    LEAD_TIME_DAYS + SAFETY_BUFFER_DAYS
)

# ---------------------------------------
# Load battery forecasts
# ---------------------------------------

forecast = pd.read_csv(
    REPORT_DIR / "W05_battery_replacement_forecast.csv"
)

forecast = forecast[
    forecast["status"] == "Forecasted"
].copy()

date_columns = [
    "predicted_crossing_date",
    "crossing_date_p10",
    "crossing_date_p90"
]

for col in date_columns:
    forecast[col] = pd.to_datetime(
        forecast[col], utc=True
    )

# ---------------------------------------
# Calculate procurement deadlines
# ---------------------------------------

advance = pd.Timedelta(
    days=TOTAL_ADVANCE_DAYS
)

forecast["order_deadline"] = (
    forecast["predicted_crossing_date"]
    - advance
)

forecast["early_order_deadline"] = (
    forecast["crossing_date_p10"]
    - advance
)

forecast["order_month"] = (
    forecast["order_deadline"]
    .dt.tz_localize(None)
    .dt.to_period("M")
)

forecast["early_order_month"] = (
    forecast["early_order_deadline"]
    .dt.tz_localize(None)
    .dt.to_period("M")
)

# ---------------------------------------
# Unit-level procurement plan
# ---------------------------------------

unit_plan = forecast[[
    "unit_id",
    "threshold_pct",
    "predicted_crossing_date",
    "crossing_date_p10",
    "crossing_date_p90",
    "order_deadline",
    "early_order_deadline",
    "order_month",
    "early_order_month",
    "model_source"
]].copy()

unit_plan.to_csv(
    REPORT_DIR / "W05_battery_procurement_units.csv",
    index=False
)

# ---------------------------------------
# Monthly procurement quantities
# ---------------------------------------

rows = []

for threshold in [80.0, 90.0]:

    subset = forecast[
        forecast["threshold_pct"] == threshold
    ]

    for scenario, month_col in [
        ("Point forecast", "order_month"),
        ("Early crossing", "early_order_month")
    ]:

        counts = (
            subset.groupby(month_col)["unit_id"]
            .nunique()
        )

        for month, count in counts.items():
            rows.append({
                "threshold_pct": threshold,
                "scenario": scenario,
                "order_month": str(month),
                "battery_units_to_order": int(count)
            })

monthly = pd.DataFrame(rows).sort_values(
    ["threshold_pct", "scenario", "order_month"]
)

monthly.to_csv(
    REPORT_DIR / "W05_battery_procurement_monthly.csv",
    index=False
)

# ---------------------------------------
# Summary
# ---------------------------------------

print("=== Battery Procurement Planning ===")
print("Lead time:", LEAD_TIME_DAYS, "days")
print("Safety buffer:", SAFETY_BUFFER_DAYS, "days")
print("Total advance:", TOTAL_ADVANCE_DAYS, "days")

for threshold in [80.0, 90.0]:

    subset = monthly[
        (monthly["threshold_pct"] == threshold)
        & (monthly["scenario"] == "Point forecast")
    ]

    print(f"\n=== {threshold:.0f}% Threshold ===")

    print(
        "Total battery units:",
        subset["battery_units_to_order"].sum()
    )

    print("First 10 procurement months:")

    print(
        subset[[
            "order_month",
            "battery_units_to_order"
        ]].head(10).to_string(index=False)
    )

print("\nSaved:")
print("W05_battery_procurement_units.csv")
print("W05_battery_procurement_monthly.csv")
