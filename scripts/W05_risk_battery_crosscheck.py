
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
W04 = ROOT / "reports" / "W04"
W05 = ROOT / "reports" / "W05"

# Week 4 historical validation risk
risk = pd.read_csv(
    W04 / "W04_validation_predictions.csv"
)

# Week 5 battery replacement forecast
battery = pd.read_csv(
    W05 / "W05_battery_replacement_forecast.csv"
)

# One historical risk summary per robot
risk_summary = (
    risk.groupby("unit_id")
    .agg(
        mean_failure_risk=("gb_leaf7_iter80", "mean"),
        max_failure_risk=("gb_leaf7_iter80", "max"),
        observed_failure_days=("label", "sum"),
        validation_days=("label", "size")
    )
    .reset_index()
)

# Select one battery record per robot
# Both threshold scenarios share current capacity
battery_current = (
    battery[battery["threshold_pct"] == 80.0]
    [["unit_id", "current_capacity_pct"]]
    .copy()
)

# Join by robot ID
comparison = risk_summary.merge(
    battery_current,
    on="unit_id",
    how="inner",
    validate="one_to_one"
)

# Define top 20% by historical mean risk
comparison["risk_rank"] = (
    comparison["mean_failure_risk"]
    .rank(method="first", ascending=False)
)

comparison["risk_group"] = "Other 80%"
comparison.loc[
    comparison["risk_rank"] <= 10,
    "risk_group"
] = "Top 20% risk"

# Save comparison
comparison.to_csv(
    W05 / "W05_risk_battery_crosscheck.csv",
    index=False
)

print("=== Risk and Battery Cross-Check ===")
print("Robots matched:", len(comparison))

print("\n=== Group Comparison ===")
print(
    comparison.groupby("risk_group").agg(
        robots=("unit_id", "count"),
        mean_risk=("mean_failure_risk", "mean"),
        mean_capacity=("current_capacity_pct", "mean"),
        observed_failure_days=("observed_failure_days", "sum")
    ).round(4).to_string()
)

print("\n=== Correlation ===")
print(
    "Mean historical risk vs current battery capacity:",
    round(
        comparison["mean_failure_risk"].corr(
            comparison["current_capacity_pct"]
        ),
        4
    )
)

print("\n=== Top 10 Historical Risk Robots ===")
print(
    comparison.sort_values(
        "mean_failure_risk", ascending=False
    )[[
        "unit_id",
        "mean_failure_risk",
        "current_capacity_pct",
        "observed_failure_days"
    ]].head(10).to_string(index=False)
)

print("\nSaved: W05_risk_battery_crosscheck.csv")
