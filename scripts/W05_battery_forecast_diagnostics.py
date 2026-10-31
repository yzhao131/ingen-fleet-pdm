
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"
RAW_DIR = ROOT / "data" / "causal_fixed" / "raw"

df = pd.read_csv(
    REPORT_DIR / "W05_battery_lifecycle.csv",
    parse_dates=["timestamp"]
)

forecast = pd.read_csv(
    REPORT_DIR / "W05_battery_replacement_forecast.csv"
)

maintenance = pd.read_csv(
    RAW_DIR / "maintenance.csv",
    parse_dates=["start_time", "end_time"]
)

# Check unusual degradation slopes
slopes = forecast[
    forecast["threshold_pct"] == 80
].copy()

print("=== Degradation Slope Summary ===")
print(
    slopes["degradation_slope"]
    .describe()
    .round(6)
)

print("\n=== Unusual Slopes ===")
unusual = slopes[
    (slopes["degradation_slope"] < -0.025)
    | (slopes["degradation_slope"] > -0.010)
]

print(
    unusual[
        [
            "unit_id",
            "current_capacity_pct",
            "degradation_slope",
            "estimated_cycles_per_day"
        ]
    ].to_string(index=False)
)

# Inspect R004
unit_id = "R004"

robot = df[
    df["unit_id"] == unit_id
].sort_values("timestamp")

print(f"\n=== {unit_id} Lifecycle Summary ===")

summary = robot.groupby("battery_segment").agg(
    observations=("timestamp", "size"),
    start_date=("timestamp", "min"),
    end_date=("timestamp", "max"),
    min_cycles=("cycles_since_repair", "min"),
    max_cycles=("cycles_since_repair", "max"),
    first_capacity=("battery_capacity_pct", "first"),
    last_capacity=("battery_capacity_pct", "last")
)

print(summary.round(3).to_string())

print(f"\n=== {unit_id} Maintenance History ===")

robot_maintenance = maintenance[
    maintenance["unit_id"] == unit_id
]

print(
    robot_maintenance[
        ["start_time", "end_time", "reason"]
    ].to_string(index=False)
)

print(f"\n=== {unit_id} Latest Segment Sample ===")

latest_segment = robot["battery_segment"].max()

latest = robot[
    robot["battery_segment"] == latest_segment
]

print("\nFirst 5 observations:")
print(
    latest[
        [
            "timestamp",
            "battery_capacity_pct",
            "charge_cycles",
            "cycles_since_repair"
        ]
    ].head().to_string(index=False)
)

print("\nLast 5 observations:")
print(
    latest[
        [
            "timestamp",
            "battery_capacity_pct",
            "charge_cycles",
            "cycles_since_repair"
        ]
    ].tail().to_string(index=False)
)
