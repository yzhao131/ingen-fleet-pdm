
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "causal_fixed" / "raw"

telemetry = pd.read_csv(
    DATA_DIR / "telemetry.csv",
    parse_dates=["timestamp"]
)

maintenance = pd.read_csv(
    DATA_DIR / "maintenance.csv",
    parse_dates=["start_time", "end_time"]
)

# Sort chronologically for each robot
telemetry = telemetry.sort_values(
    ["unit_id", "timestamp"]
).reset_index(drop=True)

# Remove exact duplicate observations for analysis
telemetry = telemetry.drop_duplicates(
    subset=["unit_id", "timestamp"],
    keep="first"
)

print("=== Battery Dataset Overview ===")
print("Telemetry records:", len(telemetry))
print("Unique robots:", telemetry["unit_id"].nunique())
print("Start date:", telemetry["timestamp"].min())
print("End date:", telemetry["timestamp"].max())

print("\n=== Battery Feature Summary ===")
print(
    telemetry[
        [
            "battery_capacity_pct",
            "charge_cycles",
            "temperature_c"
        ]
    ].describe().round(2)
)

# Changes between consecutive observations
telemetry["capacity_change"] = (
    telemetry.groupby("unit_id")[
        "battery_capacity_pct"
    ].diff()
)

telemetry["cycle_change"] = (
    telemetry.groupby("unit_id")[
        "charge_cycles"
    ].diff()
)

# Flag potential resets, not confirmed replacements
capacity_increases = telemetry[
    telemetry["capacity_change"] > 2
]

cycle_resets = telemetry[
    telemetry["cycle_change"] < 0
]

print("\n=== Potential Battery Resets ===")
print(
    "Capacity increases > 2 percentage points:",
    len(capacity_increases)
)
print(
    "Charge cycle decreases:",
    len(cycle_resets)
)

print("\n=== Maintenance Reasons ===")
print(
    maintenance["reason"]
    .value_counts()
    .to_string()
)

print("\n=== Battery Maintenance Events ===")
battery_maintenance = maintenance[
    maintenance["reason"] == "battery"
]

print("Battery maintenance count:", len(battery_maintenance))

print("\n=== Per-Robot Capacity Change ===")

robot_summary = (
    telemetry.groupby("unit_id")
    .agg(
        first_capacity=("battery_capacity_pct", "first"),
        last_capacity=("battery_capacity_pct", "last"),
        first_cycles=("charge_cycles", "first"),
        last_cycles=("charge_cycles", "last")
    )
)

robot_summary["capacity_change"] = (
    robot_summary["last_capacity"]
    - robot_summary["first_capacity"]
)

print(robot_summary.head(10).round(2).to_string())

print("\n=== Overall Capacity Change ===")
print(
    robot_summary["capacity_change"]
    .describe()
    .round(2)
)
