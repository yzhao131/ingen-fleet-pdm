
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "causal_fixed" / "raw"
OUT_DIR = ROOT / "reports" / "W05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

telemetry = pd.read_csv(
    RAW_DIR / "telemetry.csv",
    parse_dates=["timestamp"]
)

maintenance = pd.read_csv(
    RAW_DIR / "maintenance.csv",
    parse_dates=["start_time", "end_time"]
)

telemetry["timestamp"] = pd.to_datetime(
    telemetry["timestamp"], utc=True
)

maintenance["end_time"] = pd.to_datetime(
    maintenance["end_time"], utc=True
)

# Remove duplicate telemetry observations
telemetry = (
    telemetry.sort_values(["unit_id", "timestamp"])
    .drop_duplicates(["unit_id", "timestamp"])
    .reset_index(drop=True)
)

# Battery repairs change the battery-wear state
battery_repairs = (
    maintenance[maintenance["reason"] == "battery"]
    [["unit_id", "end_time"]]
    .sort_values(["unit_id", "end_time"])
    .copy()
)

battery_repairs = battery_repairs.rename(
    columns={"end_time": "repair_end"}
)

# Assign a lifecycle segment to each observation
# A new segment starts after battery repair completion.
telemetry["battery_segment"] = 0
telemetry["last_battery_repair"] = pd.NaT

for unit_id, group in telemetry.groupby("unit_id"):
    repairs = battery_repairs.loc[
        battery_repairs["unit_id"] == unit_id,
        "repair_end"
    ].to_numpy()

    timestamps = group["timestamp"].to_numpy()

    segment = np.searchsorted(
        repairs,
        timestamps,
        side="right"
    )

    telemetry.loc[group.index, "battery_segment"] = segment

# Calculate cycles relative to the first observation
# within each battery lifecycle segment.
telemetry["segment_start_cycles"] = (
    telemetry.groupby(["unit_id", "battery_segment"])
    ["charge_cycles"]
    .transform("first")
)

telemetry["cycles_since_repair"] = (
    telemetry["charge_cycles"]
    - telemetry["segment_start_cycles"]
)

telemetry["segment_start_time"] = (
    telemetry.groupby(["unit_id", "battery_segment"])
    ["timestamp"]
    .transform("first")
)

telemetry["days_since_segment_start"] = (
    telemetry["timestamp"]
    - telemetry["segment_start_time"]
).dt.total_seconds() / 86400

output = OUT_DIR / "W05_battery_lifecycle.csv"
telemetry.to_csv(output, index=False)

print("=== Battery Lifecycle Dataset ===")
print("Records:", len(telemetry))
print("Robots:", telemetry["unit_id"].nunique())
print(
    "Battery maintenance events:",
    len(battery_repairs)
)

segments = (
    telemetry.groupby("unit_id")["battery_segment"]
    .nunique()
)

print("Total observed lifecycle segments:", segments.sum())
print(
    "Robots with multiple segments:",
    (segments > 1).sum()
)

print("\n=== Cycles Since Repair ===")
print(
    telemetry["cycles_since_repair"]
    .describe()
    .round(2)
)

print("\n=== Sample ===")
print(
    telemetry[
        [
            "unit_id",
            "timestamp",
            "battery_segment",
            "battery_capacity_pct",
            "charge_cycles",
            "cycles_since_repair"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:", output)
