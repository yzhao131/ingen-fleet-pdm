
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/causal_fixed/raw")
OUT = Path("reports/W06_corrected")
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 14

robots = pd.read_csv(DATA_DIR / "robots.csv")
telemetry = pd.read_csv(DATA_DIR / "telemetry.csv")
missions = pd.read_csv(DATA_DIR / "missions.csv")
faults = pd.read_csv(DATA_DIR / "faults.csv")
maintenance = pd.read_csv(DATA_DIR / "maintenance.csv")

for frame, cols in [
    (telemetry, ["timestamp"]),
    (missions, ["start_time", "end_time"]),
    (faults, ["occurred_at"]),
    (maintenance, ["start_time"]),
]:
    for col in cols:
        frame[col] = pd.to_datetime(
            frame[col], format="mixed", utc=True
        )

# Establish each robot's first observed telemetry timestamp.
first_seen = (
    telemetry.groupby("unit_id")["timestamp"]
    .min()
    .rename("window_start")
)

first_fault = (
    faults.groupby("unit_id")["occurred_at"]
    .min()
    .rename("first_fault")
)

first_repair = (
    maintenance.groupby("unit_id")["start_time"]
    .min()
    .rename("first_repair")
)

eligibility = (
    robots[["unit_id"]]
    .merge(first_seen, on="unit_id", how="left")
    .merge(first_fault, on="unit_id", how="left")
    .merge(first_repair, on="unit_id", how="left")
)

eligibility["window_end"] = (
    eligibility["window_start"]
    + pd.Timedelta(days=WINDOW_DAYS)
)

eligibility["fault_free"] = (
    eligibility["first_fault"].isna()
    | (eligibility["first_fault"]
       >= eligibility["window_end"])
)

eligibility["repair_free"] = (
    eligibility["first_repair"].isna()
    | (eligibility["first_repair"]
       >= eligibility["window_end"])
)

eligibility["eligible"] = (
    eligibility["window_start"].notna()
    & eligibility["fault_free"]
    & eligibility["repair_free"]
)

eligibility["exclusion_reason"] = np.select(
    [
        eligibility["window_start"].isna(),
        ~eligibility["fault_free"],
        ~eligibility["repair_free"],
    ],
    [
        "missing_telemetry",
        "fault_during_window",
        "repair_during_window",
    ],
    default="eligible",
)

eligibility.to_csv(
    OUT / "W06_fixed_window_eligibility.csv",
    index=False,
)

eligible = eligibility.loc[
    eligibility["eligible"],
    ["unit_id", "window_start", "window_end"]
]

# Keep telemetry within the 14-day window.
t = telemetry.merge(eligible, on="unit_id", how="inner")

t = t[
    (t["timestamp"] >= t["window_start"])
    & (t["timestamp"] < t["window_end"])
].copy()

t = t.sort_values(["unit_id", "timestamp"])

telemetry_features = (
    t.groupby("unit_id")
    .agg(
        telemetry_count=("timestamp", "size"),
        avg_battery_capacity=(
            "battery_capacity_pct", "mean"
        ),
        avg_motor_current=("motor_current_a", "mean"),
        avg_temperature=("temperature_c", "mean"),
        latest_battery_capacity=(
            "battery_capacity_pct", "last"
        ),
        latest_wheel_wear=("wheel_wear_pct", "last"),
        first_charge_cycles=("charge_cycles", "first"),
        last_charge_cycles=("charge_cycles", "last"),
    )
    .reset_index()
)

telemetry_features["charge_cycle_delta"] = (
    telemetry_features["last_charge_cycles"]
    - telemetry_features["first_charge_cycles"]
)

# Only count missions completed within the window.
m = missions.merge(eligible, on="unit_id", how="inner")

m = m[
    (m["mission_status"] == "completed")
    & (m["start_time"] >= m["window_start"])
    & (m["end_time"] < m["window_end"])
].copy()

mission_features = (
    m.groupby("unit_id")
    .agg(
        mission_count=("mission_id", "nunique"),
        total_operating_hours=("operating_hours", "sum"),
        avg_mission_hours=("operating_hours", "mean"),
        total_distance_km=("distance_km", "sum"),
    )
    .reset_index()
)

features = (
    robots
    .merge(eligible, on="unit_id", how="inner")
    .merge(telemetry_features, on="unit_id", how="left")
    .merge(mission_features, on="unit_id", how="left")
)

features["charge_cycles_per_100_hours"] = (
    features["charge_cycle_delta"]
    / features["total_operating_hours"] * 100
)

features["distance_per_operating_hour"] = (
    features["total_distance_km"]
    / features["total_operating_hours"]
)

features["feature_window_days"] = WINDOW_DAYS

output = OUT / "W06_robot_features_14d.csv"
features.to_csv(output, index=False)

print("\n=== ELIGIBILITY ===")
print(eligibility["exclusion_reason"].value_counts())

print("\n=== FEATURE TABLE ===")
print("Shape:", features.shape)

print("\n=== MISSING VALUES ===")
print(features.isna().sum().to_string())

print("\n=== COVERAGE ===")
print(
    features[
        [
            "telemetry_count",
            "mission_count",
            "total_operating_hours",
            "charge_cycles_per_100_hours",
        ]
    ].describe().to_string()
)

print("\nSaved:", output)
