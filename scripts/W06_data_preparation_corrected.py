
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/causal_fixed/raw")
OUTPUT_DIR = Path("reports/W06_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

robots = pd.read_csv(DATA_DIR / "robots.csv")
missions = pd.read_csv(DATA_DIR / "missions.csv")
telemetry = pd.read_csv(DATA_DIR / "telemetry.csv")
faults = pd.read_csv(DATA_DIR / "faults.csv")
maintenance = pd.read_csv(DATA_DIR / "maintenance.csv")

# Parse timestamps consistently.
telemetry["timestamp"] = pd.to_datetime(
    telemetry["timestamp"], utc=True
)
missions["start_time"] = pd.to_datetime(
    missions["start_time"], utc=True
)
missions["end_time"] = pd.to_datetime(
    missions["end_time"], utc=True
)
faults["occurred_at"] = pd.to_datetime(
    faults["occurred_at"], utc=True
)
maintenance["start_time"] = pd.to_datetime(
    maintenance["start_time"], utc=True
)

# Use a common observation end for robots without repairs.
observation_end = min(
    telemetry["timestamp"].max(),
    missions["end_time"].max(),
)

first_repair = (
    maintenance.groupby("unit_id")["start_time"]
    .min()
    .rename("first_repair_time")
)

features = robots.merge(
    first_repair, on="unit_id", how="left"
)

features["cutoff_time"] = features[
    "first_repair_time"
].fillna(observation_end)

# Join each event with its robot-specific cutoff.
cutoffs = features[["unit_id", "cutoff_time"]]

t = telemetry.merge(cutoffs, on="unit_id", how="inner")
t = t[t["timestamp"] < t["cutoff_time"]].copy()

m = missions.merge(cutoffs, on="unit_id", how="inner")
m = m[
    (m["mission_status"] == "completed")
    & (m["end_time"] < m["cutoff_time"])
].copy()

f = faults.merge(cutoffs, on="unit_id", how="inner")
f = f[f["occurred_at"] < f["cutoff_time"]].copy()

# Mission utilization features.
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

# Telemetry features from pre-repair observations only.
telemetry_avg = (
    t.groupby("unit_id")
    .agg(
        avg_battery_capacity=("battery_capacity_pct", "mean"),
        avg_motor_current=("motor_current_a", "mean"),
        avg_temperature=("temperature_c", "mean"),
        telemetry_count=("timestamp", "size"),
    )
    .reset_index()
)

telemetry_latest = (
    t.sort_values(["unit_id", "timestamp"])
    .groupby("unit_id")
    .tail(1)[
        [
            "unit_id",
            "battery_capacity_pct",
            "charge_cycles",
            "wheel_wear_pct",
        ]
    ]
    .rename(
        columns={
            "battery_capacity_pct": "latest_battery_capacity",
            "charge_cycles": "latest_charge_cycles",
            "wheel_wear_pct": "latest_wheel_wear",
        }
    )
)

telemetry_features = telemetry_avg.merge(
    telemetry_latest, on="unit_id", how="left"
)

fault_features = (
    f.groupby("unit_id")
    .agg(fault_count=("fault_id", "nunique"))
    .reset_index()
)

features = (
    features
    .merge(mission_features, on="unit_id", how="left")
    .merge(telemetry_features, on="unit_id", how="left")
    .merge(fault_features, on="unit_id", how="left")
)

for col in ["mission_count", "fault_count", "telemetry_count"]:
    features[col] = features[col].fillna(0).astype(int)

features["faults_per_1000_hours"] = np.where(
    features["total_operating_hours"] > 0,
    1000 * features["fault_count"]
    / features["total_operating_hours"],
    np.nan,
)

# Explicitly label the dataset as pre-repair.
features["feature_window"] = "pre_first_repair"

assert len(features) == len(robots)
assert features["unit_id"].is_unique
assert features["telemetry_count"].min() > 0
assert features["total_operating_hours"].min() > 0

output_path = (
    OUTPUT_DIR / "W06_robot_features_pre_repair.csv"
)
features.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Shape:", features.shape)
print("\nMissing values:")
print(features.isna().sum().to_string())

print("\nFeature summary:")
print(
    features[
        [
            "telemetry_count",
            "mission_count",
            "total_operating_hours",
            "fault_count",
            "faults_per_1000_hours",
        ]
    ].describe().to_string()
)
