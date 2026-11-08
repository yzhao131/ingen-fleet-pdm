
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
DATA_DIR = Path("data/causal_fixed/raw")
OUTPUT_DIR = Path("reports/W06")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
robots = pd.read_csv(DATA_DIR / "robots.csv")
missions = pd.read_csv(DATA_DIR / "missions.csv")
telemetry = pd.read_csv(DATA_DIR / "telemetry.csv")
faults = pd.read_csv(DATA_DIR / "faults.csv")
maintenance = pd.read_csv(DATA_DIR / "maintenance.csv")

print("Loaded datasets successfully.")

# --------------------------------------------------
# 3. Missions: utilization features
# --------------------------------------------------
completed = missions[
    missions["mission_status"] == "completed"
].copy()

mission_features = (
    completed.groupby("unit_id")
    .agg(
        mission_count=("mission_id", "nunique"),
        total_operating_hours=("operating_hours", "sum"),
        avg_mission_hours=("operating_hours", "mean"),
        total_distance_km=("distance_km", "sum"),
    )
    .reset_index()
)

# --------------------------------------------------
# 4. Telemetry: equipment health features
# --------------------------------------------------
telemetry["timestamp"] = pd.to_datetime(
    telemetry["timestamp"], utc=True
)

telemetry = telemetry.sort_values(
    ["unit_id", "timestamp"]
)

telemetry_avg = (
    telemetry.groupby("unit_id")
    .agg(
        avg_battery_capacity=("battery_capacity_pct", "mean"),
        avg_motor_current=("motor_current_a", "mean"),
        avg_temperature=("temperature_c", "mean"),
    )
    .reset_index()
)

telemetry_latest = (
    telemetry.groupby("unit_id")
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

# --------------------------------------------------
# 5. Fault history
# --------------------------------------------------
fault_features = (
    faults.groupby("unit_id")
    .agg(fault_count=("fault_id", "nunique"))
    .reset_index()
)

# --------------------------------------------------
# 6. Maintenance history
# --------------------------------------------------
maintenance_features = (
    maintenance.groupby("unit_id")
    .agg(
        maintenance_count=("maintenance_id", "nunique")
    )
    .reset_index()
)

# --------------------------------------------------
# 7. Merge into one robot-level table
# --------------------------------------------------
features = robots.merge(
    mission_features, on="unit_id", how="left"
)

features = features.merge(
    telemetry_features, on="unit_id", how="left"
)

features = features.merge(
    fault_features, on="unit_id", how="left"
)

features = features.merge(
    maintenance_features, on="unit_id", how="left"
)

# Zero is appropriate for robots without recorded events.
count_cols = [
    "mission_count",
    "fault_count",
    "maintenance_count",
]
features[count_cols] = features[count_cols].fillna(0)

# --------------------------------------------------
# 8. Normalize historical fault frequency
# --------------------------------------------------
features["faults_per_1000_hours"] = np.where(
    features["total_operating_hours"] > 0,
    features["fault_count"]
    / features["total_operating_hours"] * 1000,
    np.nan,
)

# --------------------------------------------------
# 9. Data quality checks
# --------------------------------------------------
assert features["unit_id"].is_unique
assert len(features) == len(robots)

print("\nFeature table shape:", features.shape)
print("\nMissing values:")
print(features.isna().sum())

print("\nFeature preview:")
print(features.head().to_string(index=False))

# --------------------------------------------------
# 10. Export
# --------------------------------------------------
output_path = OUTPUT_DIR / "W06_robot_features.csv"
features.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")
