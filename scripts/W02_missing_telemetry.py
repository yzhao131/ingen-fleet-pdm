
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
REPORT_DIR = BASE_DIR / "reports" / "W02"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

telemetry = pd.read_csv(
    CLEAN_DIR / "telemetry.csv",
    parse_dates=["timestamp"]
)

missions = pd.read_csv(
    CLEAN_DIR / "missions.csv",
    parse_dates=["start_time", "end_time"]
)

maintenance = pd.read_csv(
    CLEAN_DIR / "maintenance.csv",
    parse_dates=["start_time", "end_time"]
)

# Ensure UTC timezone
for df, columns in [
    (telemetry, ["timestamp"]),
    (missions, ["start_time", "end_time"]),
    (maintenance, ["start_time", "end_time"]),
]:
    for col in columns:
        df[col] = pd.to_datetime(df[col], utc=True)

EXPECTED_INTERVAL = pd.Timedelta(hours=6)

# Expected observation period from Week 1
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-06-30 18:00:00", tz="UTC")

expected_times = pd.date_range(
    start=START,
    end=END,
    freq="6h"
)

results = []

# Identify missing timestamps for each robot
for unit_id, group in telemetry.groupby("unit_id"):

    observed = pd.DatetimeIndex(group["timestamp"])

    missing = expected_times.difference(observed)

    for missing_time in missing:

        # Check whether any mission overlaps the expected
        # six-hour interval [missing_time, missing_time + 6h)
        window_end = missing_time + EXPECTED_INTERVAL

        unit_missions = missions[
            missions["unit_id"] == unit_id
        ]

        has_mission = (
            (unit_missions["start_time"] < window_end)
            & (unit_missions["end_time"] > missing_time)
        ).any()

        unit_maintenance = maintenance[
            maintenance["unit_id"] == unit_id
        ]

        has_maintenance = (
            (unit_maintenance["start_time"] < window_end)
            & (unit_maintenance["end_time"] > missing_time)
        ).any()

        # Classification based on available evidence
        if has_mission:
            status = "possible_connectivity_gap"
        elif has_maintenance:
            status = "maintenance_overlap"
        else:
            status = "unknown"

        results.append({
            "unit_id": unit_id,
            "missing_timestamp": missing_time,
            "has_mission": bool(has_mission),
            "has_maintenance": bool(has_maintenance),
            "status": status,
        })

missing_df = pd.DataFrame(results, columns=[
    "unit_id", "missing_timestamp",
    "has_mission", "has_maintenance", "status"
])

print("\n=== MISSING TELEMETRY ANALYSIS ===")
print("Expected readings per robot:", len(expected_times))
print("Total expected readings:",
      len(expected_times) * telemetry["unit_id"].nunique())
print("Observed readings:", len(telemetry))
print("Missing readings:", len(missing_df))

print("\nMissing readings by status:")
print(missing_df["status"].value_counts())

print("\nMissing readings by robot:")
print(
    missing_df["unit_id"]
    .value_counts()
    .head(10)
)

missing_df.to_csv(
    REPORT_DIR / "missing_telemetry.csv",
    index=False
)

print("\nSaved: reports/W02/missing_telemetry.csv")
