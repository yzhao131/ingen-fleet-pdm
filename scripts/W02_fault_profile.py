
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
REPORT_DIR = BASE_DIR / "reports" / "W02"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

faults = pd.read_csv(
    CLEAN_DIR / "faults.csv",
    parse_dates=["occurred_at", "logged_at"]
)

maintenance = pd.read_csv(
    CLEAN_DIR / "maintenance.csv",
    parse_dates=["start_time", "end_time"]
)

robots = pd.read_csv(CLEAN_DIR / "robots.csv")

# ==========================================
# 1. Overall fault statistics
# ==========================================

print("\n=== OVERALL FAULT STATISTICS ===")

print("Total faults:", len(faults))

severe_faults = faults[
    faults["severity"] >= 3
].copy()

print("Severe faults (severity >= 3):", len(severe_faults))

print("\nFaults by severity:")
print(faults["severity"].value_counts().sort_index())

print("\nFaults by type:")
print(faults["fault_type"].value_counts())

print("\nSevere faults by type:")
print(severe_faults["fault_type"].value_counts())

# ==========================================
# 2. Faults by site
# ==========================================

faults_with_site = faults.merge(
    robots[["unit_id", "site_id"]],
    on="unit_id",
    how="left",
    validate="many_to_one"
)

print("\n=== FAULTS BY SITE ===")

site_summary = (
    faults_with_site.groupby("site_id")
    .agg(
        total_faults=("fault_id", "count"),
        severe_faults=(
            "severity",
            lambda x: (x >= 3).sum()
        )
    )
    .reset_index()
    .sort_values("severe_faults", ascending=False)
)

print(site_summary.to_string(index=False))

site_summary.to_csv(
    REPORT_DIR / "faults_by_site.csv",
    index=False
)

# ==========================================
# 3. Faults by robot
# ==========================================

print("\n=== TOP 10 ROBOTS BY SEVERE FAULTS ===")

robot_summary = (
    faults.groupby("unit_id")
    .agg(
        total_faults=("fault_id", "count"),
        severe_faults=(
            "severity",
            lambda x: (x >= 3).sum()
        )
    )
    .reset_index()
    .sort_values("severe_faults", ascending=False)
)

print(robot_summary.head(10).to_string(index=False))

robot_summary.to_csv(
    REPORT_DIR / "faults_by_robot.csv",
    index=False
)

# ==========================================
# 4. Maintenance statistics
# ==========================================

print("\n=== MAINTENANCE STATISTICS ===")

print("Total maintenance records:", len(maintenance))

maintenance["duration_hours"] = (
    maintenance["end_time"]
    - maintenance["start_time"]
).dt.total_seconds() / 3600

print("\nMaintenance duration (hours):")
print(maintenance["duration_hours"].describe())

# Count severe faults followed by maintenance
# for the same robot within 7 days.
# This is a descriptive association, not proof
# that the maintenance was caused by the fault.

matched_faults = 0

for _, fault in severe_faults.iterrows():

    matches = maintenance[
        (maintenance["unit_id"] == fault["unit_id"])
        & (maintenance["start_time"] >= fault["occurred_at"])
        & (
            maintenance["start_time"]
            <= fault["occurred_at"] + pd.Timedelta(days=7)
        )
    ]

    if not matches.empty:
        matched_faults += 1

print(
    "\nSevere faults followed by maintenance within 7 days:",
    matched_faults
)

print(
    "Association rate:",
    round(matched_faults / len(severe_faults) * 100, 2)
    if len(severe_faults) else 0,
    "%"
)

# ==========================================
# 5. Faults per robot-day
# ==========================================

print("\n=== FAULT BASE RATE ===")

START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-06-30", tz="UTC")

days = (END - START).days + 1
robot_days = len(robots) * days

print("Calendar days:", days)
print("Robot-days:", robot_days)

print(
    "Severe fault events per 1,000 robot-days:",
    round(len(severe_faults) / robot_days * 1000, 3)
)

print("\nReports saved successfully!")
