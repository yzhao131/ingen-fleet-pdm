
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

df = pd.read_csv(
    CLEAN_DIR / "telemetry.csv",
    parse_dates=["timestamp"]
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True
)

EXPECTED_INTERVAL = pd.Timedelta(hours=6)

# ==========================================
# 1. Check timestamp gaps per robot
# ==========================================

df = df.sort_values(["unit_id", "timestamp"]).copy()

df["time_diff"] = (
    df.groupby("unit_id")["timestamp"].diff()
)

large_gaps = df[
    df["time_diff"] > EXPECTED_INTERVAL
].copy()

print("\n=== TELEMETRY GAP ANALYSIS ===")
print(f"Rows analyzed: {len(df):,}")
print(f"Gaps longer than 6 hours: {len(large_gaps):,}")

if not large_gaps.empty:
    print("\nGap duration summary:")
    print(large_gaps["time_diff"].describe())

    print("\nFirst 10 gaps:")
    print(
        large_gaps[
            ["unit_id", "timestamp", "time_diff"]
        ].head(10).to_string(index=False)
    )

# ==========================================
# 2. Check clock drift
# ==========================================

# Seconds elapsed since UTC midnight
seconds_of_day = (
    df["timestamp"].dt.hour * 3600
    + df["timestamp"].dt.minute * 60
    + df["timestamp"].dt.second
)

# Offset from nearest 6-hour grid point
grid_seconds = 6 * 3600

offset_seconds = (
    (seconds_of_day + grid_seconds / 2)
    % grid_seconds
    - grid_seconds / 2
)

df["clock_offset_minutes"] = (
    offset_seconds / 60
)

clock_summary = (
    df.groupby("unit_id")["clock_offset_minutes"]
    .agg(["median", "min", "max"])
    .reset_index()
)

clock_summary["abs_median_offset"] = (
    clock_summary["median"].abs()
)

drifted_robots = clock_summary[
    clock_summary["abs_median_offset"] > 1
].copy()

print("\n=== CLOCK DRIFT ANALYSIS ===")
print(f"Robots analyzed: {df['unit_id'].nunique()}")
print(
    f"Robots with median offset > 1 minute: "
    f"{len(drifted_robots)}"
)

print("\nDetected clock offsets:")
print(
    drifted_robots.sort_values(
        "abs_median_offset",
        ascending=False
    ).to_string(index=False)
)

# ==========================================
# 3. Save reports
# ==========================================

REPORT_DIR = BASE_DIR / "reports" / "W02"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

large_gaps.to_csv(
    REPORT_DIR / "telemetry_gaps.csv",
    index=False
)

clock_summary.to_csv(
    REPORT_DIR / "clock_drift_summary.csv",
    index=False
)

print("\nReports saved successfully!")
