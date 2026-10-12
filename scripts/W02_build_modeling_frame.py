
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

robots = pd.read_csv(CLEAN_DIR / "robots.csv")

faults = pd.read_csv(CLEAN_DIR / "faults.csv")
maintenance = pd.read_csv(CLEAN_DIR / "maintenance.csv")

faults["occurred_at"] = pd.to_datetime(
    faults["occurred_at"], utc=True
)

for col in ["start_time", "end_time"]:
    maintenance[col] = pd.to_datetime(
        maintenance[col], utc=True
    )

# ==========================================
# 1. Define observation period
# ==========================================

START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-06-30", tz="UTC")

dates = pd.date_range(
    START, END, freq="D"
)

# Create one row per robot-day
frame = pd.MultiIndex.from_product(
    [robots["unit_id"], dates],
    names=["unit_id", "date"]
).to_frame(index=False)

print("Initial robot-days:", len(frame))

# ==========================================
# 2. Exclude maintenance days
# ==========================================

frame["during_maintenance"] = False

for row in maintenance.itertuples(index=False):
    mask = (
        (frame["unit_id"] == row.unit_id)
        & (frame["date"] >= row.start_time)
        & (frame["date"] < row.end_time)
    )

    frame.loc[mask, "during_maintenance"] = True

maintenance_days = int(
    frame["during_maintenance"].sum()
)

frame = frame[
    ~frame["during_maintenance"]
].copy()

print("Excluded maintenance robot-days:", maintenance_days)

# ==========================================
# 3. Create 7-day prediction label
# ==========================================

severe_faults = faults[
    faults["severity"] >= 3
].copy()

print("Severe faults available:", len(severe_faults))

frame["severe_fault_next_7d"] = 0

for fault in severe_faults.itertuples(index=False):

    mask = (
        (frame["unit_id"] == fault.unit_id)
        & (frame["date"] <= fault.occurred_at)
        & (
            frame["date"] + pd.Timedelta(days=7)
            > fault.occurred_at
        )
    )

    frame.loc[mask, "severe_fault_next_7d"] = 1

print(
    "Positive labels before splitting:",
    int(frame["severe_fault_next_7d"].sum())
)

# ==========================================
# 4. Exclude incomplete outcome windows
# ==========================================

# Last complete 7-day window starts June 24.
LAST_VALID_DATE = pd.Timestamp(
    "2026-06-24", tz="UTC"
)

frame = frame[
    frame["date"] <= LAST_VALID_DATE
].copy()

# ==========================================
# 5. Chronological splits with 7-day gaps
# ==========================================

train_end = pd.Timestamp("2025-10-31", tz="UTC")
val_start = pd.Timestamp("2025-11-08", tz="UTC")
val_end = pd.Timestamp("2026-02-28", tz="UTC")
test_start = pd.Timestamp("2026-03-08", tz="UTC")

conditions = [
    frame["date"] <= train_end,
    (frame["date"] >= val_start)
    & (frame["date"] <= val_end),
    frame["date"] >= test_start,
]

frame["split"] = np.select(
    conditions,
    ["train", "validation", "test"],
    default="gap"
)

frame = frame[
    frame["split"] != "gap"
].copy()

# ==========================================
# 6. Validation checks
# ==========================================

assert not frame.duplicated(
    ["unit_id", "date"]
).any(), "Duplicate robot-day rows!"

assert frame["split"].isin(
    ["train", "validation", "test"]
).all()

assert (
    frame["date"] <= LAST_VALID_DATE
).all()

assert not frame["during_maintenance"].any()

# ==========================================
# 7. Save and summarize
# ==========================================

frame = frame.sort_values(
    ["unit_id", "date"]
).reset_index(drop=True)

output_path = (
    OUTPUT_DIR / "W02_robot_day_modeling_frame.csv"
)

frame.to_csv(output_path, index=False)

summary = (
    frame.groupby("split")
    .agg(
        robot_days=("unit_id", "size"),
        positives=("severe_fault_next_7d", "sum")
    )
)

summary["positive_rate_pct"] = (
    summary["positives"]
    / summary["robot_days"] * 100
)

print("\n=== MODELING FRAME SUMMARY ===")
print(summary.to_string())

print("\nTotal rows:", len(frame))
print("Duplicate robot-days:", frame.duplicated(
    ["unit_id", "date"]
).sum())

print("\nSaved to:", output_path)
