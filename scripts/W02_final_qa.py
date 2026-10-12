
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

frame = pd.read_csv(
    BASE_DIR / "data/processed/W02_robot_day_modeling_frame.csv"
)

maintenance = pd.read_csv(
    BASE_DIR / "data/clean/maintenance.csv"
)

robots = pd.read_csv(
    BASE_DIR / "data/clean/robots.csv"
)

frame["date"] = pd.to_datetime(frame["date"], utc=True)

for col in ["start_time", "end_time"]:
    maintenance[col] = pd.to_datetime(
        maintenance[col], utc=True
    )

print("\n=== WEEK 2 FINAL QA ===")

# ==========================================
# 1. Check unique robot-day keys
# ==========================================

duplicate_count = frame.duplicated(
    ["unit_id", "date"]
).sum()

print("Duplicate robot-days:", duplicate_count)

assert duplicate_count == 0

# ==========================================
# 2. Check maintenance overlap
# ==========================================

overlap_count = 0

for row in maintenance.itertuples(index=False):

    overlap = (
    (frame["unit_id"] == row.unit_id)
    & (frame["date"] >= row.start_time)
    & (frame["date"] < row.end_time)
    )

    overlap_count += int(overlap.sum())

print("Maintenance overlaps:", overlap_count)

assert overlap_count == 0

# ==========================================
# 3. Check valid labels
# ==========================================

invalid_labels = (
    ~frame["severe_fault_next_7d"].isin([0, 1])
).sum()

print("Invalid labels:", invalid_labels)

assert invalid_labels == 0

# ==========================================
# 4. Check robot IDs
# ==========================================

unknown_robots = (
    ~frame["unit_id"].isin(robots["unit_id"])
).sum()

print("Unknown robot IDs:", unknown_robots)

assert unknown_robots == 0

# ==========================================
# 5. Check split boundaries
# ==========================================

ranges = {
    "train": ("2025-01-01", "2025-10-31"),
    "validation": ("2025-11-08", "2026-02-28"),
    "test": ("2026-03-08", "2026-06-24"),
}

invalid_split_rows = 0

for split, (start, end) in ranges.items():

    subset = frame[frame["split"] == split]

    invalid_split_rows += int((
        (subset["date"] < pd.Timestamp(start, tz="UTC"))
        | (subset["date"] > pd.Timestamp(end, tz="UTC"))
    ).sum())

invalid_split_rows += int(
    (~frame["split"].isin(ranges)).sum()
)

print("Invalid split rows:", invalid_split_rows)

assert invalid_split_rows == 0

# ==========================================
# 6. Check missing values
# ==========================================

missing_keys = frame[
    ["unit_id", "date", "split", "severe_fault_next_7d"]
].isna().sum().sum()

print("Missing critical values:", missing_keys)

assert missing_keys == 0

print("\nAll final QA checks passed!")
