
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

frame = pd.read_csv(
    BASE_DIR / "data/processed/W02_robot_day_modeling_frame.csv"
)

faults = pd.read_csv(
    BASE_DIR / "data/clean/faults.csv"
)

frame["date"] = pd.to_datetime(frame["date"], utc=True)
faults["occurred_at"] = pd.to_datetime(
    faults["occurred_at"], utc=True
)

severe = faults[faults["severity"] >= 3]

# ==========================================
# 1. Check duplicate robot-days
# ==========================================

assert not frame.duplicated(
    ["unit_id", "date"]
).any()

print("Duplicate robot-days: 0")

# ==========================================
# 2. Verify all labels independently
# ==========================================

errors = 0

fault_times = {
    unit: group["occurred_at"].tolist()
    for unit, group in severe.groupby("unit_id")
}

for row in frame.itertuples(index=False):

    start = row.date
    end = start + pd.Timedelta(days=7)

    expected_label = int(any(
        start <= fault_time < end
        for fault_time in fault_times.get(row.unit_id, [])
    ))

    if expected_label != row.severe_fault_next_7d:
        errors += 1

print("Label mismatches:", errors)

assert errors == 0, "Incorrect labels detected!"

# ==========================================
# 3. Verify chronological split boundaries
# ==========================================

expected_ranges = {
    "train": ("2025-01-01", "2025-10-31"),
    "validation": ("2025-11-08", "2026-02-28"),
    "test": ("2026-03-08", "2026-06-24"),
}

for split, (start, end) in expected_ranges.items():

    subset = frame[frame["split"] == split]

    assert not subset.empty

    assert subset["date"].min() >= pd.Timestamp(
        start, tz="UTC"
    )

    assert subset["date"].max() <= pd.Timestamp(
        end, tz="UTC"
    )

    print(
        f"{split}:",
        subset["date"].min().date(),
        "to",
        subset["date"].max().date()
    )

# ==========================================
# 4. Check class balance
# ==========================================

print("\n=== CLASS BALANCE ===")

summary = (
    frame.groupby("split")["severe_fault_next_7d"]
    .agg(["count", "sum", "mean"])
)

print(summary.to_string())

print("\nAll modeling frame checks passed!")
