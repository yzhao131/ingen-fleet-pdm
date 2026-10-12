
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

raw = pd.read_csv(
    BASE_DIR / "data/raw/telemetry.csv"
)

clean = pd.read_csv(
    BASE_DIR / "data/clean/telemetry.csv"
)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"], utc=True
)

clean["timestamp"] = pd.to_datetime(
    clean["timestamp"], utc=True
)

# Telemetry is expected every 6 hours
INTERVAL = pd.Timedelta(hours=6)

def calculate_drift(df):
    results = []

    for unit_id, group in df.groupby("unit_id"):

        timestamps = group["timestamp"]

        # Nearest expected 6-hour timestamp
        nearest_grid = timestamps.dt.round("6h")

        offsets = (
            timestamps - nearest_grid
        ).dt.total_seconds() / 60

        median_offset = offsets.median()

        results.append({
            "unit_id": unit_id,
            "median_offset_minutes": median_offset,
            "drift_detected": abs(median_offset) > 1
        })

    return pd.DataFrame(results)


raw_drift = calculate_drift(raw)
clean_drift = calculate_drift(clean)

audit = raw_drift.merge(
    clean_drift,
    on="unit_id",
    suffixes=("_raw", "_clean")
)

print("\n=== CLOCK DRIFT AUDIT ===")

print(
    "Raw robots with drift:",
    int(audit["drift_detected_raw"].sum())
)

print(
    "Clean robots with drift:",
    int(audit["drift_detected_clean"].sum())
)

print("\nDrift correction details:")

print(
    audit.loc[
        audit["drift_detected_raw"],
        [
            "unit_id",
            "median_offset_minutes_raw",
            "median_offset_minutes_clean"
        ]
    ].to_string(index=False)
)

assert audit["drift_detected_clean"].sum() == 0, (
    "Clock drift remains after cleaning!"
)

output_dir = BASE_DIR / "reports/W02"
output_dir.mkdir(parents=True, exist_ok=True)

audit.to_csv(
    output_dir / "clock_drift_audit.csv",
    index=False
)

print("\nClock drift audit passed!")
print("Saved to:", output_dir / "clock_drift_audit.csv")
