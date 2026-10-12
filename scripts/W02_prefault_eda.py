
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "W02"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Load data
# ==========================================

telemetry = pd.read_csv(
    CLEAN_DIR / "telemetry.csv",
    parse_dates=["timestamp"]
)

frame = pd.read_csv(
    PROCESSED_DIR / "W02_robot_day_modeling_frame.csv",
    parse_dates=["date"]
)

telemetry["timestamp"] = pd.to_datetime(
    telemetry["timestamp"], utc=True
)

frame["date"] = pd.to_datetime(
    frame["date"], utc=True
)

features = [
    "motor_current_a",
    "battery_capacity_pct",
    "wheel_wear_pct",
    "temperature_c",
]

# ==========================================
# 2. Build historical 7-day features
# ==========================================

telemetry = telemetry.sort_values(
    ["unit_id", "timestamp"]
)

frame = frame.sort_values(
    ["unit_id", "date"]
).copy()

results = []

for unit_id, unit_frame in frame.groupby("unit_id"):

    unit_telemetry = telemetry[
        telemetry["unit_id"] == unit_id
    ].set_index("timestamp")

    for row in unit_frame.itertuples(index=False):

        prediction_time = row.date
        history_start = (
            prediction_time - pd.Timedelta(days=7)
        )

        # Only use observations before prediction time
        history = unit_telemetry.loc[
            (unit_telemetry.index >= history_start)
            & (unit_telemetry.index < prediction_time)
        ]

        record = {
            "unit_id": unit_id,
            "date": prediction_time,
            "split": row.split,
            "severe_fault_next_7d":
                row.severe_fault_next_7d,
            "history_readings": len(history),
        }

        for col in features:
            record[f"{col}_mean_7d"] = (
                history[col].mean()
            )

        results.append(record)

eda = pd.DataFrame(results)

# ==========================================
# 3. Exclude incomplete history
# ==========================================

# Expected 4 readings/day x 7 days = 28
# Require at least 21 readings (75% coverage)
eda["sufficient_history"] = (
    eda["history_readings"] >= 21
)

print("\n=== HISTORY COVERAGE ===")
print("Total robot-days:", len(eda))
print(
    "Sufficient history:",
    int(eda["sufficient_history"].sum())
)
print(
    "Insufficient history:",
    int((~eda["sufficient_history"]).sum())
)

eda_valid = eda[
    eda["sufficient_history"]
].copy()

# ==========================================
# 4. Compare positive vs negative samples
# ==========================================

mean_columns = [
    f"{col}_mean_7d" for col in features
]

comparison = (
    eda_valid.groupby("severe_fault_next_7d")[
        mean_columns
    ]
    .mean()
    .T
)

comparison.columns = [
    "negative" if col == 0 else "positive"
    for col in comparison.columns
]

comparison["difference"] = (
    comparison["positive"]
    - comparison["negative"]
)

print("\n=== PRE-FAULT TELEMETRY COMPARISON ===")
print(comparison.to_string())

# ==========================================
# 5. Save CSV reports
# ==========================================

eda.to_csv(
    REPORT_DIR / "prefault_robot_day_features.csv",
    index=False
)

comparison.to_csv(
    REPORT_DIR / "prefault_comparison.csv"
)

# ==========================================
# 6. Generate boxplots
# ==========================================

for col in mean_columns:

    negative = eda_valid.loc[
        eda_valid["severe_fault_next_7d"] == 0,
        col
    ].dropna()

    positive = eda_valid.loc[
        eda_valid["severe_fault_next_7d"] == 1,
        col
    ].dropna()

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [negative, positive],
        tick_labels=[
            "No severe fault",
            "Severe fault within 7 days"
        ]
    )

    plt.title(f"Pre-Fault Comparison: {col}")
    plt.ylabel(col)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"prefault_{col}.png",
        dpi=200
    )

    plt.close()

print("\nPre-fault EDA reports and figures saved!")
