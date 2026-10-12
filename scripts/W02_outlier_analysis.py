
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
REPORT_DIR = BASE_DIR / "reports" / "W02"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_DIR / "telemetry.csv")

numeric_cols = [
    "battery_capacity_pct",
    "charge_cycles",
    "motor_current_a",
    "wheel_wear_pct",
    "temperature_c",
]

# ==========================================
# 1. Summary statistics
# ==========================================

print("\n=== TELEMETRY SUMMARY STATISTICS ===")

summary = df[numeric_cols].describe()
print(summary.to_string())

summary.to_csv(
    REPORT_DIR / "telemetry_summary.csv"
)

# ==========================================
# 2. IQR-based outlier detection
# ==========================================

results = []

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (
        (df[col] < lower)
        | (df[col] > upper)
    )

    results.append({
        "column": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(outliers.sum()),
        "outlier_pct": round(
            outliers.mean() * 100, 3
        ),
    })

outlier_report = pd.DataFrame(results)

print("\n=== IQR OUTLIER ANALYSIS ===")
print(outlier_report.to_string(index=False))

outlier_report.to_csv(
    REPORT_DIR / "outlier_summary.csv",
    index=False
)

# ==========================================
# 3. Physical validity checks
# ==========================================

checks = {
    "battery_capacity_out_of_range": (
        (df["battery_capacity_pct"] < 0)
        | (df["battery_capacity_pct"] > 100)
    ),
    "negative_charge_cycles": (
        df["charge_cycles"] < 0
    ),
    "negative_motor_current": (
        df["motor_current_a"] < 0
    ),
    "wheel_wear_out_of_range": (
        (df["wheel_wear_pct"] < 0)
        | (df["wheel_wear_pct"] > 100)
    ),
}

print("\n=== PHYSICAL VALIDITY CHECKS ===")

for name, mask in checks.items():
    print(f"{name}: {int(mask.sum())}")

print("\nReports saved successfully!")
