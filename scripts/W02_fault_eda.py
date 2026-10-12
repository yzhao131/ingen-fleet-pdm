
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
REPORT_DIR = BASE_DIR / "reports" / "W02"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

robots = pd.read_csv(CLEAN_DIR / "robots.csv")
faults = pd.read_csv(CLEAN_DIR / "faults.csv")

faults["occurred_at"] = pd.to_datetime(
    faults["occurred_at"], utc=True
)

severe = faults[
    faults["severity"] >= 3
].copy()

# ==========================================
# 1. Site-level severe fault rates
# ==========================================

site_robot_counts = (
    robots.groupby("site_id")["unit_id"]
    .nunique()
    .rename("robot_count")
)

site_fault_counts = (
    severe.merge(
        robots[["unit_id", "site_id"]],
        on="unit_id",
        how="left",
        validate="many_to_one"
    )
    .groupby("site_id")
    .size()
    .rename("severe_faults")
)

site_summary = pd.concat(
    [site_robot_counts, site_fault_counts],
    axis=1
).fillna(0)

# Observation period: 546 days
site_summary["robot_days"] = (
    site_summary["robot_count"] * 546
)

site_summary["faults_per_1000_robot_days"] = (
    site_summary["severe_faults"]
    / site_summary["robot_days"]
    * 1000
)

site_summary = site_summary.sort_values(
    "faults_per_1000_robot_days",
    ascending=False
)

print("\n=== SITE-LEVEL FAULT RATES ===")
print(site_summary.to_string())

site_summary.to_csv(
    REPORT_DIR / "site_fault_rates.csv"
)

# ==========================================
# 2. Monthly severe fault trends
# ==========================================

severe["month"] = (
    severe["occurred_at"]
    .dt.strftime("%Y-%m")
)

monthly = (
    severe.groupby("month")
    .size()
    .rename("severe_faults")
)

# Include months with zero faults
months = pd.period_range(
    "2025-01", "2026-06", freq="M"
).astype(str)

monthly = monthly.reindex(
    months, fill_value=0
)

print("\n=== MONTHLY SEVERE FAULTS ===")
print(monthly.to_string())

monthly.to_csv(
    REPORT_DIR / "monthly_severe_faults.csv"
)

# ==========================================
# 3. Save figures
# ==========================================

plt.figure(figsize=(10, 5))

site_summary["faults_per_1000_robot_days"].plot(
    kind="bar"
)

plt.title("Severe Fault Rate by Site")
plt.ylabel("Severe faults per 1,000 robot-days")
plt.xlabel("Site")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "site_fault_rates.png",
    dpi=200
)

plt.close()

plt.figure(figsize=(11, 5))

monthly.plot(
    kind="line",
    marker="o"
)

plt.title("Monthly Severe Fault Events")
plt.ylabel("Number of severe faults")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "monthly_severe_faults.png",
    dpi=200
)

plt.close()

print("\nEDA reports and figures saved!")
