
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("reports/W06")

df = pd.read_csv(
    OUTPUT_DIR / "W06_robot_clusters_k3.csv"
)

# --------------------------------------------------
# 1. Cluster-level operational risk summary
# --------------------------------------------------
summary = (
    df.groupby("cluster")
    .agg(
        robot_count=("unit_id", "nunique"),
        total_faults=("fault_count", "sum"),
        total_maintenance=("maintenance_count", "sum"),
        total_operating_hours=("total_operating_hours", "sum"),
        mean_robot_fault_rate=(
            "faults_per_1000_hours", "mean"
        ),
        mean_battery_capacity=(
            "latest_battery_capacity", "mean"
        ),
        mean_wheel_wear=("latest_wheel_wear", "mean"),
    )
    .reset_index()
)

# Exposure-weighted historical fault frequency
summary["pooled_faults_per_1000_hours"] = (
    summary["total_faults"]
    / summary["total_operating_hours"]
    * 1000
)

summary = summary.round(3)

print("\nCluster Risk Summary:")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "W06_cluster_risk_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. Compare fault frequencies
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    summary["cluster"].astype(str),
    summary["pooled_faults_per_1000_hours"]
)

plt.xlabel("Cluster")
plt.ylabel("Faults per 1,000 Operating Hours")
plt.title("Week 6 - Historical Fault Frequency by Cluster")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_cluster_fault_frequency.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 3. Robot-level risk distribution
# --------------------------------------------------
groups = [
    df.loc[
        df["cluster"] == cluster,
        "faults_per_1000_hours"
    ].dropna()
    for cluster in sorted(df["cluster"].unique())
]

plt.figure(figsize=(8, 5))

plt.boxplot(groups)

plt.xticks(
    range(1, len(groups) + 1),
    sorted(df["cluster"].unique())
)

plt.xlabel("Cluster")
plt.ylabel("Faults per 1,000 Operating Hours")
plt.title("Week 6 - Robot-Level Fault Rate Distribution")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_cluster_fault_distribution.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 4. Site composition
# --------------------------------------------------
site_composition = pd.crosstab(
    df["site_id"],
    df["cluster"]
)

print("\nSite Composition:")
print(site_composition.to_string())

site_composition.to_csv(
    OUTPUT_DIR / "W06_site_cluster_composition.csv"
)

print("\nCluster risk analysis completed.")
