
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("reports/W06")

df = pd.read_csv(
    OUTPUT_DIR / "W06_robot_clusters_k3.csv"
)

# --------------------------------------------------
# 1. Site-level summary
# --------------------------------------------------
site_summary = (
    df.groupby("site_id")
    .agg(
        robot_count=("unit_id", "nunique"),
        total_faults=("fault_count", "sum"),
        total_maintenance=("maintenance_count", "sum"),
        total_operating_hours=("total_operating_hours", "sum"),
        mean_battery_capacity=(
            "latest_battery_capacity", "mean"
        ),
        mean_wheel_wear=("latest_wheel_wear", "mean"),
    )
    .reset_index()
)

site_summary["faults_per_1000_hours"] = (
    site_summary["total_faults"]
    / site_summary["total_operating_hours"]
    * 1000
)

site_summary = site_summary.round(3)

print("\nSite-Level Summary:")
print(site_summary.to_string(index=False))

site_summary.to_csv(
    OUTPUT_DIR / "W06_site_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. Cluster proportions by site
# --------------------------------------------------
site_cluster = pd.crosstab(
    df["site_id"],
    df["cluster"],
    normalize="index"
) * 100

site_cluster = site_cluster.round(2)

print("\nCluster Proportions by Site (%):")
print(site_cluster.to_string())

site_cluster.to_csv(
    OUTPUT_DIR / "W06_site_cluster_proportions.csv"
)

# --------------------------------------------------
# 3. Plot site cluster composition
# --------------------------------------------------
ax = site_cluster.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    colormap="tab10"
)

ax.set_xlabel("Site")
ax.set_ylabel("Percentage of Robots")
ax.set_title("Week 6 - Cluster Composition by Site")
ax.legend(title="Cluster")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_site_cluster_proportions.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 4. Plot historical fault frequency
# --------------------------------------------------
plt.figure(figsize=(9, 5))

plt.bar(
    site_summary["site_id"],
    site_summary["faults_per_1000_hours"]
)

plt.xlabel("Site")
plt.ylabel("Faults per 1,000 Operating Hours")
plt.title("Week 6 - Historical Fault Frequency by Site")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_site_fault_frequency.png",
    dpi=200
)
plt.close()

print("\nSite-level analysis completed.")
