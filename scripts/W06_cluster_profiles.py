
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUT_DIR = Path("reports/W06")

df = pd.read_csv(
    OUTPUT_DIR / "W06_robot_features.csv"
)

feature_cols = [
    "total_operating_hours",
    "avg_mission_hours",
    "avg_battery_capacity",
    "avg_motor_current",
    "avg_temperature",
    "latest_battery_capacity",
    "latest_charge_cycles",
    "latest_wheel_wear",
]

# Standardization
X_scaled = StandardScaler().fit_transform(
    df[feature_cols]
)

# PCA
X_pca = PCA(n_components=4).fit_transform(
    X_scaled
)

# K-Means with K=3
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

df["cluster"] = model.fit_predict(X_pca)

# Save robot-level cluster assignments
df.to_csv(
    OUTPUT_DIR / "W06_robot_clusters_k3.csv",
    index=False
)

# Cluster profiles
profile_cols = feature_cols + [
    "fault_count",
    "maintenance_count",
    "faults_per_1000_hours",
]

profiles = (
    df.groupby("cluster")[profile_cols]
    .mean()
    .round(3)
)

profiles.insert(
    0,
    "robot_count",
    df.groupby("cluster").size()
)

print("\nK=3 Cluster Profiles:")
print(profiles.to_string())

print(
    "\nSilhouette Score:",
    round(silhouette_score(X_pca, df["cluster"]), 4)
)

profiles.to_csv(
    OUTPUT_DIR / "W06_cluster_profiles_k3.csv"
)

print("\nSaved K=3 clustering results.")
