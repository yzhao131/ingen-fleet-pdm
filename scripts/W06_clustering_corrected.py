
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
OUT = Path("reports/W06_corrected")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    OUT / "W06_robot_features_pre_repair.csv"
)

# --------------------------------------------------
# 2. Exposure-normalized features
# --------------------------------------------------
df["charge_cycles_per_100_hours"] = (
    df["latest_charge_cycles"]
    / df["total_operating_hours"] * 100
)

df["distance_per_operating_hour"] = (
    df["total_distance_km"]
    / df["total_operating_hours"]
)

feature_cols = [
    "avg_mission_hours",
    "distance_per_operating_hour",
    "avg_battery_capacity",
    "avg_motor_current",
    "avg_temperature",
    "latest_battery_capacity",
    "charge_cycles_per_100_hours",
    "latest_wheel_wear",
]

X = df[feature_cols].copy()

assert X.notna().all().all()
assert np.isfinite(X.to_numpy()).all()

# --------------------------------------------------
# 3. Standardization and PCA
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variance = pd.DataFrame({
    "component": [
        f"PC{i + 1}" for i in range(X_pca.shape[1])
    ],
    "explained_variance_ratio":
        pca.explained_variance_ratio_,
    "cumulative_variance":
        np.cumsum(pca.explained_variance_ratio_),
})

variance.to_csv(
    OUT / "W06_corrected_pca_variance.csv",
    index=False,
)

n_components = min(
    4, X_pca.shape[1]
)
Z = X_pca[:, :n_components]

print("\nPCA variance:")
print(variance.to_string(index=False))

print(
    "\nVariance explained by first 4 PCs:",
    round(
        pca.explained_variance_ratio_[:4].sum(),
        4
    )
)

# --------------------------------------------------
# 4. Evaluate K=2 to K=8
# --------------------------------------------------
results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(Z)
    counts = pd.Series(labels).value_counts()

    results.append({
        "k": k,
        "silhouette": silhouette_score(Z, labels),
        "davies_bouldin": davies_bouldin_score(
            Z, labels
        ),
        "inertia": model.inertia_,
        "min_cluster_size": counts.min(),
        "max_cluster_size": counts.max(),
    })

evaluation = pd.DataFrame(results)

evaluation.to_csv(
    OUT / "W06_corrected_cluster_evaluation.csv",
    index=False,
)

print("\nCluster evaluation:")
print(evaluation.to_string(index=False))

# --------------------------------------------------
# 5. K=3 candidate clustering
# --------------------------------------------------
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20,
)

df["cluster"] = model.fit_predict(Z)

print("\nK=3 cluster sizes:")
print(
    df["cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)

df.to_csv(
    OUT / "W06_robot_clusters_k3_corrected.csv",
    index=False,
)

# --------------------------------------------------
# 6. Cluster profiles
# --------------------------------------------------
profile_cols = feature_cols + [
    "total_operating_hours",
    "fault_count",
    "faults_per_1000_hours",
]

profile = (
    df.groupby("cluster")[profile_cols]
    .mean()
    .round(3)
)

profile.insert(
    0,
    "robot_count",
    df.groupby("cluster").size(),
)

profile.to_csv(
    OUT / "W06_cluster_profiles_k3_corrected.csv"
)

print("\nCluster profiles:")
print(profile.to_string())

# --------------------------------------------------
# 7. Kruskal-Wallis tests
# --------------------------------------------------
tests = []

for col in profile_cols:
    groups = [
        group[col].dropna().to_numpy()
        for _, group in df.groupby("cluster")
    ]

    stat, p_value = kruskal(*groups)

    tests.append({
        "feature": col,
        "H_statistic": stat,
        "p_value": p_value,
    })

tests_df = pd.DataFrame(tests)
tests_df.to_csv(
    OUT / "W06_cluster_difference_tests.csv",
    index=False,
)

print("\nCluster difference tests:")
print(tests_df.to_string(index=False))

# --------------------------------------------------
# 8. Compare original vs corrected clusters
# --------------------------------------------------
old_path = Path(
    "reports/W06/W06_robot_clusters_k3.csv"
)

if old_path.exists():
    old = pd.read_csv(old_path)

    comparison = old[
        ["unit_id", "cluster"]
    ].merge(
        df[["unit_id", "cluster"]],
        on="unit_id",
        suffixes=("_original", "_corrected"),
        validate="one_to_one",
    )

    ari = adjusted_rand_score(
        comparison["cluster_original"],
        comparison["cluster_corrected"],
    )

    print(
        "\nOriginal vs corrected cluster ARI:",
        round(ari, 4)
    )

    comparison.to_csv(
        OUT / "W06_cluster_assignment_comparison.csv",
        index=False,
    )

# --------------------------------------------------
# 9. PCA cluster visualization
# --------------------------------------------------
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    Z[:, 0],
    Z[:, 1],
    c=df["cluster"],
    cmap="tab10",
    s=65,
    alpha=0.85,
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Corrected Pre-Repair Robot Clusters (K=3)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig(
    OUT / "W06_corrected_clusters_k3.png",
    dpi=200,
)
plt.close()

print("\nCorrected clustering analysis completed.")
print("Outputs saved to:", OUT)
