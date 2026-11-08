
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load robot features
# --------------------------------------------------
OUTPUT_DIR = Path("reports/W06")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

X = df[feature_cols]

# --------------------------------------------------
# 2. StandardScaler + PCA
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

print(
    "PCA cumulative explained variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

# --------------------------------------------------
# 3. Evaluate different K values
# --------------------------------------------------
results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_pca)

    silhouette = silhouette_score(X_pca, labels)
    db_index = davies_bouldin_score(X_pca, labels)

    results.append({
        "k": k,
        "silhouette_score": silhouette,
        "davies_bouldin_index": db_index,
        "inertia": model.inertia_,
    })

evaluation = pd.DataFrame(results)

print("\nClustering Evaluation:")
print(evaluation.to_string(index=False))

evaluation.to_csv(
    OUTPUT_DIR / "W06_cluster_evaluation.csv",
    index=False
)

# --------------------------------------------------
# 4. Select K using Silhouette Score
# --------------------------------------------------
best_k = int(
    evaluation.loc[
        evaluation["silhouette_score"].idxmax(),
        "k"
    ]
)

print(f"\nBest K by Silhouette Score: {best_k}")

# --------------------------------------------------
# 5. Final clustering
# --------------------------------------------------
final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df["cluster"] = final_model.fit_predict(X_pca)

# Export robot cluster assignments
df.to_csv(
    OUTPUT_DIR / "W06_robot_clusters.csv",
    index=False
)

# --------------------------------------------------
# 6. Cluster profiles
# --------------------------------------------------
profile_cols = feature_cols + [
    "fault_count",
    "maintenance_count",
    "faults_per_1000_hours",
]

cluster_profile = (
    df.groupby("cluster")[profile_cols]
    .mean()
    .round(3)
)

cluster_profile.insert(
    0,
    "robot_count",
    df.groupby("cluster").size()
)

print("\nCluster Profiles:")
print(cluster_profile.to_string())

cluster_profile.to_csv(
    OUTPUT_DIR / "W06_cluster_profiles.csv"
)

# --------------------------------------------------
# 7. Plot clustering evaluation
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    evaluation["k"],
    evaluation["silhouette_score"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Week 6 - K-Means Cluster Evaluation")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_silhouette_scores.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 8. PCA cluster visualization
# --------------------------------------------------
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["cluster"],
    cmap="tab10",
    s=65,
    alpha=0.85
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Week 6 - Robot Clusters (PCA Projection)")

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_robot_clusters_pca.png",
    dpi=200
)
plt.close()

print("\nClustering completed.")
print("Results saved to:", OUTPUT_DIR)
