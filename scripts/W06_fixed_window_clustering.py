
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

OUT = Path("reports/W06_corrected")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "W06_robot_features_14d.csv")

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

# 1. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variance = pd.DataFrame({
    "component": [
        f"PC{i+1}" for i in range(X_pca.shape[1])
    ],
    "explained_variance_ratio":
        pca.explained_variance_ratio_,
    "cumulative_variance":
        np.cumsum(pca.explained_variance_ratio_),
})

variance.to_csv(
    OUT / "W06_14d_pca_variance.csv",
    index=False,
)

# Match the original four-component analysis.
Z = X_pca[:, :4]

print("\n=== PCA VARIANCE ===")
print(variance.to_string(index=False))

# 3. Evaluate K=2..8
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
        "davies_bouldin":
            davies_bouldin_score(Z, labels),
        "min_cluster_size": counts.min(),
        "max_cluster_size": counts.max(),
        "inertia": model.inertia_,
    })

evaluation = pd.DataFrame(results)
evaluation.to_csv(
    OUT / "W06_14d_cluster_evaluation.csv",
    index=False,
)

print("\n=== CLUSTER EVALUATION ===")
print(evaluation.to_string(index=False))

# 4. Compare K=2 and K=3
for k in [2, 3]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(Z)
    df[f"cluster_k{k}"] = labels

    print(f"\n=== K={k} CLUSTER SIZES ===")
    print(
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_string()
    )

    profile = (
        df.groupby(f"cluster_k{k}")[feature_cols]
        .mean()
        .round(3)
    )

    print(f"\n=== K={k} PROFILES ===")
    print(profile.to_string())

    profile.to_csv(
        OUT / f"W06_14d_cluster_profiles_k{k}.csv"
    )

# 5. Compare original K=3 with new K=3
old_path = Path(
    "reports/W06/W06_robot_clusters_k3.csv"
)

if old_path.exists():
    old = pd.read_csv(old_path)

    comparison = old[
        ["unit_id", "cluster"]
    ].merge(
        df[["unit_id", "cluster_k3"]],
        on="unit_id",
        how="inner",
        validate="one_to_one",
    )

    ari = adjusted_rand_score(
        comparison["cluster"],
        comparison["cluster_k3"],
    )

    print("\n=== ORIGINAL VS 14D K=3 ===")
    print("Matched robots:", len(comparison))
    print("ARI:", round(ari, 4))

# 6. Save assignments
df.to_csv(
    OUT / "W06_14d_robot_clusters.csv",
    index=False,
)

# 7. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, k in zip(axes, [2, 3]):
    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=df[f"cluster_k{k}"],
        cmap="tab10",
        s=65,
        alpha=0.85,
    )

    ax.set_title(f"14-Day Clustering (K={k})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.savefig(
    OUT / "W06_14d_clusters_k2_k3.png",
    dpi=200,
)
plt.close()

print("\nSaved results to:", OUT)
