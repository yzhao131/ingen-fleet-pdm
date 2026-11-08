
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

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

# --------------------------------------------------
# 1. Standardization and PCA
# --------------------------------------------------
X_scaled = StandardScaler().fit_transform(
    df[feature_cols]
)

X_pca = PCA(n_components=4).fit_transform(X_scaled)

# --------------------------------------------------
# 2. Compare random initializations
# --------------------------------------------------
seeds = range(30)

results = []

for k in [3, 4, 5]:

    reference = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    ).fit_predict(X_pca)

    ari_scores = []

    for seed in seeds:

        labels = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=20
        ).fit_predict(X_pca)

        ari = adjusted_rand_score(
            reference, labels
        )

        ari_scores.append(ari)

    results.append({
        "k": k,
        "mean_ari": np.mean(ari_scores),
        "min_ari": np.min(ari_scores),
        "std_ari": np.std(ari_scores),
    })

summary = pd.DataFrame(results)

print("\nClustering Stability Results:")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "W06_clustering_stability.csv",
    index=False
)

# --------------------------------------------------
# 3. Visualization
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    summary["k"].astype(str),
    summary["mean_ari"]
)

plt.ylim(0, 1)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Mean Adjusted Rand Index")
plt.title("Week 6 - K-Means Initialization Stability")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_clustering_stability.png",
    dpi=200
)

plt.close()

print("\nStability analysis completed.")
