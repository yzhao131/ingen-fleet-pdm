
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUT = Path("reports/W06_corrected")

df = pd.read_csv(
    OUT / "W06_14d_robot_clusters.csv"
)

features = [
    "avg_mission_hours",
    "distance_per_operating_hour",
    "avg_battery_capacity",
    "avg_motor_current",
    "avg_temperature",
    "latest_battery_capacity",
    "charge_cycles_per_100_hours",
    "latest_wheel_wear",
]

X = df[features].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)
Z = pca.fit_transform(X_scaled)

rng = np.random.default_rng(42)
results = []

for k in [2, 3]:
    baseline = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    ).fit_predict(Z)

    # Stability across different random initializations
    seed_scores = []

    for seed in range(50):
        labels = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=20,
        ).fit_predict(Z)

        seed_scores.append(
            adjusted_rand_score(baseline, labels)
        )

    # Stability under subsampling
    subsample_scores = []

    for repeat in range(100):
        idx = np.sort(
            rng.choice(
                len(X),
                size=int(len(X) * 0.8),
                replace=False,
            )
        )

        X_sub = X[idx]

        X_sub_scaled = (
            StandardScaler().fit_transform(X_sub)
        )

        Z_sub = PCA(
            n_components=4
        ).fit_transform(X_sub_scaled)

        sub_labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        ).fit_predict(Z_sub)

        subsample_scores.append(
            adjusted_rand_score(
                baseline[idx],
                sub_labels,
            )
        )

    results.append({
        "k": k,
        "seed_ARI_mean": np.mean(seed_scores),
        "seed_ARI_min": np.min(seed_scores),
        "subsample_ARI_mean":
            np.mean(subsample_scores),
        "subsample_ARI_std":
            np.std(subsample_scores),
        "subsample_ARI_min":
            np.min(subsample_scores),
    })

results = pd.DataFrame(results)

print("\n=== CLUSTER STABILITY ===")
print(results.round(4).to_string(index=False))

results.to_csv(
    OUT / "W06_14d_cluster_stability.csv",
    index=False,
)
