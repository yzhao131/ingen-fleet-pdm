
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUT = Path("reports/W06")
df = pd.read_csv(OUT / "W06_robot_features.csv")

features = [
    "total_operating_hours",
    "avg_mission_hours",
    "avg_battery_capacity",
    "avg_motor_current",
    "avg_temperature",
    "latest_battery_capacity",
    "latest_charge_cycles",
    "latest_wheel_wear",
]

X = df[features].to_numpy()

# Full-data reference clustering
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=4).fit_transform(X_scaled)

reference = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
).fit_predict(X_pca)

rng = np.random.default_rng(42)

results = []

for iteration in range(100):

    # Randomly select 80% of robots
    idx = np.sort(
        rng.choice(
            len(df),
            size=int(len(df) * 0.8),
            replace=False
        )
    )

    X_sample = X[idx]

    # Refit preprocessing on each sample
    scaled = StandardScaler().fit_transform(X_sample)
    projected = PCA(n_components=4).fit_transform(scaled)

    labels = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=20
    ).fit_predict(projected)

    ari = adjusted_rand_score(
        reference[idx],
        labels
    )

    sizes = np.bincount(labels, minlength=3)

    results.append({
        "iteration": iteration + 1,
        "ari": ari,
        "smallest_cluster": sizes.min()
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "W06_resampling_stability.csv",
    index=False
)

print("\nResampling Stability Results")
print("Iterations:", len(results_df))
print("Sample fraction: 80%")
print("Mean ARI:", round(results_df["ari"].mean(), 4))
print("Median ARI:", round(results_df["ari"].median(), 4))
print("Minimum ARI:", round(results_df["ari"].min(), 4))
print("Std ARI:", round(results_df["ari"].std(), 4))
print(
    "10th percentile ARI:",
    round(results_df["ari"].quantile(0.10), 4)
)
print(
    "Runs with cluster size < 3:",
    int((results_df["smallest_cluster"] < 3).sum())
)

print("\nResampling stability completed.")
