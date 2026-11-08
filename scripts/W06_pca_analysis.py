
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Paths and data
# --------------------------------------------------
OUTPUT_DIR = Path("reports/W06")
df = pd.read_csv(OUTPUT_DIR / "W06_robot_features.csv")

# Select features for unsupervised clustering.
# Keep fault/maintenance outcomes for later interpretation,
# rather than using them to define the clusters.
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

X = df[feature_cols].copy()

assert X.notna().all().all(), "Missing feature values found."
assert np.isfinite(X.to_numpy()).all(), "Non-finite values found."

# --------------------------------------------------
# 2. Standardization
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. PCA
# --------------------------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

summary = pd.DataFrame({
    "component": [
        f"PC{i+1}" for i in range(len(explained_variance))
    ],
    "explained_variance_ratio": explained_variance,
    "cumulative_variance": cumulative_variance,
})

print("\nPCA Explained Variance:")
print(summary.to_string(index=False))

# Number of PCs needed to explain at least 80% variance
n_components_80 = np.searchsorted(
    cumulative_variance, 0.80
) + 1

print(f"\nComponents for >=80% variance: {n_components_80}")

# --------------------------------------------------
# 4. PCA loadings
# --------------------------------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[
        f"PC{i+1}" for i in range(len(feature_cols))
    ],
)

print("\nPCA Loadings (first two components):")
print(loadings[["PC1", "PC2"]].round(3))

# --------------------------------------------------
# 5. Export results
# --------------------------------------------------
summary.to_csv(
    OUTPUT_DIR / "W06_pca_variance.csv",
    index=False
)

loadings.to_csv(
    OUTPUT_DIR / "W06_pca_loadings.csv"
)

coordinates = df[["unit_id", "site_id"]].copy()
coordinates["PC1"] = X_pca[:, 0]
coordinates["PC2"] = X_pca[:, 1]

coordinates.to_csv(
    OUTPUT_DIR / "W06_pca_coordinates.csv",
    index=False
)

# --------------------------------------------------
# 6. Scree plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance,
    label="Individual variance"
)

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
    label="Cumulative variance"
)

plt.axhline(
    0.80,
    color="red",
    linestyle="--",
    label="80% threshold"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Week 6 - PCA Explained Variance")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_pca_scree_plot.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 7. PCA scatter plot
# --------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.75
)

plt.xlabel(
    f"PC1 ({explained_variance[0]:.1%} variance)"
)
plt.ylabel(
    f"PC2 ({explained_variance[1]:.1%} variance)"
)

plt.title("Week 6 - Robot PCA Projection")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "W06_pca_scatter.png",
    dpi=200
)
plt.close()

print("\nPCA analysis completed.")
print("Outputs saved to:", OUTPUT_DIR)
