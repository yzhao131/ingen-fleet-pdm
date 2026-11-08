
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("reports/W06")

# Load PCA coordinates and final K=3 cluster assignments
pca = pd.read_csv(OUT / "W06_pca_coordinates.csv")
clusters = pd.read_csv(OUT / "W06_robot_clusters_k3.csv")
variance = pd.read_csv(OUT / "W06_pca_variance.csv")

print("PCA columns:", pca.columns.tolist())
print("Cluster columns:", clusters.columns.tolist())
print("Variance columns:", variance.columns.tolist())

# Join using robot ID
df = pca.merge(
    clusters[["unit_id", "cluster"]],
    on="unit_id",
    how="inner",
    validate="one_to_one"
)

if len(df) != len(clusters):
    raise ValueError("PCA and cluster robot counts do not match.")

plt.figure(figsize=(9, 6))

for cluster_id in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cluster_id]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=70,
        alpha=0.8,
        label=f"Cluster {cluster_id} (n={len(subset)})"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Week 6 - Final Robot Segmentation (K=3)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUT / "W06_final_pca_clusters_k3.png",
    dpi=200
)
plt.close()

print("\nFinal cluster plot saved.")
print(df["cluster"].value_counts().sort_index())
