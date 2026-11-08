
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT = Path("reports/W06")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "W06_robot_clusters_k3.csv")

required = [
    "unit_id", "site_id", "fault_count",
    "total_operating_hours",
    "avg_mission_hours", "avg_motor_current"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()
df = df[df["total_operating_hours"] > 0].copy()

print("\nRobots:", len(df))
print("Sites:", df["site_id"].nunique())
print("Total faults:", df["fault_count"].sum())

# Check whether terrain is confounded with site
if "terrain_type" in df.columns:
    terrain_check = pd.crosstab(
        df["site_id"], df["terrain_type"]
    )
    print("\nSite x Terrain:")
    print(terrain_check.to_string())

# Standardize continuous predictors
for col in ["avg_mission_hours", "avg_motor_current"]:
    std = df[col].std()
    if std == 0:
        raise ValueError(f"No variation in {col}")
    df[col + "_z"] = (df[col] - df[col].mean()) / std

# SITE_01 is the reference site
df["site_id"] = pd.Categorical(
    df["site_id"],
    categories=sorted(df["site_id"].unique())
)

formula = (
    "fault_count ~ C(site_id) "
    "+ avg_mission_hours_z "
    "+ avg_motor_current_z"
)

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["total_operating_hours"])
).fit(cov_type="HC0")

print("\nPoisson Regression Results:")
print(model.summary())

# Dispersion diagnostic
dispersion = (
    model.pearson_chi2 / model.df_resid
    if model.df_resid > 0 else np.nan
)
print(f"\nPearson dispersion: {dispersion:.3f}")

# Incidence Rate Ratios with robust 95% CIs
ci = model.conf_int()

results = pd.DataFrame({
    "term": model.params.index,
    "coefficient": model.params.values,
    "IRR": np.exp(model.params.values),
    "CI95_low": np.exp(ci.iloc[:, 0].values),
    "CI95_high": np.exp(ci.iloc[:, 1].values),
    "p_value": model.pvalues.values
})

results.to_csv(
    OUT / "W06_site_regression_results.csv",
    index=False
)

print("\nIncidence Rate Ratios:")
print(results.round(4).to_string(index=False))

# Plot site IRRs relative to SITE_01
site_results = results[
    results["term"].str.contains("C(site_id)", regex=False)
].copy()

site_results["site"] = site_results["term"].str.extract(
    r"\[T\.(SITE_\d+)\]"
)

# Add reference site
reference = pd.DataFrame([{
    "site": "SITE_01",
    "IRR": 1.0,
    "CI95_low": 1.0,
    "CI95_high": 1.0
}])

plot_df = pd.concat(
    [
        reference,
        site_results[
            ["site", "IRR", "CI95_low", "CI95_high"]
        ]
    ],
    ignore_index=True
).sort_values("site")

x = np.arange(len(plot_df))

plt.figure(figsize=(10, 5))

plt.errorbar(
    x,
    plot_df["IRR"],
    yerr=[
        plot_df["IRR"] - plot_df["CI95_low"],
        plot_df["CI95_high"] - plot_df["IRR"]
    ],
    fmt="o",
    capsize=4
)

plt.axhline(1.0, color="gray", linestyle="--")
plt.xticks(x, plot_df["site"], rotation=45)
plt.ylabel("Adjusted Incidence Rate Ratio")
plt.xlabel("Site (reference: SITE_01)")
plt.title("Week 6 - Adjusted Site Fault Rate Ratios")
plt.tight_layout()

plt.savefig(
    OUT / "W06_site_regression_irr.png",
    dpi=200
)
plt.close()

with open(
    OUT / "W06_site_regression_diagnostics.txt", "w"
) as f:
    f.write(f"Robots: {len(df)}\n")
    f.write(f"Sites: {df['site_id'].nunique()}\n")
    f.write(f"Pearson dispersion: {dispersion:.4f}\n\n")
    f.write(model.summary().as_text())

print("\nSite regression completed.")
