
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = Path("reports/W06_corrected")
DATA = Path("data/causal_fixed/raw")

clusters = pd.read_csv(
    OUT / "W06_14d_robot_clusters.csv"
)
faults = pd.read_csv(DATA / "faults.csv")
missions = pd.read_csv(DATA / "missions.csv")

for frame, column in [
    (clusters, "window_end"),
    (faults, "occurred_at"),
    (missions, "start_time"),
    (missions, "end_time"),
]:
    frame[column] = pd.to_datetime(
        frame[column],
        format="mixed",
        utc=True,
    )

# Future faults
f = faults.merge(
    clusters[["unit_id", "window_end"]],
    on="unit_id",
    how="inner",
)

f = f[f["occurred_at"] >= f["window_end"]]

fault_counts = (
    f.groupby("unit_id")
    .size()
    .rename("future_fault_count")
)

# Future operating hours
m = missions.merge(
    clusters[["unit_id", "window_end"]],
    on="unit_id",
    how="inner",
)

m = m[
    (m["mission_status"] == "completed")
    & (m["start_time"] >= m["window_end"])
    & (m["end_time"] > m["start_time"])
]

hours = (
    m.groupby("unit_id")["operating_hours"]
    .sum()
    .rename("future_operating_hours")
)

df = (
    clusters
    .merge(fault_counts, on="unit_id", how="left")
    .merge(hours, on="unit_id", how="left")
)

df["future_fault_count"] = (
    df["future_fault_count"]
    .fillna(0)
    .astype(int)
)

df["future_operating_hours"] = (
    df["future_operating_hours"]
    .fillna(0)
)

df = df[df["future_operating_hours"] > 0].copy()

df.to_csv(
    OUT / "W06_followup_robot_outcomes.csv",
    index=False,
)

results = []

for k in [2, 3]:
    cluster_col = f"cluster_k{k}"

    X = pd.get_dummies(
        df[cluster_col].astype("category"),
        prefix="cluster",
        drop_first=True,
        dtype=float,
    )

    X = sm.add_constant(X)

    model = sm.GLM(
        df["future_fault_count"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(df["future_operating_hours"]),
    )

    fit = model.fit(cov_type="HC0")

    print(f"\n=== K={k} POISSON GLM ===")
    print(fit.summary())

    for term in fit.params.index:
        if term == "const":
            continue

        beta = fit.params[term]
        se = fit.bse[term]

        results.append({
            "k": k,
            "term": term,
            "IRR": np.exp(beta),
            "CI_lower": np.exp(beta - 1.96 * se),
            "CI_upper": np.exp(beta + 1.96 * se),
            "p_value": fit.pvalues[term],
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "W06_followup_risk_glm.csv",
    index=False,
)

print("\n=== RATE RATIO RESULTS ===")
print(results_df.round(4).to_string(index=False))
