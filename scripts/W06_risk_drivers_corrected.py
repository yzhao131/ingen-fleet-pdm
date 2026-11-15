
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = Path("reports/W06_corrected")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    OUT / "W06_followup_robot_outcomes.csv"
)

features = [
    "avg_battery_capacity",
    "latest_battery_capacity",
    "avg_motor_current",
    "avg_temperature",
    "charge_cycles_per_100_hours",
    "latest_wheel_wear",
    "avg_mission_hours",
    "distance_per_operating_hour",
]

required = features + [
    "future_fault_count",
    "future_operating_hours",
]

assert df[required].notna().all().all()
assert (df["future_operating_hours"] > 0).all()

# 1. Feature correlations
corr = df[features].corr()

corr.to_csv(
    OUT / "W06_corrected_feature_correlations.csv"
)

print("\n=== FEATURE CORRELATIONS ===")
print(corr.round(3).to_string())

# 2. Univariate Poisson regression
results = []

for feature in features:
    x = df[feature].astype(float)

    mean = x.mean()
    std = x.std(ddof=0)

    if std == 0:
        continue

    x_standardized = (x - mean) / std

    X = sm.add_constant(
        pd.DataFrame({
            feature: x_standardized
        })
    )

    model = sm.GLM(
        df["future_fault_count"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(
            df["future_operating_hours"]
        ),
    )

    fit = model.fit(cov_type="HC0")

    beta = fit.params[feature]
    se = fit.bse[feature]

    results.append({
        "feature": feature,
        "IRR_per_1SD": np.exp(beta),
        "CI_lower": np.exp(beta - 1.96 * se),
        "CI_upper": np.exp(beta + 1.96 * se),
        "p_value": fit.pvalues[feature],
        "coefficient": beta,
        "feature_mean": mean,
        "feature_std": std,
    })

results_df = (
    pd.DataFrame(results)
    .sort_values("p_value")
)

results_df.to_csv(
    OUT / "W06_corrected_risk_driver_glm.csv",
    index=False,
)

print("\n=== CORRECTED RISK DRIVERS ===")
print(
    results_df[
        [
            "feature",
            "IRR_per_1SD",
            "CI_lower",
            "CI_upper",
            "p_value",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved results to:", OUT)
