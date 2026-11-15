
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = Path("reports/W06_corrected")

df = pd.read_csv(
    OUT / "W06_followup_robot_outcomes.csv"
)

features = [
    "avg_motor_current",
    "latest_wheel_wear",
    "avg_temperature",
    "latest_battery_capacity",
    "avg_mission_hours",
    "charge_cycles_per_100_hours",
    "avg_battery_capacity",
    "distance_per_operating_hour",
]

site_dummies = pd.get_dummies(
    df["site_id"].astype("category"),
    prefix="site",
    drop_first=True,
    dtype=float,
)

results = []

for feature in features:
    x = df[feature].astype(float)

    sd = x.std(ddof=0)
    if sd == 0:
        continue

    standardized = (x - x.mean()) / sd

    X = pd.concat(
        [
            pd.DataFrame({
                feature: standardized
            }, index=df.index),
            site_dummies,
        ],
        axis=1,
    )

    X = sm.add_constant(X)

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
        "adjusted_IRR": np.exp(beta),
        "CI_lower": np.exp(beta - 1.96 * se),
        "CI_upper": np.exp(beta + 1.96 * se),
        "p_value": fit.pvalues[feature],
    })

results = (
    pd.DataFrame(results)
    .sort_values("p_value")
)

results.to_csv(
    OUT / "W06_site_adjusted_risk_drivers.csv",
    index=False,
)

print("\n=== SITE-ADJUSTED RISK DRIVERS ===")
print(results.round(4).to_string(index=False))
