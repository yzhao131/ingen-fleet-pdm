
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

df = pd.read_csv(
    REPORT_DIR / "W05_battery_lifecycle.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values(
    ["unit_id", "battery_segment", "timestamp"]
).reset_index(drop=True)

# --------------------------------------------------
# Chronological split within each battery segment
# --------------------------------------------------

train_parts = []
test_parts = []

for (_, _), group in df.groupby(
    ["unit_id", "battery_segment"]
):
    group = group.sort_values("timestamp")

    if len(group) < 20:
        continue

    split = int(len(group) * 0.8)

    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train = pd.concat(train_parts).copy()
test = pd.concat(test_parts).copy()

print("=== Dataset Split ===")
print("Train records:", len(train))
print("Test records:", len(test))

results = []

# --------------------------------------------------
# Model 1: Per-unit / per-segment Linear Regression
# --------------------------------------------------

for (unit, segment), train_group in train.groupby(
    ["unit_id", "battery_segment"]
):
    test_group = test[
        (test["unit_id"] == unit)
        & (test["battery_segment"] == segment)
    ]

    if test_group.empty:
        continue

    model = LinearRegression()

    model.fit(
        train_group[["cycles_since_repair"]],
        train_group["battery_capacity_pct"]
    )

    predictions = np.clip(
        model.predict(
            test_group[["cycles_since_repair"]]
        ),
        50,
        100
    )

    results.extend([
        {
            "model": "Per-Unit Linear",
            "actual": actual,
            "predicted": pred
        }
        for actual, pred in zip(
            test_group["battery_capacity_pct"],
            predictions
        )
    ])

# --------------------------------------------------
# Model 2: Per-unit / per-segment Exponential Regression
# Fit log(capacity) as a linear function of cycles
# --------------------------------------------------

for (unit, segment), train_group in train.groupby(
    ["unit_id", "battery_segment"]
):
    test_group = test[
        (test["unit_id"] == unit)
        & (test["battery_segment"] == segment)
    ]

    if test_group.empty:
        continue

    model = LinearRegression()

    model.fit(
        train_group[["cycles_since_repair"]],
        np.log(train_group["battery_capacity_pct"])
    )

    log_predictions = model.predict(
        test_group[["cycles_since_repair"]]
    )

    predictions = np.clip(
        np.exp(log_predictions),
        50,
        100
    )

    results.extend([
        {
            "model": "Per-Unit Exponential",
            "actual": actual,
            "predicted": pred
        }
        for actual, pred in zip(
            test_group["battery_capacity_pct"],
            predictions
        )
    ])

# --------------------------------------------------
# Model 3: Pooled Linear Regression
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

train_units = encoder.fit_transform(
    train[["unit_id"]]
)

test_units = encoder.transform(
    test[["unit_id"]]
)

train_features = np.column_stack([
    train["cycles_since_repair"].to_numpy(),
    train["temperature_c"].to_numpy(),
    train_units
])

test_features = np.column_stack([
    test["cycles_since_repair"].to_numpy(),
    test["temperature_c"].to_numpy(),
    test_units
])

pooled_model = LinearRegression()

pooled_model.fit(
    train_features,
    train["battery_capacity_pct"]
)

pooled_predictions = np.clip(
    pooled_model.predict(test_features),
    50,
    100
)

results.extend([
    {
        "model": "Pooled Linear",
        "actual": actual,
        "predicted": pred
    }
    for actual, pred in zip(
        test["battery_capacity_pct"],
        pooled_predictions
    )
])

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df["absolute_error"] = (
    results_df["actual"]
    - results_df["predicted"]
).abs()

summary = (
    results_df.groupby("model")
    .agg(
        MAE=("absolute_error", "mean"),
        Predictions=("absolute_error", "size")
    )
    .sort_values("MAE")
)

results_df.to_csv(
    REPORT_DIR / "W05_battery_model_predictions.csv",
    index=False
)

summary.to_csv(
    REPORT_DIR / "W05_battery_model_comparison.csv"
)

print("\n=== Battery Model Comparison ===")
print(summary.round(4).to_string())

print("\n=== Pooled Model Coefficients ===")
print(
    "Cycles coefficient:",
    round(pooled_model.coef_[0], 6)
)
print(
    "Temperature coefficient:",
    round(pooled_model.coef_[1], 6)
)
