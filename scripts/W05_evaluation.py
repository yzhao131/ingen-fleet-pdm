
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

# Load backtesting predictions for all five models
baseline = pd.read_csv(
    REPORT_DIR / "W05_backtesting_results.csv"
)

advanced = pd.read_csv(
    REPORT_DIR / "W05_advanced_backtesting.csv"
)

results = pd.concat(
    [baseline, advanced],
    ignore_index=True
)

# Absolute errors
results["absolute_error"] = (
    results["actual"] - results["predicted"]
).abs()

# MAPE is undefined when actual == 0.
# Exclude zero-actual observations from MAPE.
results["APE"] = np.where(
    results["actual"] > 0,
    results["absolute_error"]
    / results["actual"] * 100,
    np.nan
)


def evaluate(group):
    actual_sum = group["actual"].sum()

    return pd.Series({
        "MAE": group["absolute_error"].mean(),
        "MAPE_nonzero_pct": group["APE"].mean(),
        "WAPE_pct": (
            group["absolute_error"].sum()
            / actual_sum * 100
            if actual_sum > 0 else np.nan
        ),
        "Forecasts": len(group),
        "Zero_actuals": int(
            (group["actual"] == 0).sum()
        )
    })


overall = (
    results.groupby("model")
    .apply(evaluate, include_groups=False)
    .sort_values("MAE")
)

by_horizon = (
    results.groupby(["model", "horizon"])
    .apply(evaluate, include_groups=False)
    .reset_index()
)

overall.to_csv(
    REPORT_DIR / "W05_full_model_metrics.csv"
)

by_horizon.to_csv(
    REPORT_DIR / "W05_metrics_by_horizon.csv",
    index=False
)

print("=== Overall Model Evaluation ===")
print(overall.round(2).to_string())

print("\n=== MAE by Horizon ===")
print(
    by_horizon.pivot(
        index="horizon",
        columns="model",
        values="MAE"
    ).round(2).to_string()
)

print("\n=== MAPE (Nonzero Actuals) by Horizon ===")
print(
    by_horizon.pivot(
        index="horizon",
        columns="model",
        values="MAPE_nonzero_pct"
    ).round(2).to_string()
)
