

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_PATH = Path(
    "data/causal_preview/processed/W03_robot_day_features.csv"
)
MODEL_PATH = Path("models/W07/rf_platt.joblib")
THRESHOLD_PATH = Path(
    "reports/W04/W04_threshold_sensitivity.csv"
)
OUTPUT_PATH = Path(
    "reports/W07/W07_impact_results.csv"
)

COST_RATIOS = [5, 10, 20]
CAPACITIES = [5, 10, 15]


def evaluate(y_true, selected, cost_ratio):
    y_true = np.asarray(y_true, dtype=int)
    selected = np.asarray(selected, dtype=bool)

    tp = int(((selected == 1) & (y_true == 1)).sum())
    fp = int(((selected == 1) & (y_true == 0)).sum())
    fn = int(((selected == 0) & (y_true == 1)).sum())
    tn = int(((selected == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    total_cost = fp + cost_ratio * fn
    no_alert_cost = cost_ratio * int(y_true.sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "alerts": int(selected.sum()),
        "total_cost_units": total_cost,
        "no_alert_cost_units": no_alert_cost,
        "cost_difference_vs_no_alert": (
            total_cost - no_alert_cost
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    test = df[df["split"] == "test"].copy()
    assert test["severe_fault_next_7d"].notna().all()
    assert not test.duplicated(["date", "unit_id"]).any()

    bundle = joblib.load(MODEL_PATH)

    raw = bundle["pipeline"].predict_proba(
        test[bundle["feature_columns"]]
    )[:, 1]

    test["risk"] = bundle["calibrator"].predict_proba(
        raw.reshape(-1, 1)
    )[:, 1]

    thresholds = pd.read_csv(THRESHOLD_PATH)
    thresholds = thresholds[
        (thresholds["model"] == "rf_depth4_leaf50")
        & (thresholds["calibration"] == "platt")
    ]

    results = []
    y = test["severe_fault_next_7d"]

    for ratio in COST_RATIOS:
        matched = thresholds[
            thresholds["fn_to_fp_cost_ratio"] == ratio
        ]
        assert len(matched) == 1
        threshold = float(matched.iloc[0]["threshold"])

        # No-alert baseline.
        metrics = evaluate(
            y, np.zeros(len(test), dtype=bool), ratio
        )
        results.append({
            "strategy": "no_alert",
            "cost_ratio": ratio,
            "threshold": threshold,
            "daily_capacity": None,
            **metrics,
        })

        # Threshold-only policy.
        eligible = test["risk"] >= threshold
        metrics = evaluate(y, eligible, ratio)

        results.append({
            "strategy": "threshold_only",
            "cost_ratio": ratio,
            "threshold": threshold,
            "daily_capacity": None,
            **metrics,
        })

        # Capacity-constrained policies.
        for capacity in CAPACITIES:
            ranked = test.copy()
            ranked["eligible"] = eligible

            ranked = ranked.sort_values(
                ["date", "risk", "unit_id"],
                ascending=[True, False, True],
            )

            ranked["daily_rank"] = (
                ranked[ranked["eligible"]]
                .groupby("date")
                .cumcount()
                .add(1)
                .reindex(ranked.index)
            )

            ranked["selected"] = (
                ranked["eligible"]
                & (ranked["daily_rank"] <= capacity)
            )

            selected = ranked["selected"].reindex(
                test.index
            ).fillna(False)

            metrics = evaluate(y, selected, ratio)

            results.append({
                "strategy": "capacity_limited",
                "cost_ratio": ratio,
                "threshold": threshold,
                "daily_capacity": capacity,
                **metrics,
            })

    output = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.to_csv(OUTPUT_PATH, index=False)

    print("=== WEEK 7 IMPACT ANALYSIS ===")
    print(
        output[
            [
                "strategy",
                "cost_ratio",
                "daily_capacity",
                "tp",
                "fp",
                "fn",
                "recall",
                "alerts",
                "total_cost_units",
                "cost_difference_vs_no_alert",
            ]
        ].to_string(index=False)
    )

    print("\nOutput:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
