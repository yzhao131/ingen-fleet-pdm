

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DATA_PATH = Path(
    "data/causal_preview/processed/W03_robot_day_features.csv"
)
OUTPUT_PATH = Path(
    "reports/W07/W07_feature_drift.csv"
)

FEATURES = [
    "battery_capacity_pct_7d_mean",
    "motor_current_a_7d_mean",
    "wheel_wear_pct_7d_mean",
    "temperature_c_7d_mean",
    "operating_hours_7d",
    "fault_count_7d",
    "days_since_last_maintenance",
    "telemetry_coverage_7d",
]


def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(
        reference, errors="coerce"
    ).dropna().to_numpy(dtype=float)

    current = pd.to_numeric(
        current, errors="coerce"
    ).dropna().to_numpy(dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Derive fixed bin boundaries from baseline only.
    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
    )

    if len(edges) < 2:
        # Constant baseline: separate the reference value
        # from any shifted values.
        value = reference[0]
        edges = np.array(
            [-np.inf, value, np.inf]
        )
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference, bins=edges
    )
    cur_counts, _ = np.histogram(
        current, bins=edges
    )

    # Add small smoothing to avoid log(0).
    eps = 1e-6

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    ref_pct = np.maximum(ref_pct, eps)
    cur_pct = np.maximum(cur_pct, eps)

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def drift_status(psi, ks):
    if np.isnan(psi) or np.isnan(ks):
        return "INSUFFICIENT_DATA"

    if psi >= 0.25 or ks >= 0.20:
        return "CRITICAL"

    if psi >= 0.10 or ks >= 0.10:
        return "WARNING"

    return "STABLE"


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    baseline = df[df["split"] == "train"]

    monitor = df[
        (df["split"] == "test")
        & (df["date"] >= "2026-03-08")
        & (df["date"] <= "2026-03-31")
    ]

    results = []

    for feature in FEATURES:
        reference = baseline[feature]
        current = monitor[feature]

        ref_valid = pd.to_numeric(
            reference, errors="coerce"
        ).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        cur_valid = pd.to_numeric(
            current, errors="coerce"
        ).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        psi = calculate_psi(
            ref_valid, cur_valid
        )

        if len(ref_valid) and len(cur_valid):
            ks = float(
                ks_2samp(
                    ref_valid,
                    cur_valid,
                ).statistic
            )
        else:
            ks = np.nan

        ref_missing = (
            1 - len(ref_valid) / len(reference)
        )
        cur_missing = (
            1 - len(cur_valid) / len(current)
        )

        missing_change = (
            cur_missing - ref_missing
        )

        status = drift_status(psi, ks)

        results.append({
            "feature": feature,
            "baseline_count": len(reference),
            "monitor_count": len(current),
            "baseline_missing_pct": round(
                ref_missing * 100, 2
            ),
            "monitor_missing_pct": round(
                cur_missing * 100, 2
            ),
            "missing_change_pp": round(
                missing_change * 100, 2
            ),
            "psi": psi,
            "ks_statistic": ks,
            "drift_status": status,
            "missingness_flag": (
                "REVIEW"
                if abs(missing_change) >= 0.10
                else "OK"
            ),
        })

    output = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.to_csv(OUTPUT_PATH, index=False)

    print("=== FEATURE DRIFT MONITORING ===")
    print(
        output[
            [
                "feature",
                "psi",
                "ks_statistic",
                "drift_status",
                "missing_change_pp",
                "missingness_flag",
            ]
        ].to_string(index=False)
    )

    print("\nOutput:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
