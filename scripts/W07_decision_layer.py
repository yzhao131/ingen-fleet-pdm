

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURE_PATH = Path(
    "data/causal_preview/processed/W03_robot_day_features.csv"
)
MODEL_PATH = Path("models/W07/rf_platt.joblib")
STATUS_PATH = Path("data/W07/W07_operational_status.csv")
THRESHOLD_PATH = Path(
    "reports/W04/W04_threshold_sensitivity.csv"
)
OUTPUT_DIR = Path("reports/W07")

MODEL_NAME = "rf_depth4_leaf50"
CALIBRATION = "platt"


def get_threshold(cost_ratio):
    df = pd.read_csv(THRESHOLD_PATH)

    match = df[
        (df["model"] == MODEL_NAME)
        & (df["calibration"] == CALIBRATION)
        & (df["fn_to_fp_cost_ratio"] == cost_ratio)
    ]

    if len(match) != 1:
        raise ValueError("Expected exactly one threshold")

    return float(match.iloc[0]["threshold"])


def apply_suppression(queue, statuses, as_of,
                      cooldown_days=7,
                      escalation_factor=1.5):
    queue = queue.copy()
    queue["suppressed"] = False
    queue["suppression_reason"] = ""

    if statuses.empty:
        return queue

    statuses = statuses.copy()
    statuses["status_time"] = pd.to_datetime(
        statuses["status_time"], utc=True
    )

    # Never use operational events recorded after scoring time.
    statuses = statuses[
        statuses["status_time"] <= as_of
    ].sort_values("status_time")

    if statuses.empty:
        return queue

    latest = statuses.drop_duplicates(
        "unit_id", keep="last"
    )

    for idx, row in queue.iterrows():
        record = latest[
            latest["unit_id"] == row["unit_id"]
        ]

        if record.empty:
            continue

        status = record.iloc[0]
        state = str(status["status"]).strip().lower()

        if state == "scheduled_service":
            scheduled = pd.to_datetime(
                status["scheduled_service_date"],
                utc=True,
                errors="coerce",
            )

            if pd.notna(scheduled) and scheduled >= as_of:
                queue.at[idx, "suppressed"] = True
                queue.at[idx, "suppression_reason"] = (
                    "Service already scheduled"
                )

        elif state == "inspection_completed":
            result = str(
                status["inspection_result"]
            ).strip().lower()

            elapsed = (
                as_of - status["status_time"]
            ).total_seconds() / 86400

            if result == "no_finding" and (
                0 <= elapsed < cooldown_days
            ):
                previous_risk = pd.to_numeric(
                    status["risk_at_inspection"],
                    errors="coerce",
                )

                # Missing baseline risk cannot establish escalation.
                if pd.isna(previous_risk):
                    queue.at[idx, "suppressed"] = True
                    queue.at[idx, "suppression_reason"] = (
                        "Inspection cooldown; baseline risk missing"
                    )
                elif row["calibrated_risk"] < (
                    previous_risk * escalation_factor
                ):
                    queue.at[idx, "suppressed"] = True
                    queue.at[idx, "suppression_reason"] = (
                        "Negative inspection cooldown"
                    )

    return queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--cost-ratio", type=int, default=10,
        choices=[5, 10, 20]
    )
    parser.add_argument(
        "--capacity", type=int, default=5
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=STATUS_PATH,
        help="Operational status CSV for suppression rules",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom CSV output path",
    )
    args = parser.parse_args()

    if args.capacity < 1:
        raise ValueError("Capacity must be positive")

    as_of = pd.Timestamp(args.as_of)

    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # The feature table is a historical, precomputed snapshot.
    # Only exact snapshot dates are supported by this script.
    daily = df[df["date"] == as_of].copy()

    if daily.empty:
        raise ValueError(
            f"No feature snapshot for {as_of.date()}"
        )

    if daily["unit_id"].duplicated().any():
        raise ValueError("Duplicate units in scoring snapshot")

    feature_cols = bundle["feature_columns"]

    raw_risk = bundle["pipeline"].predict_proba(
        daily[feature_cols]
    )[:, 1]

    daily["calibrated_risk"] = (
        bundle["calibrator"].predict_proba(
            raw_risk.reshape(-1, 1)
        )[:, 1]
    )

    threshold = get_threshold(args.cost_ratio)

    daily["above_threshold"] = (
        daily["calibrated_risk"] >= threshold
    )

    if args.status_file.exists():
        statuses = pd.read_csv(args.status_file)
    else:
        raise FileNotFoundError(
            f"Operational status file missing: {args.status_file}"
        )

    daily = apply_suppression(
        daily, statuses, as_of
    )

    daily["eligible"] = (
        daily["above_threshold"]
        & ~daily["suppressed"]
    )

    daily = daily.sort_values(
        ["calibrated_risk", "unit_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    daily["queue_rank"] = pd.NA
    eligible_idx = daily.index[daily["eligible"]]

    for rank, idx in enumerate(eligible_idx, start=1):
        daily.at[idx, "queue_rank"] = rank

    daily["selected_for_service"] = False
    selected_idx = eligible_idx[:args.capacity]
    daily.loc[
        selected_idx, "selected_for_service"
    ] = True

    daily["recommended_action"] = np.select(
        [
            daily["suppressed"],
            daily["selected_for_service"],
            daily["eligible"],
        ],
        [
            "Do not alert: suppression active",
            "Schedule diagnostic inspection",
            "Defer: technician capacity reached",
        ],
        default="Routine monitoring",
    )

    daily["model_version"] = bundle["model_version"]
    daily["feature_version"] = bundle["feature_version"]
    daily["cost_ratio"] = args.cost_ratio
    daily["risk_threshold"] = threshold
    daily["daily_capacity"] = args.capacity

    output_cols = [
        "date",
        "unit_id",
        "site_id",
        "calibrated_risk",
        "days_since_last_maintenance",
        "above_threshold",
        "suppressed",
        "suppression_reason",
        "eligible",
        "queue_rank",
        "selected_for_service",
        "recommended_action",
        "model_version",
        "feature_version",
        "cost_ratio",
        "risk_threshold",
        "daily_capacity",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_output = OUTPUT_DIR / (
        f"W07_service_queue_{as_of:%Y%m%d}"
        f"_cost{args.cost_ratio}.csv"
    )

    output_path = (
        args.output if args.output is not None
        else default_output
    )

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    daily[output_cols].to_csv(
        output_path, index=False
    )

    print("Scoring date:", as_of.date())
    print("Robots scored:", len(daily))
    print("Threshold:", threshold)
    print("Above threshold:", int(
        daily["above_threshold"].sum()
    ))
    print("Suppressed:", int(
        daily["suppressed"].sum()
    ))
    print("Selected for service:", int(
        daily["selected_for_service"].sum()
    ))
    print("Output:", output_path)


if __name__ == "__main__":
    main()
