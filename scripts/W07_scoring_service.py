


import argparse
import json

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd


MODEL_PATH = Path("models/W07/rf_platt.joblib")
STATUS_PATH = Path("data/W07/W07_operational_status.csv")
OUTPUT_ROOT = Path("reports/W07/scoring_runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--cost-ratio", type=int, default=20,
        choices=[5, 10, 20]
    )
    parser.add_argument(
        "--capacity", type=int, default=5
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=STATUS_PATH
    )
    args = parser.parse_args()

    if args.capacity < 1:
        raise ValueError("Capacity must be positive")

    as_of = pd.Timestamp(args.as_of)

    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    if not args.status_file.is_file():
        raise FileNotFoundError(args.status_file)

    bundle = joblib.load(MODEL_PATH)

    timestamp = datetime.now(timezone.utc)
    run_id = (
        timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + uuid4().hex[:8]
    )

    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    output_path = run_dir / "service_queue.csv"

    command = [
        sys.executable,
        "scripts/W07_decision_layer.py",
        "--as-of", as_of.strftime("%Y-%m-%d"),
        "--cost-ratio", str(args.cost_ratio),
        "--capacity", str(args.capacity),
        "--status-file", str(args.status_file),
        "--output", str(output_path),
    ]

    # Existing decision layer currently writes a shared output.
    # Copy it immediately into this run-specific directory.
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    # shared_path = Path(
    #     f"reports/W07/W07_service_queue_"
    #     f"{as_of:%Y%m%d}_cost{args.cost_ratio}.csv"
    # )

    # if not shared_path.is_file():
    #     raise FileNotFoundError(shared_path)

    # output_path = run_dir / "service_queue.csv"
    # shutil.copy2(shared_path, output_path)

    if not output_path.is_file():
        raise FileNotFoundError(
            f"Scoring output not created: {output_path}"
        )

    scored = pd.read_csv(output_path)

    metadata = {
        "run_id": run_id,
        "scoring_timestamp_utc": timestamp.isoformat(),
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "model_version": bundle["model_version"],
        "feature_version": bundle["feature_version"],
        "cost_ratio": args.cost_ratio,
        "risk_threshold": float(
            scored["risk_threshold"].iloc[0]
        ),
        "capacity": args.capacity,
        "status_file": str(args.status_file),
        "robots_scored": int(len(scored)),
        "robots_above_threshold": int(
            scored["above_threshold"].sum()
        ),
        "robots_suppressed": int(
            scored["suppressed"].sum()
        ),
        "robots_selected": int(
            scored["selected_for_service"].sum()
        ),
        "historical_backfill": True,
        "data_type": "synthetic",
        "output_file": str(output_path),
    }

    metadata_path = run_dir / "run_metadata.json"

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(result.stdout)
    print("RUN ID:", run_id)
    print("SAVED QUEUE:", output_path)
    print("SAVED METADATA:", metadata_path)


if __name__ == "__main__":
    main()
