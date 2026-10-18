
from pathlib import Path
import pandas as pd

# ==========================================
# 1. Set up directories
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "causal_preview" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "causal_preview" / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Define datetime columns
# ==========================================

datetime_columns = {
    "sites": [],
    "robots": ["commission_date"],
    "telemetry": ["timestamp"],
    "missions": ["start_time", "end_time"],
    "faults": ["occurred_at", "logged_at"],
    "maintenance": ["start_time", "end_time"],
}

# ==========================================
# 3. Clean each table
# ==========================================

for table, date_cols in datetime_columns.items():

    print(f"\nCleaning {table.upper()}")

    df = pd.read_csv(RAW_DIR / f"{table}.csv")

    original_rows = len(df)

    # Convert datetime columns to UTC
    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            utc=True,
            errors="raise"
        )

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()


    # ==========================================
    # Week 2: Clock drift correction
    # ==========================================

    if table == "telemetry":

        # Expected sampling interval: 6 hours
        grid_seconds = 6 * 3600

        # Calculate seconds since UTC midnight
        seconds_of_day = (
            df["timestamp"].dt.hour * 3600
            + df["timestamp"].dt.minute * 60
            + df["timestamp"].dt.second
        )

        # Signed offset from the nearest 6-hour grid
        offset_seconds = (
            (seconds_of_day + grid_seconds / 2)
            % grid_seconds
            - grid_seconds / 2
        )

        df["clock_offset_minutes"] = (
            offset_seconds / 60
        )

        # Estimate a fixed offset for each robot
        unit_offsets = (
            df.groupby("unit_id")["clock_offset_minutes"]
            .median()
        )

        # Preserve original timestamp for audit
        df["original_timestamp"] = df["timestamp"]

        # Correct timestamp using per-unit offset
        df["timestamp"] = (
            df["timestamp"]
            - pd.to_timedelta(
                df["unit_id"].map(unit_offsets),
                unit="m"
            )
        )

        # Validate corrected timestamps
        corrected_seconds = (
            df["timestamp"].dt.hour * 3600
            + df["timestamp"].dt.minute * 60
            + df["timestamp"].dt.second
        )

        assert (
            corrected_seconds % grid_seconds == 0
        ).all(), "Some timestamps are not aligned!"

        print(
            "Clock drift corrected for "
            f"{(unit_offsets.abs() > 1).sum()} robots"
        )

        df = df.drop(columns=["clock_offset_minutes"])


        # ==========================================
        # Week 2: Outlier flagging
        # ==========================================

        numeric_cols = [
            "battery_capacity_pct",
            "charge_cycles",
            "motor_current_a",
            "wheel_wear_pct",
            "temperature_c",
        ]

        for col in numeric_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df[f"{col}_outlier"] = (
                (df[col] < lower)
                | (df[col] > upper)
            )

        outlier_cols = [
            f"{col}_outlier"
            for col in numeric_cols
        ]

        df["has_any_outlier"] = (
            df[outlier_cols].any(axis=1)
        )

        print(
            "Rows with at least one outlier:",
            int(df["has_any_outlier"].sum())
        )



    removed_duplicates = original_rows - len(df)

    # Sort by appropriate keys
    sort_columns = {
        "sites": ["site_id"],
        "robots": ["unit_id"],
        "telemetry": ["unit_id", "timestamp"],
        "missions": ["unit_id", "start_time"],
        "faults": ["unit_id", "occurred_at"],
        "maintenance": ["unit_id", "start_time"],
    }

    df = df.sort_values(
        sort_columns[table]
    ).reset_index(drop=True)

    # Save cleaned data
    output_path = CLEAN_DIR / f"{table}.csv"

    df.to_csv(output_path, index=False)

    print(f"Original rows: {original_rows:,}")
    print(f"Removed duplicates: {removed_duplicates:,}")
    print(f"Cleaned rows: {len(df):,}")
    print(f"Saved to: {output_path}")

print("\nPart A cleaning completed successfully!")
