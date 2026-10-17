from pathlib import Path
import os

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

N_ROBOTS = 50
N_SITES = 7

START_DATE = "2025-01-01"
END_DATE = "2026-06-30"

TELEMETRY_FREQ = "6h"


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = Path(os.environ.get("FLEET_OUTPUT_ROOT", str(PROJECT_ROOT / "data" / "causal_preview")))
RAW_DIR = OUTPUT_ROOT / "raw"
SEALED_DIR = OUTPUT_ROOT / "sealed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SEALED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Random number generator
# --------------------------------------------------

rng = np.random.default_rng(SEED)

# --------------------------------------------------
# Site generation
# --------------------------------------------------

def generate_sites():
    site_ids = [f"SITE_{i:02d}" for i in range(1, N_SITES + 1)]

    terrain_types = [
        "paved",
        "paved",
        "mixed",
        "mixed",
        "paved",
        "mixed",
        "rough",
    ]

    sites = pd.DataFrame({
        "site_id": site_ids,
        "terrain_type": terrain_types,
    })

    return sites

# --------------------------------------------------
# Robot generation
# --------------------------------------------------

def generate_robots(sites):
    unit_ids = [f"R{i:03d}" for i in range(1, N_ROBOTS + 1)]

    site_choices = rng.choice(
        sites["site_id"],
        size=N_ROBOTS,
        replace=True
    )

    commission_dates = pd.to_datetime(
        rng.choice(
            pd.date_range("2023-01-01", "2024-12-31"),
            size=N_ROBOTS,
            replace=True
        )
    )

    robots = pd.DataFrame({
        "unit_id": unit_ids,
        "site_id": site_choices,
        "commission_date": commission_dates,
        "model": "Aido Rover",
    })

    robots = robots.merge(
        sites,
        on="site_id",
        how="left"
    )

    return robots

# --------------------------------------------------
# Telemetry generation
# --------------------------------------------------

def generate_telemetry(robots, missions):
    timestamps = pd.date_range(
        start=START_DATE,
        end=f"{END_DATE} 18:00:00",
        freq=TELEMETRY_FREQ,
        tz="UTC"
    )

    telemetry_frames = []

    for _, robot in robots.iterrows():
        unit_id = robot["unit_id"]
        terrain = robot["terrain_type"]

        n = len(timestamps)

            # Historical mission usage available at each telemetry timestamp
        robot_missions = missions[
            missions["unit_id"] == unit_id
        ].sort_values("end_time")

        cumulative_hours = np.zeros(n)

        if not robot_missions.empty:
            mission_end_times = robot_missions["end_time"].to_numpy(
                dtype="datetime64[ns]"
            )
            mission_hours = robot_missions["operating_hours"].to_numpy()

            cumulative_mission_hours = np.cumsum(mission_hours)

            completed_counts = np.searchsorted(
                mission_end_times,
                timestamps.to_numpy(dtype="datetime64[ns]"),
                side="right"
            )

            valid = completed_counts > 0

            cumulative_hours[valid] = cumulative_mission_hours[
                completed_counts[valid] - 1
            ]

        # Time progression through the simulation
        days_since_start = (
            timestamps - timestamps[0]
        ).total_seconds() / (24 * 60 * 60)

        # Seasonal temperature pattern + random variation
        seasonal_temp = 20 + 10 * np.sin(
            2 * np.pi * days_since_start / 365
        )

        temperature_c = (
            seasonal_temp
            + rng.normal(0, 4, size=n)
        )

        # Approximate cumulative charge cycles
        daily_usage = rng.uniform(0.6, 1.4)
        charge_cycles = np.floor(
            cumulative_hours / 4.0
        ).astype(int)

        # Battery degradation
        extreme_temp = np.maximum(
            np.abs(temperature_c - 20) - 10,
            0
        )

        battery_capacity_pct = (
            100
            - 0.018 * charge_cycles
            - 0.003 * extreme_temp * days_since_start
              / 30
            + rng.normal(0, 0.5, size=n)
        )

        battery_capacity_pct = np.clip(
            battery_capacity_pct,
            50,
            100
        )

        # Wheel wear
        terrain_multiplier = {
            "paved": 1.0,
            "mixed": 1.4,
            "rough": 2.0
        }[terrain]

        wheel_wear_pct = (
            cumulative_hours
            * 0.008
            * terrain_multiplier
            + rng.normal(0, 0.3, size=n)
        )

        wheel_wear_pct = np.clip(
            wheel_wear_pct,
            0,
            100
        )

        # Normal motor-current behaviour
        # Bearing degradation increases with cumulative operating hours.
        # Rougher terrain accelerates degradation.
        bearing_degradation = (
            cumulative_hours * 0.00035 * terrain_multiplier
        )

        bearing_degradation = np.clip(
            bearing_degradation,
            0,
            1
        )

        # Degraded bearings increase motor current.
        # Wheel wear also contributes to motor load.
        motor_current_a = (
            4.0
            + 2.0 * bearing_degradation
            + 0.002 * wheel_wear_pct
            + rng.normal(0, 0.25, size=n)
        )

        motor_current_a = np.clip(
            motor_current_a,
            0,
            None
        )

        robot_telemetry = pd.DataFrame({
            "unit_id": unit_id,
            "timestamp": timestamps,
            "battery_capacity_pct": battery_capacity_pct,
            "charge_cycles": charge_cycles,
            "motor_current_a": motor_current_a,
            "wheel_wear_pct": wheel_wear_pct,
            "temperature_c": temperature_c,
            "bearing_degradation": bearing_degradation
        })

        telemetry_frames.append(robot_telemetry)

    telemetry = pd.concat(
        telemetry_frames,
        ignore_index=True
    )

    return telemetry

# --------------------------------------------------
# Mission generation
# --------------------------------------------------

def generate_missions(robots):
    mission_records = []
    mission_counter = 1

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
        tz="UTC"
    )

    for _, robot in robots.iterrows():
        unit_id = robot["unit_id"]

        for date in dates:

            # Some days the robot does not operate.
            if rng.random() < 0.15:
                continue

            # An active robot performs 1-3 missions that day.
            n_missions = rng.integers(1, 4)

            for _ in range(n_missions):
                start_hour = int(
                    rng.integers(6, 18)
                )

                duration_hours = rng.uniform(
                    0.5,
                    4.0
                )

                start_time = (
                    date
                    + pd.Timedelta(hours=start_hour)
                )

                end_time = (
                    start_time
                    + pd.Timedelta(hours=duration_hours)
                )

                distance_km = (
                    duration_hours
                    * rng.uniform(2.0, 6.0)
                )

                mission_records.append({
                    "mission_id": f"M{mission_counter:06d}",
                    "unit_id": unit_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "operating_hours": duration_hours,
                    "distance_km": distance_km
                })

                mission_counter += 1

    missions = pd.DataFrame(mission_records)

    return missions

 # --------------------------------------------------
# Fault generation
# --------------------------------------------------

def generate_faults(robots, telemetry):
    fault_records = []
    fault_counter = 1

    for _, robot in robots.iterrows():
        unit_id = robot["unit_id"]

        robot_data = telemetry[
            telemetry["unit_id"] == unit_id
        ].sort_values("timestamp").copy()

        # Use one observation per day
        robot_data["date"] = robot_data["timestamp"].dt.floor("D")

        daily_data = (
            robot_data.groupby("date", as_index=False)
            .agg({
                "battery_capacity_pct": "mean",
                "wheel_wear_pct": "mean",
                "motor_current_a": "mean",
                "bearing_degradation": "mean"
            })
        )

        # Use previous-day sensor readings for today's fault risk
        sensor_columns = [
            "battery_capacity_pct",
            "wheel_wear_pct",
            "motor_current_a",
            "bearing_degradation"
        ]

        daily_data[sensor_columns] = (
            daily_data[sensor_columns].shift(1)
        )

        daily_data = daily_data.dropna(
            subset=sensor_columns
        )

        for _, row in daily_data.iterrows():

            # Battery degradation increases battery fault risk
            battery_risk = max(
                0,
                (95 - row["battery_capacity_pct"]) / 45
            )

            # Wheel wear increases wheel fault risk
            wheel_risk = row["wheel_wear_pct"] / 100

            # Higher motor current increases bearing fault risk
            # Bearing fault risk depends on the underlying degradation.
            bearing_risk = np.clip(
                row["bearing_degradation"],
                0,
                1
            )

            # Daily fault probabilities
            fault_probabilities = {
                "battery": 0.001 + 0.012 * battery_risk,
                "wheel": 0.001 + 0.015 * wheel_risk,
                "bearing": 0.001 + 0.015 * bearing_risk,
                "random": 0.001
            }

            for fault_type, probability in fault_probabilities.items():

                if rng.random() < probability:

                    severity = int(
                        rng.choice(
                            [1, 2, 3, 4, 5],
                            p=[0.15, 0.20, 0.30, 0.25, 0.10]
                        )
                    )

                    occurrence_hour = int(
                        rng.integers(0, 24)
                    )

                    occurred_at = (
                        row["date"]
                        + pd.Timedelta(hours=occurrence_hour)
                    )

                    logging_delay_hours = float(
                        rng.uniform(0, 12)
                    )

                    logged_at = (
                        occurred_at
                        + pd.Timedelta(hours=logging_delay_hours)
                    )

                    fault_records.append({
                        "fault_id": f"F{fault_counter:06d}",
                        "unit_id": unit_id,
                        "occurred_at": occurred_at,
                        "logged_at": logged_at,
                        "fault_type": fault_type,
                        "severity": severity
                    })

                    fault_counter += 1

    faults = pd.DataFrame(
        fault_records,
        columns=[
            "fault_id",
            "unit_id",
            "occurred_at",
            "logged_at",
            "fault_type",
            "severity"
        ]
    )

    return faults

# --------------------------------------------------
# Maintenance generation
# --------------------------------------------------

def generate_maintenance(faults):
    maintenance_records = []
    maintenance_counter = 1

    # Severe faults usually trigger maintenance.
    severe_faults = faults[faults["severity"] >= 3]

    for _, fault in severe_faults.iterrows():

        # 85% of severe faults result in maintenance.
        if rng.random() > 0.85:
            continue

        delay_hours = rng.uniform(1, 12)
        duration_hours = rng.uniform(6, 48)

        start_time = (
            fault["occurred_at"]
            + pd.Timedelta(hours=delay_hours)
        )

        end_time = (
            start_time
            + pd.Timedelta(hours=duration_hours)
        )

        maintenance_records.append({
            "maintenance_id": f"MT{maintenance_counter:06d}",
            "unit_id": fault["unit_id"],
            "start_time": start_time,
            "end_time": end_time,
            "reason": fault["fault_type"]
        })

        maintenance_counter += 1

    maintenance = pd.DataFrame(maintenance_records)

    return maintenance

# --------------------------------------------------
# Bearing failure precursor
# --------------------------------------------------

def inject_bearing_precursors(telemetry, faults):
    telemetry = telemetry.copy()

    bearing_faults = faults[
        (faults["fault_type"] == "bearing")
        & (faults["severity"] >= 3)
    ]

    precursor_records = []

    for _, fault in bearing_faults.iterrows():
        unit_id = fault["unit_id"]
        failure_time = fault["occurred_at"]

        # Required precursor duration: 5-15 days.
        precursor_days = int(rng.integers(5, 16))

        precursor_start = (
            failure_time
            - pd.Timedelta(days=precursor_days)
        )

        mask = (
            (telemetry["unit_id"] == unit_id)
            & (telemetry["timestamp"] >= precursor_start)
            & (telemetry["timestamp"] < failure_time)
        )

        indices = telemetry.index[mask]

        if len(indices) == 0:
            continue

        # Gradually increase motor current as failure approaches.
        increase = np.linspace(
            0,
            rng.uniform(1.5, 3.0),
            len(indices)
        )

        telemetry.loc[
            indices,
            "motor_current_a"
        ] += increase

        precursor_records.append({
            "fault_id": fault["fault_id"],
            "unit_id": unit_id,
            "fault_type": "bearing",
            "precursor_start": precursor_start,
            "failure_time": failure_time,
            "precursor_days": precursor_days
        })

    precursor_truth = pd.DataFrame(
        precursor_records
    )

    return telemetry, precursor_truth

# --------------------------------------------------
# Data imperfection injection
# --------------------------------------------------

def inject_telemetry_imperfections(telemetry):
    telemetry = telemetry.copy()

    truth_records = []

    # ----------------------------------------------
    # 1. Clock drift
    # ----------------------------------------------

    affected_units = rng.choice(
        telemetry["unit_id"].unique(),
        size=8,
        replace=False
    )

    for unit_id in affected_units:
        drift_minutes = int(
            rng.integers(-20, 21)
        )

        # Make sure drift is not zero.
        if drift_minutes == 0:
            drift_minutes = 10

        mask = telemetry["unit_id"] == unit_id

        telemetry.loc[mask, "timestamp"] += pd.Timedelta(
            minutes=drift_minutes
        )

        truth_records.append({
            "imperfection": "clock_drift",
            "unit_id": unit_id,
            "detail": f"{drift_minutes} minutes"
        })

    # ----------------------------------------------
    # 2. Connectivity gaps
    # ----------------------------------------------

    for _ in range(30):
        unit_id = rng.choice(
            telemetry["unit_id"].unique()
        )

        unit_rows = telemetry[
            telemetry["unit_id"] == unit_id
        ]

        start_index = rng.choice(
            unit_rows.index
        )

        gap_start = telemetry.loc[
            start_index,
            "timestamp"
        ]

        gap_hours = int(
            rng.choice([6, 12, 18, 24])
        )

        gap_end = (
            gap_start
            + pd.Timedelta(hours=gap_hours)
        )

        mask = (
            (telemetry["unit_id"] == unit_id)
            & (telemetry["timestamp"] >= gap_start)
            & (telemetry["timestamp"] < gap_end)
        )

        rows_removed = int(mask.sum())

        telemetry = telemetry.loc[~mask].copy()

        truth_records.append({
            "imperfection": "connectivity_gap",
            "unit_id": unit_id,
            "detail": (
                f"{gap_start} to {gap_end}; "
                f"{rows_removed} rows removed"
            )
        })

    # ----------------------------------------------
    # 3. Duplicate rows
    # ----------------------------------------------

    duplicate_rows = telemetry.sample(
        n=200,
        random_state=SEED
    )

    telemetry = pd.concat(
        [telemetry, duplicate_rows],
        ignore_index=True
    )

    truth_records.append({
        "imperfection": "duplicates",
        "unit_id": "multiple",
        "detail": "200 duplicate telemetry rows"
    })

    # Shuffle so duplicate rows are not all at bottom.
    telemetry = telemetry.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    imperfection_truth = pd.DataFrame(
        truth_records
    )

    return telemetry, imperfection_truth

    # --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_generated_data(
    sites,
    robots,
    telemetry,
    missions,
    faults,
    maintenance
):
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)

    # Fleet size
    print(f"7 sites: {len(sites) == 7}")
    print(f"50 robots: {len(robots) == 50}")

    # Robot IDs
    print(
        "Robot IDs unique:",
        robots["unit_id"].is_unique
    )

    # UTC timestamps
    print(
        "Telemetry timestamps timezone-aware:",
        telemetry["timestamp"].dt.tz is not None
    )

    # Battery range
    print(
        "Battery capacity valid:",
        telemetry["battery_capacity_pct"]
        .between(50, 100)
        .all()
    )

    # Wheel wear range
    print(
        "Wheel wear valid:",
        telemetry["wheel_wear_pct"]
        .between(0, 100)
        .all()
    )

    # Fault severity
    print(
        "Fault severity valid:",
        faults["severity"]
        .between(1, 5)
        .all()
    )

    # Delayed logging
    print(
        "Fault logging occurs at/after occurrence:",
        (
            faults["logged_at"]
            >= faults["occurred_at"]
        ).all()
    )

    # Duplicate telemetry
    print(
        "Duplicate telemetry rows:",
        telemetry.duplicated().sum()
    )

    # Required fault types
    required_fault_types = {
        "bearing",
        "battery",
        "wheel",
        "random"
    }

    print(
        "Required fault types present:",
        required_fault_types.issubset(
            set(faults["fault_type"])
        )
    )

    # Maintenance validity
    print(
        "Maintenance intervals valid:",
        (
            maintenance["end_time"]
            > maintenance["start_time"]
        ).all()
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

def simulate_causal_fleet(robots, missions):
    """Chronological fleet simulator: usage -> wear -> faults -> repair -> recovery.

    Fault probabilities use health as of the beginning of each day. Repairs
    reduce component wear only after maintenance completion. The latent bearing
    health is used internally but is not included in the public telemetry CSV.
    """
    timestamps = pd.date_range(
        START_DATE, f"{END_DATE} 18:00:00", freq=TELEMETRY_FREQ, tz="UTC"
    )
    telemetry_records, fault_records, maintenance_records = [], [], []
    mission_status = {}
    fault_counter = maintenance_counter = 1
    terrain_factors = {"paved": 1.0, "mixed": 1.4, "rough": 2.0}

    for _, robot in robots.iterrows():
        unit_id = robot["unit_id"]
        terrain_factor = terrain_factors[robot["terrain_type"]]
        robot_missions = missions.loc[missions["unit_id"] == unit_id].sort_values("end_time")
        mission_ends = list(robot_missions.itertuples(index=False))
        next_mission = 0
        operating_hours = 0.0
        battery_wear = 0.0
        wheel_wear = 0.0
        bearing_wear = 0.0
        repairs = []
        active_intervals = []
        last_time = timestamps[0] - pd.Timedelta(hours=6)

        for timestamp in timestamps:
            # Apply repairs only after they finish, never retrospectively.
            for repair in repairs:
                if not repair["applied"] and repair["end"] <= timestamp:
                    if repair["reason"] == "battery":
                        battery_wear *= 0.20
                    elif repair["reason"] == "wheel":
                        wheel_wear *= 0.20
                    elif repair["reason"] == "bearing":
                        bearing_wear *= 0.20
                    else:
                        battery_wear *= 0.90
                        wheel_wear *= 0.90
                        bearing_wear *= 0.90
                    repair["applied"] = True

            # Only completed missions contribute usage. Exclude missions
            # overlapping known maintenance downtime.
            while next_mission < len(mission_ends) and mission_ends[next_mission].end_time <= timestamp:
                mission = mission_ends[next_mission]
                overlaps_repair = any(
                    mission.start_time < end and mission.end_time > start
                    for start, end in active_intervals
                )

                if overlaps_repair:
                    mission_status[mission.mission_id] = "cancelled_maintenance"
                else:
                    mission_status[mission.mission_id] = "completed"

                    hours = float(mission.operating_hours)
                    operating_hours += hours
                    battery_wear += 0.018 * hours / 4.0
                    wheel_wear += 0.008 * terrain_factor * hours
                    bearing_wear += 0.00035 * terrain_factor * hours

                next_mission += 1

            # Once per day, sample faults using state known before that day.
            # No future telemetry or fault-conditioned precursor injection.
            if timestamp.hour == 0 and timestamp > timestamps[0]:
                battery_capacity = max(50.0, 100.0 - battery_wear)
                probabilities = {
                    "battery": 0.001 + 0.012 * max(0, (95 - battery_capacity) / 45),
                    "wheel": 0.001 + 0.015 * min(1.0, wheel_wear / 100),
                    "bearing": 0.001 + 0.015 * min(1.0, bearing_wear),
                    "random": 0.001,
                }
                # One fault at most per robot-day.
                if not any(start <= timestamp < end for start, end in active_intervals):
                    for fault_type, probability in probabilities.items():
                        if rng.random() >= probability:
                            continue
                        severity = int(rng.choice([1, 2, 3, 4, 5], p=[.15, .20, .30, .25, .10]))
                        occurred_at = timestamp + pd.Timedelta(
                            hours=int(rng.integers(0, 24))
                        )

                        # Do not generate a fault during scheduled maintenance.
                        if any(
                            start <= occurred_at < end
                            for start, end in active_intervals
                        ):
                            continue

                        logged_at = occurred_at + pd.Timedelta(
                            hours=float(rng.uniform(0, 12))
                        )

                        fault_records.append({
                            "fault_id": f"F{fault_counter:06d}", "unit_id": unit_id,
                            "occurred_at": occurred_at, "logged_at": logged_at,
                            "fault_type": fault_type, "severity": severity,
                        })
                        fault_counter += 1
                        if severity >= 3 and rng.random() < 0.85:
                            repair_start = occurred_at + pd.Timedelta(hours=float(rng.uniform(1, 12)))
                            repair_end = repair_start + pd.Timedelta(hours=float(rng.uniform(6, 48)))
                            maintenance_records.append({
                                "maintenance_id": f"MT{maintenance_counter:06d}",
                                "unit_id": unit_id, "start_time": repair_start,
                                "end_time": repair_end, "reason": fault_type,
                            })
                            maintenance_counter += 1
                            repairs.append({"end": repair_end, "reason": fault_type, "applied": False})
                            active_intervals.append((repair_start, repair_end))
                        break

            days = (timestamp - timestamps[0]).total_seconds() / 86400
            temp = 20 + 10 * np.sin(2 * np.pi * days / 365) + rng.normal(0, 4)
            capacity = np.clip(100 - battery_wear + rng.normal(0, .5), 50, 100)
            wheel = np.clip(wheel_wear + rng.normal(0, .3), 0, 100)
            current = max(0, 4 + 2 * min(1, bearing_wear) + .002 * wheel + rng.normal(0, .25))
            telemetry_records.append({
                "unit_id": unit_id, "timestamp": timestamp,
                "battery_capacity_pct": capacity,
                "charge_cycles": int(operating_hours // 4),
                "motor_current_a": current, "wheel_wear_pct": wheel,
                "temperature_c": temp,
            })
            last_time = timestamp

    telemetry = pd.DataFrame(telemetry_records)
    faults = pd.DataFrame(fault_records, columns=[
        "fault_id", "unit_id", "occurred_at", "logged_at", "fault_type", "severity"
    ])
    maintenance = pd.DataFrame(maintenance_records, columns=[
        "maintenance_id", "unit_id", "start_time", "end_time", "reason"
    ])
    return telemetry, faults, maintenance, mission_status


def main():
    sites = generate_sites()
    robots = generate_robots(sites)
    missions = generate_missions(robots)
    telemetry, faults, maintenance, mission_status = simulate_causal_fleet(
        robots, missions
    )

    missions["mission_status"] = (
        missions["mission_id"].map(mission_status)
    )

    missions["mission_status"] = missions["mission_status"].fillna(
        "not_observed"
    )

    # telemetry, precursor_truth = inject_bearing_precursors(
    #     telemetry,
    #     faults
    # )
    precursor_truth = pd.DataFrame(
        columns=[
            "fault_id",
            "unit_id",
            "fault_type",
            "precursor_start",
            "failure_time",
            "precursor_days"
        ]
    )
    telemetry, imperfection_truth = inject_telemetry_imperfections(
    telemetry
    )


    sites.to_csv(
        RAW_DIR / "sites.csv",
        index=False
    )

    robots.to_csv(
        RAW_DIR / "robots.csv",
        index=False
    )

    telemetry.to_csv(
        RAW_DIR / "telemetry.csv",
        index=False
    )

    missions.to_csv(
    RAW_DIR / "missions.csv",
    index=False
    )

    faults.to_csv(
    RAW_DIR / "faults.csv",
    index=False
    )

    maintenance.to_csv(
    RAW_DIR / "maintenance.csv",
    index=False
    )

    precursor_truth.to_csv(
    SEALED_DIR / "bearing_precursor_truth.csv",
    index=False
    )

    imperfection_truth.to_csv(
    SEALED_DIR / "imperfection_truth.csv",
    index=False
    )

    print("Generation complete.")
    print(f"Sites: {len(sites):,}")
    print(f"Robots: {len(robots):,}")
    print(f"Telemetry rows: {len(telemetry):,}")
    print("\nTelemetry sample:")
    print(telemetry.head())

    print(f"Missions: {len(missions):,}")
    print(f"Faults: {len(faults):,}")
    print("\nFault types:")
    print(faults["fault_type"].value_counts())

    print("\nSeverity distribution:")
    print(faults["severity"].value_counts().sort_index())

    print(f"Maintenance events: {len(maintenance):,}")
    print(f"Retrospective bearing precursors planted: {len(precursor_truth):,} (disabled)")

    print(
    f"Final telemetry rows after imperfections: "
    f"{len(telemetry):,}"
    )

    print(
    f"Exact duplicate rows: "
    f"{telemetry.duplicated().sum():,}"
    )

    validate_generated_data(
    sites,
    robots,
    telemetry,
    missions,
    faults,
    maintenance
    )

if __name__ == "__main__":
    main()