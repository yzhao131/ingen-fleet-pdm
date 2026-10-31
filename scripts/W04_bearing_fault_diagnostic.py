import pandas as pd
import numpy as np

base = "data/causal_preview/raw/"

telemetry = pd.read_csv(
    base + "telemetry.csv",
    parse_dates=["timestamp"]
)

faults = pd.read_csv(
    base + "faults.csv",
    parse_dates=["occurred_at"]
)

telemetry = telemetry.drop_duplicates(
    subset=["unit_id", "timestamp"]
)

bearing = faults[
    faults["fault_type"] == "bearing"
].copy()

results = []

for _, fault in bearing.iterrows():
    unit = fault["unit_id"]
    event_time = fault["occurred_at"]

    robot_data = telemetry[
        telemetry["unit_id"] == unit
    ]

    # Seven days immediately before the fault
    before = robot_data[
        (robot_data["timestamp"] >= event_time - pd.Timedelta(days=7))
        & (robot_data["timestamp"] < event_time)
    ]

    # Earlier historical comparison window
    control_end = event_time - pd.Timedelta(days=14)

    control = robot_data[
        (robot_data["timestamp"] >= control_end - pd.Timedelta(days=7))
        & (robot_data["timestamp"] < control_end)
    ]

    # Exclude control periods followed by bearing faults
    nearby_faults = bearing[
        (bearing["unit_id"] == unit)
        & (bearing["occurred_at"] >= control_end)
        & (bearing["occurred_at"] < control_end + pd.Timedelta(days=7))
    ]

    if (
        len(before) >= 10
        and len(control) >= 10
        and nearby_faults.empty
    ):
        results.append({
            "unit_id": unit,
            "fault_time": event_time,
            "before_current": before["motor_current_a"].mean(),
            "control_current": control["motor_current_a"].mean()
        })

result = pd.DataFrame(results)

print("Matched events:", len(result))

if not result.empty:
    result["difference"] = (
        result["before_current"] - result["control_current"]
    )

    print("\nMean current before faults:",
          round(result["before_current"].mean(), 4))

    print("Mean control current:",
          round(result["control_current"].mean(), 4))

    print("Mean difference:",
          round(result["difference"].mean(), 4))

    print("Percentage with higher pre-fault current:",
          round((result["difference"] > 0).mean() * 100, 2), "%")
    
# Check maintenance events between matched windows

maintenance = pd.read_csv(
    base + "maintenance.csv",
    parse_dates=["start_time", "end_time"]
)

maintenance_counts = []

for _, row in result.iterrows():
    unit = row["unit_id"]
    fault_time = row["fault_time"]

    window_start = fault_time - pd.Timedelta(days=21)

    related = maintenance[
        (maintenance["unit_id"] == unit)
        & (maintenance["start_time"] < fault_time)
        & (maintenance["end_time"] > window_start)
    ]

    maintenance_counts.append(len(related))

result["maintenance_count"] = maintenance_counts

print("\n=== Maintenance Confounding Check ===")
print(
    result["maintenance_count"]
    .value_counts()
    .sort_index()
)

clean = result[result["maintenance_count"] == 0]

print("\nMatched events without maintenance:", len(clean))

if not clean.empty:
    print(
        "Mean current difference without maintenance:",
        round(clean["difference"].mean(), 4)
    )

    print(
        "Percentage with higher pre-fault current:",
        round((clean["difference"] > 0).mean() * 100, 2),
        "%"
    )

# Bearing fault frequency over time

bearing = faults[
    faults["fault_type"] == "bearing"
].copy()

bearing["month"] = (
    bearing["occurred_at"]
    .dt.to_period("M")
    .astype(str)
)

monthly = (
    bearing.groupby("month")
    .agg(
        bearing_faults=("fault_id", "count"),
        severe_bearing_faults=(
            "severity",
            lambda x: (x >= 3).sum()
        )
    )
    .reset_index()
)

monthly["severe_pct"] = (
    monthly["severe_bearing_faults"]
    / monthly["bearing_faults"]
    * 100
)

print("\n=== Monthly Bearing Fault Distribution ===")
print(monthly.to_string(index=False))

# Monthly bearing fault rate per robot-day

telemetry["month"] = (
    telemetry["timestamp"]
    .dt.strftime("%Y-%m")
)

robot_days = (
    telemetry.assign(
        day=telemetry["timestamp"].dt.floor("D")
    )
    .groupby("month")[["unit_id", "day"]]
    .apply(lambda x: x.drop_duplicates().shape[0])
    .rename("robot_days")
    .reset_index()
)

monthly_rate = monthly.merge(
    robot_days,
    on="month",
    how="left"
)

monthly_rate["faults_per_1000_robot_days"] = (
    monthly_rate["bearing_faults"]
    / monthly_rate["robot_days"]
    * 1000
)

print("\n=== Monthly Bearing Fault Rate ===")
print(
    monthly_rate[
        [
            "month",
            "bearing_faults",
            "robot_days",
            "faults_per_1000_robot_days"
        ]
    ].round(3).to_string(index=False)
)