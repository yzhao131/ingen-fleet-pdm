import pandas as pd
import numpy as np

path = "reports/W06_corrected/W06_robot_clusters_k3_corrected.csv"
df = pd.read_csv(path)

telemetry = pd.read_csv(
    "data/causal_fixed/raw/telemetry.csv"
)
telemetry["timestamp"] = pd.to_datetime(
    telemetry["timestamp"], format="mixed", utc=True
)

df["cutoff_time"] = pd.to_datetime(
    df["cutoff_time"], format="mixed", utc=True
)

t = telemetry.merge(
    df[["unit_id", "cutoff_time"]],
    on="unit_id",
    how="inner"
)

t = t[t["timestamp"] < t["cutoff_time"]].copy()
t = t.sort_values(["unit_id", "timestamp"])

cycles = t.groupby("unit_id").agg(
    first_cycles=("charge_cycles", "first"),
    last_cycles=("charge_cycles", "last"),
    first_timestamp=("timestamp", "first"),
    last_timestamp=("timestamp", "last"),
)

cycles["cycle_delta"] = (
    cycles["last_cycles"] - cycles["first_cycles"]
)

cycles["observation_days"] = (
    cycles["last_timestamp"]
    - cycles["first_timestamp"]
).dt.total_seconds() / 86400

result = df.merge(
    cycles.reset_index(),
    on="unit_id",
    how="left"
)

result["cycle_rate_corrected"] = (
    result["cycle_delta"]
    / result["total_operating_hours"] * 100
)

print("\n=== CLUSTER WINDOW DIAGNOSTICS ===")
print(
    result.groupby("cluster")[
        [
            "observation_days",
            "total_operating_hours",
            "first_cycles",
            "last_cycles",
            "cycle_delta",
            "cycle_rate_corrected",
            "faults_per_1000_hours",
        ]
    ].agg(["mean", "median"]).round(3).to_string()
)

print("\n=== SHORT-WINDOW ROBOTS ===")
print(
    result.nsmallest(
        10, "observation_days"
    )[
        [
            "unit_id",
            "cluster",
            "observation_days",
            "total_operating_hours",
            "fault_count",
            "faults_per_1000_hours",
            "cycle_rate_corrected",
        ]
    ].round(3).to_string(index=False)
)

print("\n=== CYCLE COUNTER CHECK ===")
print(
    "Negative cycle deltas:",
    (result["cycle_delta"] < 0).sum()
)
print(
    "Zero cycle deltas:",
    (result["cycle_delta"] == 0).sum()
)
print(
    "Missing cycle rates:",
    result["cycle_rate_corrected"].isna().sum()
)
