import pandas as pd

missions = pd.read_csv(
    "data/causal_fixed/raw/missions.csv",
    parse_dates=["start_time", "end_time"]
)

missions = missions.sort_values(
    ["unit_id", "start_time", "end_time"]
)

completed = missions[
    missions["mission_status"] == "completed"
].copy()

completed["previous_end"] = (
    completed.groupby("unit_id")["end_time"]
    .cummax()
    .groupby(completed["unit_id"])
    .shift()
)

overlaps = completed[
    completed["start_time"] < completed["previous_end"]
]

print("Completed missions:", len(completed))
print("Overlapping missions:", len(overlaps))
print("Robots affected:", overlaps["unit_id"].nunique())

assert len(overlaps) == 0, "Overlapping missions still exist!"
print("PASS: No overlapping completed missions")