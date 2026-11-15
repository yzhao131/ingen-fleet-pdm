

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path("scripts/W07_decision_layer.py")
spec = importlib.util.spec_from_file_location(
    "decision_layer", SCRIPT
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

apply_suppression = module.apply_suppression

AS_OF = pd.Timestamp("2026-03-08", tz="UTC")

# Use real model scores, but synthetic operational events.
queue = pd.read_csv(
    "reports/W07/W07_service_queue_20260308_cost20.csv"
)

statuses = pd.DataFrame([
    {
        "unit_id": "R009",
        "status": "scheduled_service",
        "status_time": "2026-03-07T10:00:00Z",
        "scheduled_service_date": "2026-03-09T10:00:00Z",
        "inspection_result": "",
        "risk_at_inspection": None,
    },
    {
        "unit_id": "R049",
        "status": "inspection_completed",
        "status_time": "2026-03-06T10:00:00Z",
        "scheduled_service_date": None,
        "inspection_result": "no_finding",
        "risk_at_inspection": 0.045,
    },
    {
        "unit_id": "R043",
        "status": "inspection_completed",
        "status_time": "2026-03-06T10:00:00Z",
        "scheduled_service_date": None,
        "inspection_result": "no_finding",
        "risk_at_inspection": 0.020,
    },
    {
        "unit_id": "R008",
        "status": "scheduled_service",
        "status_time": "2026-03-09T10:00:00Z",
        "scheduled_service_date": "2026-03-10T10:00:00Z",
        "inspection_result": "",
        "risk_at_inspection": None,
    },
])

result = apply_suppression(queue, statuses, AS_OF)

by_unit = result.set_index("unit_id")

assert bool(by_unit.loc["R009", "suppressed"])
assert bool(by_unit.loc["R049", "suppressed"])

# Risk rose by more than 50%, so cooldown is overridden.
assert not bool(by_unit.loc["R043", "suppressed"])

# Future-recorded status must not affect historical scoring.
assert not bool(by_unit.loc["R008", "suppressed"])

print("=== SUPPRESSION TEST RESULTS ===")

print(
    result[
        result["unit_id"].isin(
            ["R009", "R049", "R043", "R008"]
        )
    ][[
        "unit_id",
        "calibrated_risk",
        "suppressed",
        "suppression_reason",
    ]].to_string(index=False)
)

print("\nALL SUPPRESSION ASSERTIONS PASSED")
