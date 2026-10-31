
from pathlib import Path
import pandas as pd

from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment
)
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "W05"

weekly = pd.read_csv(
    REPORT_DIR / "W05_technician_staffing_plan.csv"
)

battery = pd.read_csv(
    REPORT_DIR / "W05_monthly_battery_planning.csv"
)

procurement_units = pd.read_csv(
    REPORT_DIR / "W05_battery_procurement_units.csv"
)

procurement_monthly = pd.read_csv(
    REPORT_DIR / "W05_battery_procurement_monthly.csv"
)

wb = Workbook()
wb.remove(wb.active)

HEADER_FILL = PatternFill(
    "solid", fgColor="17365D"
)

HEADER_FONT = Font(
    color="FFFFFF", bold=True
)


def add_dataframe_sheet(name, df):
    ws = wb.create_sheet(name)

    ws.append(list(df.columns))

    for row in df.itertuples(index=False, name=None):
        ws.append(list(row))

    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(
            horizontal="center"
        )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for col in ws.columns:
        letter = get_column_letter(
            col[0].column
        )

        max_length = max(
            len(str(cell.value))
            if cell.value is not None else 0
            for cell in col
        )

        ws.column_dimensions[letter].width = min(
            max_length + 3, 35
        )

    return ws


# -------------------------------------
# Weekly Staffing
# -------------------------------------

add_dataframe_sheet(
    "Weekly Staffing", weekly
)

# -------------------------------------
# Battery Replacements
# -------------------------------------

add_dataframe_sheet(
    "Battery Replacements", battery
)

add_dataframe_sheet(
    "Procurement Monthly", procurement_monthly
)

add_dataframe_sheet(
    "Procurement Units", procurement_units
)

# -------------------------------------
# Planning Summary
# -------------------------------------

summary = pd.DataFrame([
    {
        "metric": "Forecast horizon (weeks)",
        "value": len(weekly)
    },
    {
        "metric": "Total expected service events",
        "value": weekly[
            "forecast_service_events"
        ].sum()
    },
    {
        "metric": "Total expected technician-hours",
        "value": weekly[
            "expected_technician_hours"
        ].sum()
    },
    {
        "metric": "Maximum upper-scenario technicians",
        "value": weekly[
            "upper_technicians"
        ].max()
    },
    {
        "metric": "Weeks requiring 2 technicians",
        "value": int(
            (weekly["upper_technicians"] >= 2).sum()
        )
    },
    {
        "metric": "80% threshold replacements",
        "value": int(
            battery.loc[
                battery["threshold_pct"] == 80,
                "expected_replacements"
            ].sum()
        )
    },
    {
        "metric": "90% threshold replacements",
        "value": int(
            battery.loc[
                battery["threshold_pct"] == 90,
                "expected_replacements"
            ].sum()
        )
    }
])

add_dataframe_sheet(
    "Planning Summary", summary
)

# -------------------------------------
# Assumptions
# -------------------------------------

assumptions = pd.DataFrame([
    {
        "assumption": "Hours per service event",
        "value": 8,
        "notes": "Illustrative planning assumption"
    },
    {
        "assumption": "Productive hours per technician/week",
        "value": 30,
        "notes": "Illustrative planning assumption"
    },
    {
        "assumption": "Service demand model",
        "value": "Last Value",
        "notes": "Lowest backtested MAE"
    },
    {
        "assumption": "Forecast interval",
        "value": "Preliminary horizon-specific",
        "notes": "70.8% exploratory coverage; 24 evaluation forecasts"
    },
    {
        "assumption": "Primary battery threshold",
        "value": "80%",
        "notes": "Capacity threshold scenario"
    },
    {
        "assumption": "Sensitivity battery threshold",
        "value": "90%",
        "notes": "Earlier replacement scenario"
    },
    {
        "assumption": "Battery replacement counts",
        "value": "First crossing only",
        "notes": "Does not model repeated future replacements"
    },
    {
        "assumption": "Battery planning horizon",
        "value": "2026-07 to 2031-12",
        "notes": "Forecasts beyond observed simulation period"
    },
    {
        "assumption": "Supplier lead time",
        "value": "30 days",
        "notes": "Illustrative assumption; not supplier-verified"
    },
    {
        "assumption": "Procurement safety buffer",
        "value": "14 days",
        "notes": "Additional planning buffer"
    }
])

add_dataframe_sheet(
    "Assumptions", assumptions
)

# -------------------------------------
# Save workbook
# -------------------------------------

output = (
    REPORT_DIR / "W05_Planning_Outputs.xlsx"
)

wb.save(output)

print("=== Week 5 Planning Excel ===")
print("Saved:", output)
print("Worksheets:", wb.sheetnames)
print("Weekly rows:", len(weekly))
print("Battery rows:", len(battery))
