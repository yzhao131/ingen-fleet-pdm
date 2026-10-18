"""Week 3: leakage-aware daily feature engineering for causal Aido fleet data.

Run from repo root: python scripts/W03_build_features.py
Outputs: data/causal_preview/processed/W03_robot_day_features.csv
         reports/W03/W03_feature_dictionary.csv

All time-window features use events available strictly before midnight UTC on
prediction date. Existing W02 labels and chronological splits are retained.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / 'data/causal_preview/clean'
PROCESSED = ROOT / 'data/causal_preview/processed'
REPORT = ROOT / 'reports/W03'
REPORT.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(PROCESSED / 'W02_robot_day_modeling_frame.csv')
frame['date'] = pd.to_datetime(frame['date'], utc=True)
frame = frame.sort_values(['unit_id', 'date']).reset_index(drop=True)
assert not frame.duplicated(['unit_id','date']).any()

robots = pd.read_csv(CLEAN / 'robots.csv')
robots['commission_date'] = pd.to_datetime(robots['commission_date'], utc=True)
frame = frame.merge(robots[['unit_id','site_id','terrain_type','commission_date']], on='unit_id', how='left', validate='many_to_one')
frame['unit_age_days'] = (frame['date'] - frame['commission_date']).dt.total_seconds() / 86400
frame['day_of_week'] = frame['date'].dt.dayofweek
frame['month'] = frame['date'].dt.month

telemetry = pd.read_csv(CLEAN / 'telemetry.csv')
telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'], utc=True)
telemetry = telemetry.sort_values(['unit_id','timestamp'])
# Optional measurements may be present; only compute features on real columns.
measurements = [c for c in ['battery_capacity_pct','charge_cycles','motor_current_a','wheel_wear_pct','temperature_c'] if c in telemetry]
missions = pd.read_csv(CLEAN / 'missions.csv')
missions['end_time'] = pd.to_datetime(missions['end_time'], utc=True)
missions = missions[missions['mission_status'].eq('completed')].copy()
faults = pd.read_csv(CLEAN / 'faults.csv')
faults['logged_at'] = pd.to_datetime(faults['logged_at'], utc=True)
maintenance = pd.read_csv(CLEAN / 'maintenance.csv')
maintenance['end_time'] = pd.to_datetime(maintenance['end_time'], utc=True)

records = []
for unit_id, days in frame.groupby('unit_id', sort=False):
    t = telemetry[telemetry.unit_id.eq(unit_id)]
    m = missions[missions.unit_id.eq(unit_id)]
    f = faults[faults.unit_id.eq(unit_id)]
    maint = maintenance[maintenance.unit_id.eq(unit_id)]
    for row in days.itertuples(index=False):
        T = row.date
        out = {'unit_id': unit_id, 'date': T}
        for window in (7, 14, 30):
            start = T - pd.Timedelta(days=window)
            tw = t[(t.timestamp >= start) & (t.timestamp < T)]
            mw = m[(m.end_time >= start) & (m.end_time < T)]
            fw = f[(f.logged_at >= start) & (f.logged_at < T)]
            out[f'telemetry_count_{window}d'] = len(tw)
            out[f'telemetry_coverage_{window}d'] = min(len(tw) / (window * 4), 1.0)
            out[f'mission_count_{window}d'] = len(mw)
            out[f'operating_hours_{window}d'] = mw.operating_hours.sum()
            out[f'distance_km_{window}d'] = mw.distance_km.sum()
            out[f'fault_count_{window}d'] = len(fw)
            for col in (measurements if window in (7, 30) else []):
                vals = tw[col].dropna()
                prefix = f'{col}_{window}d'
                out[f'{prefix}_mean'] = vals.mean() if len(vals) else np.nan
                out[f'{prefix}_std'] = vals.std() if len(vals) > 1 else np.nan
                out[f'{prefix}_last'] = vals.iloc[-1] if len(vals) else np.nan
                # Time-based slope in units per day (actual timestamps, not row order).
                if len(vals) > 1:
                    xs = (tw.loc[vals.index, 'timestamp'] - T).dt.total_seconds().to_numpy() / 86400
                    out[f'{prefix}_slope'] = np.polyfit(xs, vals.to_numpy(dtype=float), 1)[0] if np.ptp(xs) > 0 else np.nan
                else:
                    out[f'{prefix}_slope'] = np.nan
        prior_faults = f[f.logged_at < T]
        prior_maint = maint[maint.end_time < T]
        out['fault_count_lifetime'] = len(prior_faults)
        out['maintenance_count_lifetime'] = len(prior_maint)
        out['days_since_last_fault'] = ((T - prior_faults.logged_at.max()).total_seconds() / 86400) if len(prior_faults) else np.nan
        out['days_since_last_maintenance'] = ((T - prior_maint.end_time.max()).total_seconds() / 86400) if len(prior_maint) else np.nan
        for col in measurements:
            a, b = out[f'{col}_7d_mean'], out[f'{col}_30d_mean']
            out[f'{col}_7d_vs_30d_mean'] = a - b if pd.notna(a) and pd.notna(b) else np.nan
        records.append(out)

features = pd.DataFrame.from_records(records)
result = frame.merge(features, on=['unit_id','date'], how='left', validate='one_to_one')
assert len(result) == len(frame)
assert not result.duplicated(['unit_id','date']).any()
feature_cols = [c for c in result.columns if c not in frame.columns]
assert len(feature_cols) >= 40, f'Only {len(feature_cols)} features created'
PROCESSED.mkdir(parents=True, exist_ok=True)
output = PROCESSED / 'W03_robot_day_features.csv'
result.to_csv(output, index=False)
dictionary = pd.DataFrame({'feature':feature_cols, 'definition':['Historical/context feature computed using records available before prediction midnight UTC' for _ in feature_cols]})
dictionary.to_csv(REPORT / 'W03_feature_dictionary.csv', index=False)
print(f'Rows: {len(result):,}; engineered features: {len(feature_cols)}')
print(f'Saved: {output}')
print(f'Dictionary: {REPORT / "W03_feature_dictionary.csv"}')
