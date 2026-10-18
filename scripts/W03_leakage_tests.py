"""Week 3 leakage audit for W03_build_features.py.
Run: python scripts/W03_leakage_tests.py
Checks sampled features against independently filtered historical source records.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / 'data/causal_preview/clean'
PROCESSED = ROOT / 'data/causal_preview/processed'
REPORT = ROOT / 'reports/W03'
REPORT.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(PROCESSED / 'W03_robot_day_features.csv')
features['date'] = pd.to_datetime(features['date'], utc=True)
assert not features.duplicated(['unit_id', 'date']).any(), 'Duplicate robot-days'
assert not features['during_maintenance'].fillna(False).astype(bool).any(), 'Maintenance rows included'
assert features['split'].isin(['train', 'validation', 'test']).all(), 'Unexpected split'

telemetry = pd.read_csv(CLEAN / 'telemetry.csv', parse_dates=['timestamp'])
missions = pd.read_csv(CLEAN / 'missions.csv', parse_dates=['end_time'])
faults = pd.read_csv(CLEAN / 'faults.csv', parse_dates=['logged_at'])
maintenance = pd.read_csv(CLEAN / 'maintenance.csv', parse_dates=['end_time'])
for df, col in [(telemetry, 'timestamp'), (missions, 'end_time'), (faults, 'logged_at'), (maintenance, 'end_time')]:
    df[col] = pd.to_datetime(df[col], utc=True)

# Sample evenly across train, validation, and test, reproducibly.
sample = (features.groupby('split', group_keys=False)
          .apply(lambda x: x.sample(n=min(15, len(x)), random_state=42)))
checks = []

def check(row, feature, expected):
    actual = row[feature]
    passed = bool(np.isclose(actual, expected, rtol=1e-7, atol=1e-7, equal_nan=True))
    checks.append({'unit_id': row['unit_id'], 'date': row['date'].isoformat(),
                   'feature': feature, 'expected': expected, 'actual': actual, 'passed': passed})

for _, row in sample.iterrows():
    unit, cutoff = row['unit_id'], row['date']
    t = telemetry[(telemetry.unit_id == unit) & (telemetry.timestamp < cutoff)]
    m = missions[(missions.unit_id == unit) & (missions.end_time < cutoff) &
                 (missions.mission_status == 'completed')]
    f = faults[(faults.unit_id == unit) & (faults.logged_at < cutoff)]
    maint = maintenance[(maintenance.unit_id == unit) & (maintenance.end_time < cutoff)]
    for window in (7, 14, 30):
        start = cutoff - pd.Timedelta(days=window)
        tw = t[t.timestamp >= start].sort_values('timestamp')
        mw = m[m.end_time >= start]
        fw = f[f.logged_at >= start]
        check(row, f'telemetry_count_{window}d', len(tw))
        check(row, f'mission_count_{window}d', len(mw))
        check(row, f'operating_hours_{window}d', mw.operating_hours.sum())
        check(row, f'distance_km_{window}d', mw.distance_km.sum())
        check(row, f'fault_count_{window}d', len(fw))
        check(row, f'telemetry_coverage_{window}d', min(len(tw)/(window*4), 1.0))
        if window in (7, 30):
            for col in ['battery_capacity_pct','charge_cycles','motor_current_a','wheel_wear_pct','temperature_c']:
                if col not in t.columns:
                    continue
                vals = tw[col].dropna()
                prefix = f'{col}_{window}d'
                check(row, f'{prefix}_mean', vals.mean() if len(vals) else np.nan)
                check(row, f'{prefix}_std', vals.std() if len(vals)>1 else np.nan)
                check(row, f'{prefix}_last', vals.iloc[-1] if len(vals) else np.nan)
                if len(vals)>1:
                    xs = (tw.loc[vals.index, 'timestamp'] - cutoff).dt.total_seconds().to_numpy()/86400
                    slope = np.polyfit(xs, vals.to_numpy(dtype=float), 1)[0] if np.ptp(xs)>0 else np.nan
                else:
                    slope = np.nan
                check(row, f'{prefix}_slope', slope)
    check(row, 'fault_count_lifetime', len(f))
    check(row, 'maintenance_count_lifetime', len(maint))
    check(row, 'days_since_last_fault', (cutoff-f.logged_at.max()).total_seconds()/86400 if len(f) else np.nan)
    check(row, 'days_since_last_maintenance', (cutoff-maint.end_time.max()).total_seconds()/86400 if len(maint) else np.nan)
    for col in ['battery_capacity_pct','charge_cycles','motor_current_a','wheel_wear_pct','temperature_c']:
        name = f'{col}_7d_vs_30d_mean'
        if name in row:
            a, b = row[f'{col}_7d_mean'], row[f'{col}_30d_mean']
            check(row, name, a-b if pd.notna(a) and pd.notna(b) else np.nan)

results = pd.DataFrame(checks)
results.to_csv(REPORT / 'W03_leakage_audit.csv', index=False)
failures = results[~results.passed]
print(f'Sampled robot-days: {len(sample)}')
print(f'Historical recomputation checks: {len(results)}')
print(f'Passed: {len(results)-len(failures)}; failed: {len(failures)}')
print(f'Audit: {REPORT / "W03_leakage_audit.csv"}')
if len(failures):
    print('First failures:')
    print(failures.head(10).to_string(index=False))
    raise SystemExit(1)
print('PASS: sampled features match historical-only recomputation.')
print('NOTE: This checks implementation consistency, not all possible upstream timestamp/label leakage.')
