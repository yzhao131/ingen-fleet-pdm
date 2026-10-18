"""Week 3 exploratory pre-failure signal analysis (TRAIN split only).
Run: python scripts/W03_signal_analysis.py
Requires pandas, numpy, scipy. Does not read sealed data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data/causal_preview/processed/W03_robot_day_features.csv'
OUT = ROOT / 'reports/W03'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
df = pd.read_csv(INPUT, parse_dates=['date'])
train = df[df['split'].eq('train')].copy().sort_values(['unit_id','date'])
train['date'] = pd.to_datetime(train['date'], utc=True)
label = 'severe_fault_next_7d'
# Match each positive robot-day to a negative day for the same robot, within
# 45 calendar days if possible. Use distinct controls and a 14-day separation
# to avoid overlapping prediction horizons. Train only prevents test peeking.
pairs = []
for unit, g in train.groupby('unit_id'):
    pos = g[g[label].eq(1)].sort_values('date')
    neg = g[g[label].eq(0)].sort_values('date')
    used = set()
    for p in pos.itertuples():
        eligible = neg[~neg.index.isin(used)].copy()
        if eligible.empty:
            continue
        delta = (eligible['date'] - p.date).abs().dt.total_seconds() / 86400
        eligible = eligible[delta >= 14]
        if eligible.empty:
            continue
        delta = (eligible['date'] - p.date).abs().dt.total_seconds() / 86400
        near = eligible[delta <= 45]
        if not near.empty:
            eligible = near
            delta = (eligible['date'] - p.date).abs().dt.total_seconds() / 86400
        n_idx = delta.idxmin()
        used.add(n_idx)
        pairs.append((p.Index, n_idx))
if len(pairs) < 20:
    raise RuntimeError(f'Only {len(pairs)} matched pairs; inspect label prevalence and split.')
pidx, nidx = zip(*pairs)
p = train.loc[list(pidx)].reset_index(drop=True)
n = train.loc[list(nidx)].reset_index(drop=True)
exclude = {'unit_id','date','during_maintenance',label,'split','site_id','terrain_type','commission_date','day_of_week','month'}
features = [c for c in train.select_dtypes(include='number').columns if c not in exclude]
results = []
for col in features:
    a = pd.to_numeric(p[col], errors='coerce')
    b = pd.to_numeric(n[col], errors='coerce')
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        continue
    diffs = (a[ok] - b[ok]).to_numpy(dtype=float)
    sd = diffs.std(ddof=1)
    effect = diffs.mean() / sd if sd > 0 else np.nan
    pval = ttest_rel(a[ok], b[ok]).pvalue if sd > 0 else 1.0
    # Bootstrap 95% CI for mean paired difference.
    boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(500)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    results.append({'feature':col,'matched_pairs':int(ok.sum()),
                    'positive_mean':float(a[ok].mean()),'control_mean':float(b[ok].mean()),
                    'mean_paired_difference':float(diffs.mean()),
                    'paired_cohens_d':float(effect) if np.isfinite(effect) else np.nan,
                    'ci95_low':float(lo),'ci95_high':float(hi),'p_value':float(pval)})
res = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)
# Benjamini-Hochberg FDR correction, implemented without extra dependencies.
m = len(res)
if m == 0:
    raise RuntimeError('No features have sufficient paired non-missing observations.')
res['q_value_bh'] = np.minimum.accumulate((res['p_value'].to_numpy()[::-1] * m / np.arange(m,0,-1)))[::-1].clip(0,1)
res['significant_fdr_05'] = res['q_value_bh'] < 0.05
res.to_csv(OUT / 'W03_signal_analysis.csv',index=False)
# Descriptive feature stability: coverage and train-vs-validation distribution drift.
validation = df[df['split'].eq('validation')]
stability = []
for col in features:
    x = pd.to_numeric(train[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    y = pd.to_numeric(validation[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    pooled = x.std()
    stability.append({'feature':col,'train_coverage':x.notna().mean(),
        'validation_coverage':y.notna().mean(),'train_mean':x.mean(),
        'validation_mean':y.mean(),
        'standardized_mean_shift':(y.mean()-x.mean())/pooled if pd.notna(pooled) and pooled>0 else np.nan})
pd.DataFrame(stability).to_csv(OUT/'W03_feature_stability.csv',index=False)
print(f'Matched positive/control pairs (train only): {len(pairs)}')
print(f'Numeric features analyzed: {len(res)}')
print(f'FDR-significant signals (exploratory): {int(res.significant_fdr_05.sum())}')
print('\nTop 10 signals:')
print(res[['feature','mean_paired_difference','paired_cohens_d','q_value_bh']].head(10).to_string(index=False))
print(f'\nSaved: {OUT / "W03_signal_analysis.csv"}')
print(f'Saved: {OUT / "W03_feature_stability.csv"}')
print('NOTE: Overlapping positive prediction windows are correlated; p-values are exploratory, not confirmatory.')
