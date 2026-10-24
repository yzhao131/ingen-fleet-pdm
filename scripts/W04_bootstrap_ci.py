#!/usr/bin/env python3
"""Week 4: cluster-bootstrap confidence intervals on saved validation predictions.

Run from repository root: python scripts/W04_bootstrap_ci.py
Does not load test rows or refit models.

If prediction CSV lacks unit_id/date, this script deliberately refuses to infer
row alignment. Add these keys when exporting predictions, or use an explicitly
verified row-ID mapping.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/causal_preview/processed/W03_robot_day_features.csv'
OUT = ROOT / 'reports/W04'
PRED_FILES = [OUT / 'W04_calibrated_validation_predictions.csv',
              OUT / 'W04_validation_predictions.csv']
TARGET = 'severe_fault_next_7d'
SEED = 42
N_BOOT = 1000


def normalize_date(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='raise')
    return df


# Explicit model columns from the Week 4 prediction exports. No heuristic naming.
SCORE_COLUMNS = {
    'W04_calibrated_validation_predictions.csv': [
        'rf_depth4_leaf50_raw',
        'rf_depth4_leaf50_platt',
        'rf_depth4_leaf50_isotonic',
        'gb_leaf7_iter80_raw',
        'gb_leaf7_iter80_isotonic',
    ],
    'W04_validation_predictions.csv': [
        'wear_baseline',
        'rf_depth4_leaf50',
        'gb_leaf7_iter80',
    ],
}


def identify_score_columns(df, filename):
    cols = SCORE_COLUMNS.get(filename, [])
    valid = []
    for c in cols:
        if c not in df.columns:
            print(f'SKIPPED column {c}: not present')
            continue
        vals = pd.to_numeric(df[c], errors='coerce')
        if vals.isna().any() or not vals.between(0, 1).all():
            print(f'SKIPPED column {c}: missing or outside [0,1]')
            continue
        valid.append(c)
    return valid


def get_predictions(path, validation):
    pred = pd.read_csv(path)
    print(f'\nInspecting {path.name}: {len(pred)} rows')
    print('Columns:', list(pred.columns))
    # Explicitly require stable keys. Matching on position is unsafe because
    # training/prediction scripts may sort or reset the dataframe.
    if not {'unit_id', 'date'}.issubset(pred.columns):
        print('SKIPPED: prediction file has no unit_id + date keys; cannot safely align robot-day rows.')
        return None
    pred = normalize_date(pred)
    if 'split' in pred.columns:
        pred = pred.loc[pred['split'].eq('validation')].copy()
    keys = ['unit_id', 'date']
    if pred.duplicated(keys).any():
        print('SKIPPED: duplicate unit_id/date in prediction file (possibly long-format models).')
        return None
    cols = identify_score_columns(pred, path.name)
    if not cols:
        print('SKIPPED: none of the explicitly listed model columns are valid.')
        return None
    extra = ['label'] if 'label' in pred.columns else []
    merged = validation[keys + [TARGET]].merge(
        pred[keys + extra + cols], on=keys, how='left', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        print('SKIPPED: some validation unit_id/date rows have no predictions.')
        return None
    if 'label' in merged.columns:
        if merged['label'].isna().any() or not np.array_equal(
            merged['label'].astype(int).to_numpy(), merged[TARGET].astype(int).to_numpy()
        ):
            print('SKIPPED: exported labels disagree with validation labels.')
            return None
        merged = merged.drop(columns='label')
    if merged[cols].isna().any().any():
        print('SKIPPED: missing probability values after joining.')
        return None
    return merged.drop(columns='_merge'), cols


def bootstrap_metric(y, p, groups, rng, n_boot):
    unique, group_codes = np.unique(groups, return_inverse=True)
    row_indices = [np.flatnonzero(group_codes == j) for j in range(len(unique))]
    ap = np.empty(n_boot)
    auc = np.empty(n_boot)
    prevalence = np.empty(n_boot)
    valid = 0
    for _ in range(n_boot):
        selected = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([row_indices[j] for j in selected])
        yy, pp = y[idx], p[idx]
        if np.unique(yy).size < 2:
            continue
        ap[valid] = average_precision_score(yy, pp)
        auc[valid] = roc_auc_score(yy, pp)
        prevalence[valid] = yy.mean()
        valid += 1
    if valid < n_boot * .9:
        raise ValueError(f'Only {valid}/{n_boot} valid bootstrap replicates; too few positive clusters?')
    return ap[:valid], auc[:valid], prevalence[:valid]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    val = df.loc[df['split'].eq('validation'), ['unit_id', 'date', TARGET]].copy()
    val = normalize_date(val)
    if val.duplicated(['unit_id', 'date']).any():
        raise ValueError('Validation data has duplicate unit_id/date keys')
    print(f'Validation: {len(val)} robot-days, {val.unit_id.nunique()} robots, '
          f'{int(val[TARGET].sum())} positives; prevalence={val[TARGET].mean():.5f}')
    outputs = []
    all_draws = []
    seen = set()
    for path in PRED_FILES:
        if not path.exists():
            print(f'Not found: {path}')
            continue
        result = get_predictions(path, val)
        if result is None:
            continue
        merged, cols = result
        y = merged[TARGET].astype(int).to_numpy()
        groups = merged['unit_id'].astype(str).to_numpy()
        for c in cols:
            label = f'{path.stem}::{c}'
            if label in seen:
                continue
            seen.add(label)
            p = merged[c].to_numpy(dtype=float)
            ap0 = average_precision_score(y, p)
            auc0 = roc_auc_score(y, p)
            rng = np.random.default_rng(SEED)
            ap, auc, prev = bootstrap_metric(y, p, groups, rng, N_BOOT)
            diff = ap - prev  # prevalence changes with sampled clusters
            outputs.append({
                'model': label, 'n_robot_days': len(y), 'n_robots': len(np.unique(groups)),
                'n_positive': int(y.sum()), 'validation_prevalence': y.mean(),
                'ap': ap0, 'ap_ci_lower': np.quantile(ap, .025),
                'ap_ci_upper': np.quantile(ap, .975),
                'roc_auc': auc0, 'roc_auc_ci_lower': np.quantile(auc, .025),
                'roc_auc_ci_upper': np.quantile(auc, .975),
                'ap_minus_prevalence': ap0-y.mean(),
                'ap_minus_prevalence_ci_lower': np.quantile(diff, .025),
                'ap_minus_prevalence_ci_upper': np.quantile(diff, .975),
                'n_boot_valid': len(ap),
            })
            all_draws.append((label, ap))
            print(f'{label}: AP={ap0:.5f} [{np.quantile(ap,.025):.5f}, {np.quantile(ap,.975):.5f}], '
                  f'ROC-AUC={auc0:.5f} [{np.quantile(auc,.025):.5f}, {np.quantile(auc,.975):.5f}]')
    if not outputs:
        raise SystemExit('\nNo safely alignable prediction files found. Please send the printed columns '
                         'and the first few rows (head) of the prediction CSV. '
                         'We will adapt the script without guessing row order.')
    result = pd.DataFrame(outputs)
    result.to_csv(OUT / 'W04_bootstrap_confidence_intervals.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(all_draws)*.65 + 1)))
    for i, (label, ap) in enumerate(all_draws):
        ax.boxplot(ap, positions=[i], vert=False, widths=.55, showfliers=False)
    ax.set_yticks(range(len(all_draws)), [x[0] for x in all_draws], fontsize=8)
    ax.axvline(val[TARGET].mean(), linestyle='--', color='gray', label='Validation prevalence')
    ax.set_xlabel('Cluster bootstrap average precision (AP)')
    ax.set_title('Week 4 validation PR-AUC uncertainty (resampled robots)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(OUT / 'W04_bootstrap_distributions.png', dpi=180)
    plt.close(fig)
    print('\nSaved:', OUT / 'W04_bootstrap_confidence_intervals.csv')
    print('Saved:', OUT / 'W04_bootstrap_distributions.png')
    print('Caution: exploratory validation intervals; model selection on validation is not accounted for.')


if __name__ == '__main__':
    main()
