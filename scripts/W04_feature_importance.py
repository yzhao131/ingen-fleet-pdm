#!/usr/bin/env python3
"""Week 4: exploratory permutation importance on temporal validation; no test access."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score

INPUT = Path('data/causal_preview/processed/W03_robot_day_features.csv')
OUTPUT = Path('reports/W04')
TARGET = 'severe_fault_next_7d'
EXCLUDE = {'unit_id', 'date', 'during_maintenance', TARGET, 'split', 'commission_date'}
SEED = 42

def main():
    df = pd.read_csv(INPUT)
    train = df.loc[df['split'].eq('train')].copy()
    val = df.loc[df['split'].eq('validation')].copy()
    if train.empty or val.empty:
        raise ValueError('Missing train/validation data')
    train['date'] = pd.to_datetime(train['date'], utc=True)
    val['date'] = pd.to_datetime(val['date'], utc=True)
    if train['date'].max() >= val['date'].min():
        raise ValueError('Chronological split overlap')
    unique_dates = np.sort(train['date'].unique())
    boundary = unique_dates[int(len(unique_dates) * .8)]
    fit = train.loc[train['date'] < boundary - pd.Timedelta(days=7)].copy()
    print(f'Model fit: {len(fit)} rows; validation: {len(val)} rows')
    features = [c for c in df.columns if c not in EXCLUDE]
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in features if c not in cats]
    transforms = [('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), nums)]
    if cats:
        transforms.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats))
    preprocess = ColumnTransformer(transforms)
    models = {
        'rf_depth4_leaf50': RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1),
        'gb_leaf7_iter80': HistGradientBoostingClassifier(max_iter=80, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=5., random_state=SEED),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Use complete validation for comparability with calibration results.
    X_val = val[features].reset_index(drop=True)
    y_val = val[TARGET].astype(int).to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, estimator in models.items():
        print(f'Training {name} ...', flush=True)
        pipe = Pipeline([('preprocess', preprocess), ('model', estimator)])
        pipe.fit(fit[features], fit[TARGET].astype(int))
        base_ap = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
        print(f'  validation AP={base_ap:.5f}; permuting {len(features)} features x 2', flush=True)
        for j, feature in enumerate(features, 1):
            drops = []
            for _ in range(2):
                X_perm = X_val.copy(deep=False).copy()
                X_perm[feature] = rng.permutation(X_val[feature].to_numpy())
                perm_ap = average_precision_score(y_val, pipe.predict_proba(X_perm)[:, 1])
                drops.append(base_ap - perm_ap)
            rows.append({'model': name, 'feature': feature, 'baseline_validation_ap': base_ap,
                         'mean_ap_drop': float(np.mean(drops)), 'std_ap_drop': float(np.std(drops))})
            if j % 15 == 0:
                print(f'  {j}/{len(features)} features done', flush=True)
        model_rows = sorted([r for r in rows if r['model'] == name], key=lambda r: r['mean_ap_drop'], reverse=True)[:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([r['feature'] for r in model_rows][::-1], [r['mean_ap_drop'] for r in model_rows][::-1])
        ax.set_xlabel('Decrease in validation AP when feature is shuffled')
        ax.set_title(f'{name}: top 15 permutation importances (exploratory)')
        fig.tight_layout()
        fig.savefig(OUTPUT / f'W04_{name}_permutation_top15.png', dpi=160)
        plt.close(fig)
        print(f'  top five: {[(r["feature"], round(r["mean_ap_drop"], 5)) for r in model_rows[:5]]}')
    result = pd.DataFrame(rows).sort_values(['model', 'mean_ap_drop'], ascending=[True, False])
    path = OUTPUT / 'W04_permutation_importance.csv'
    result.to_csv(path, index=False)
    print(f'\nSaved {path} and two PNG charts.')
    print('Importance is exploratory, uses validation already used for model selection, and may be unstable for correlated features.')
    print('A negative AP drop means shuffling improved AP; it is not evidence of a protective causal effect.')

if __name__ == '__main__':
    main()
