#!/usr/bin/env python3
"""Week 4: leakage-aware model selection using train and validation only."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT = Path('data/causal_preview/processed/W03_robot_day_features.csv')
OUTPUT = Path('reports/W04')
TARGET = 'severe_fault_next_7d'
EXCLUDE = {'unit_id', 'date', 'during_maintenance', TARGET, 'split', 'commission_date'}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing {INPUT}; run the Week 3 feature pipeline first.')
    df = pd.read_csv(INPUT)
    required = {'split', TARGET, 'date'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing required columns: {required - set(df.columns)}')
    if df.duplicated(['unit_id', 'date']).any():
        raise ValueError('Duplicate robot-day records found.')
    if not df['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Unexpected split labels.')
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError('Target must be binary and nonmissing.')
    train = df[df['split'] == 'train'].copy()
    val = df[df['split'] == 'validation'].copy()
    test = df[df['split'] == 'test'].copy()
    for name, part in [('train', train), ('validation', val), ('test', test)]:
        print(f'{name:>10}: {len(part):>6,} rows, {int(part[TARGET].sum()):>4} positives')
    dates = [pd.to_datetime(x['date']) for x in (train, val, test)]
    if not (dates[0].max() < dates[1].min() and dates[1].max() < dates[2].min()):
        raise ValueError('Chronological split boundaries overlap.')
    print('Test set reserved: no model fitting, tuning or scoring on test in this script.')

    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    # Identify categorical features
    categorical = [
        c for c in ['site_id', 'terrain_type']
        if c in feature_cols
    ]

    # Detect any additional categorical features
    categorical += [
        c for c in feature_cols
        if c not in categorical
        and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # Identify numeric features
    numeric = [
        c for c in feature_cols
        if c not in categorical
    ]

    if not numeric:
        raise ValueError('No numeric features found.')

    print(
        f'Features: {len(feature_cols)} '
        f'({len(numeric)} numeric, {len(categorical)} categorical)'
    )
    print('Categorical columns:', categorical)

    X_train, y_train = train[feature_cols], train[TARGET].astype(int)
    X_val, y_val = val[feature_cols], val[TARGET].astype(int)

    # All preprocessing is fitted on training data only.
    transformers = [('num', Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler())
    ]), numeric)]
    if categorical:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    models = {
        # Logistic Regression: compare regularization and class weighting
        'LR_C0.01_balanced': LogisticRegression(
            C=0.01, max_iter=1500,
            class_weight='balanced', random_state=42
        ),
        'LR_C0.1_balanced': LogisticRegression(
            C=0.1, max_iter=1500,
            class_weight='balanced', random_state=42
        ),
        'LR_C1_unweighted': LogisticRegression(
            C=1.0, max_iter=1500,
            class_weight=None, random_state=42
        ),

        # Random Forest: reduce tree complexity
        'RF_depth4_leaf50': RandomForestClassifier(
            n_estimators=200, max_depth=4,
            min_samples_leaf=50,
            class_weight='balanced_subsample',
            random_state=42, n_jobs=-1
        ),
        'RF_depth6_leaf100': RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=100,
            class_weight='balanced_subsample',
            random_state=42, n_jobs=-1
        ),

        # Gradient Boosting: reduce complexity
        'GB_leaf7_iter80': HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.05,
            max_leaf_nodes=7, min_samples_leaf=50,
            l2_regularization=5.0, random_state=42
        ),
        'GB_leaf7_iter120': HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.03,
            max_leaf_nodes=7, min_samples_leaf=100,
            l2_regularization=10.0, random_state=42
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    predictions = pd.DataFrame({'unit_id': val['unit_id'].to_numpy(), 'date': val['date'].to_numpy(), 'label': y_val.to_numpy()})

    # Baseline is deliberately simple and explicitly labeled a proxy, not the original Week-1 rule.
    # Choose threshold from training distribution only, not validation.
    baseline_feature = 'wheel_wear_pct_7d_last'
    if baseline_feature in X_train:
        threshold = float(X_train[baseline_feature].quantile(0.90))
        score = pd.to_numeric(X_val[baseline_feature], errors='coerce').fillna(-np.inf).to_numpy()
        pred = (score >= threshold).astype(int)
        ap = float(average_precision_score(y_val, pred))
        tp = int(((pred == 1) & (y_val.to_numpy() == 1)).sum())
        fp = int(((pred == 1) & (y_val.to_numpy() == 0)).sum())
        fn = int(((pred == 0) & (y_val.to_numpy() == 1)).sum())
        records.append({'model': 'Wear Threshold Baseline (proxy)', 'validation_pr_auc': ap, 'precision_at_0_5': tp / max(tp + fp, 1), 'recall_at_0_5': tp / max(tp + fn, 1), 'threshold_note': f'{baseline_feature} >= train p90 ({threshold:.3f})'})
        predictions['wear_baseline'] = pred
        print(f'Wear Threshold Baseline (proxy): validation AP={ap:.4f}')

    for name, estimator in models.items():
        print(f'Training {name} ...', flush=True)
        pipeline = Pipeline([('preprocess', preprocessor), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        train_scores = pipeline.predict_proba(X_train)[:, 1]
        train_ap = average_precision_score(y_train, train_scores)

        print(f'  training AP={train_ap:.4f}')
        scores = pipeline.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, scores))
        roc = float(roc_auc_score(y_val, scores))
        pred = (scores >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_val.to_numpy() == 1)).sum())
        fp = int(((pred == 1) & (y_val.to_numpy() == 0)).sum())
        fn = int(((pred == 0) & (y_val.to_numpy() == 1)).sum())
        records.append({'model': name, 'validation_pr_auc': ap, 'validation_roc_auc': roc, 'precision_at_0_5': tp / max(tp + fp, 1), 'recall_at_0_5': tp / max(tp + fn, 1), 'threshold_note': '0.5 (diagnostic only; not optimized)'})
        predictions[name.lower().replace(' ', '_')] = scores
        print(f'  validation AP={ap:.4f} | ROC-AUC={roc:.4f} | precision@0.5={records[-1]["precision_at_0_5"]:.4f} | recall@0.5={records[-1]["recall_at_0_5"]:.4f}')

    results = pd.DataFrame(records).sort_values('validation_pr_auc', ascending=False)
    results.to_csv(OUTPUT / 'W04_validation_model_comparison.csv', index=False)
    predictions.to_csv(OUTPUT / 'W04_validation_predictions.csv', index=False)
    print('\nValidation comparison:')
    print(results.to_string(index=False))
    print(f'\nSaved: {OUTPUT / "W04_validation_model_comparison.csv"}')
    print(f'Saved: {OUTPUT / "W04_validation_predictions.csv"}')
    print('Next: calibration and cost-sensitive threshold analysis on held-out validation data.')

if __name__ == '__main__':
    main()
