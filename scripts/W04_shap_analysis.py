#!/usr/bin/env python3
"""Week 4: exploratory SHAP interpretation of the reduced-complexity GB model.

Run from the repository root: python scripts/W04_shap_analysis.py
Only train/validation rows are accessed. The test split is never scored.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/causal_preview/processed/W03_robot_day_features.csv'
OUT = ROOT / 'reports/W04'
TARGET = 'severe_fault_next_7d'
EXCLUDE = {TARGET, 'split', 'unit_id', 'date', 'commission_date', 'during_maintenance'}
SEED = 42
SAMPLE_SIZE = 300


def main():
    try:
        import shap
    except ImportError as exc:
        raise SystemExit('Install SHAP in the active environment: python -m pip install shap') from exc

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    train = df.loc[df['split'].eq('train')].copy()
    val = df[df['split'] == 'validation'].copy()
    assert len(train) > 0 and len(val) > 0
    # Match the earlier W04 calibration / permutation experiment: a chronological
    # earlier training portion, not the full train split.
    unique_dates = np.sort(train['date'].unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]

# Match the 7-day gap used in W04_feature_importance.py
    fit = train.loc[
        train['date'] < cutoff - pd.Timedelta(days=7)
    ].copy()
    print(f'Chronological fit rows: {len(fit)}; validation rows: {len(val)}')
    print('IMPORTANT: verify this fit row count against W04_feature_importance.py (expected 11746).')

    features = [c for c in df.columns if c not in EXCLUDE]
    categorical = [c for c in features if c in ('site_id', 'terrain_type') or not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in features if c not in categorical]
    transformers = [('num', Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ]), numeric)]
    if categorical:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_fit = preprocessor.fit_transform(fit[features])
    X_val = preprocessor.transform(val[features])
    if hasattr(X_fit, 'toarray'):
        X_fit = X_fit.toarray()
    if hasattr(X_val, 'toarray'):
        X_val = X_val.toarray()
    names = list(preprocessor.get_feature_names_out())
    print(f'Preprocessed features: {len(names)}')

    model = HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=50, l2_regularization=5.0, random_state=42,
    )
    print('Training GB_leaf7_iter80 ...', flush=True)
    model.fit(X_fit, fit[TARGET].astype(int))
    ap = average_precision_score(val[TARGET].astype(int), model.predict_proba(X_val)[:, 1])
    print(f'Validation AP: {ap:.5f} (compare with earlier 0.05307)')

    rng = np.random.default_rng(SEED)
    n = min(SAMPLE_SIZE, len(val))
    idx = np.sort(rng.choice(len(val), size=n, replace=False))
    X_sample = X_val[idx]
    print(f'Computing SHAP on {n} validation rows ...', flush=True)
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)
    if isinstance(raw, list):
        values = np.asarray(raw[-1])
    else:
        values = np.asarray(raw)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != X_sample.shape:
        raise ValueError(f'Unexpected SHAP shape {values.shape}; expected {X_sample.shape}')

    importance = pd.DataFrame({
        'transformed_feature': names,
        'mean_abs_shap': np.abs(values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    importance.to_csv(OUT / 'W04_SHAP_Importance.csv', index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, X_sample, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT / 'W04_SHAP_Summary.png', dpi=180, bbox_inches='tight')
    plt.close('all')

    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, X_sample, feature_names=names, plot_type='bar', show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT / 'W04_SHAP_Bar.png', dpi=180, bbox_inches='tight')
    plt.close('all')

    print('\nTop 15 transformed features by mean |SHAP|:')
    print(importance.head(15).to_string(index=False))
    print('\nSaved: W04_SHAP_Importance.csv, W04_SHAP_Summary.png, W04_SHAP_Bar.png')
    print('SHAP values are on the model raw-score (log-odds) scale, not probability or causal effects.')
    print('Correlated features and transformed missing indicators/one-hot columns require caution.')


if __name__ == '__main__':
    main()
