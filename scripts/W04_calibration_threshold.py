#!/usr/bin/env python3
"""Week 4 exploratory calibration and cost analysis. Test data never accessed."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve

INPUT = Path('data/causal_preview/processed/W03_robot_day_features.csv')
OUTPUT = Path('reports/W04')
TARGET = 'severe_fault_next_7d'
EXCLUDE = {'unit_id', 'date', 'during_maintenance', TARGET, 'split', 'commission_date'}
COST_RATIOS = [5, 10, 20]


def counts(y, scores, threshold):
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    train = df.loc[df['split'].eq('train')].copy()
    val = df.loc[df['split'].eq('validation')].copy()
    if train.empty or val.empty:
        raise ValueError('Train or validation data missing')
    train['date'] = pd.to_datetime(train['date'], utc=True)
    val['date'] = pd.to_datetime(val['date'], utc=True)
    if train['date'].max() >= val['date'].min():
        raise ValueError('Train and validation periods overlap')
    # Time-based internal calibration set; seven-day embargo limits label-window overlap.
    unique_dates = np.sort(train['date'].unique())
    boundary = unique_dates[int(len(unique_dates) * 0.8)]
    fit = train.loc[train['date'] < boundary - pd.Timedelta(days=7)].copy()
    cal = train.loc[train['date'] >= boundary].copy()
    print(f'Model-fit: {len(fit)} rows, {int(fit[TARGET].sum())} positives')
    print(f'Calibration: {len(cal)} rows, {int(cal[TARGET].sum())} positives')
    print(f'Validation: {len(val)} rows, {int(val[TARGET].sum())} positives')
    if min(fit[TARGET].nunique(), cal[TARGET].nunique(), val[TARGET].nunique()) < 2:
        raise ValueError('A split lacks one of the classes; change the internal split')
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    categorical = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in feature_cols if c not in categorical]
    transforms = [('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), numeric)]
    if categorical:
        transforms.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical))
    preprocess = ColumnTransformer(transforms)
    models = {
        'rf_depth4_leaf50': RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        'gb_leaf7_iter80': HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=5.0, random_state=42),
    }
    y_cal = cal[TARGET].astype(int).to_numpy()
    y_val = val[TARGET].astype(int).to_numpy()
    metrics = []
    threshold_rows = []
    predictions = val[['unit_id', 'date']].copy()
    predictions['label'] = y_val
    fig_cal, axes_cal = plt.subplots(1, 2, figsize=(12, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    for axis, (name, estimator) in zip(axes_cal, models.items()):
        print(f'Fitting {name} on earlier training dates ...', flush=True)
        pipe = Pipeline([('preprocess', preprocess), ('model', estimator)])
        pipe.fit(fit[feature_cols], fit[TARGET].astype(int))
        cal_raw = pipe.predict_proba(cal[feature_cols])[:, 1]
        val_raw = pipe.predict_proba(val[feature_cols])[:, 1]
        platt = LogisticRegression(max_iter=1000)
        platt.fit(cal_raw.reshape(-1, 1), y_cal)
        val_platt = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
        isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        isotonic.fit(cal_raw, y_cal)
        val_iso = isotonic.predict(val_raw)
        for method, scores in [('raw', val_raw), ('platt', val_platt), ('isotonic', val_iso)]:
            scores = np.clip(scores, 0, 1)
            predictions[f'{name}_{method}'] = scores
            metrics.append({'model': name, 'method': method, 'validation_ap': average_precision_score(y_val, scores), 'validation_roc_auc': roc_auc_score(y_val, scores), 'validation_brier': brier_score_loss(y_val, scores), 'validation_mean_prediction': np.mean(scores), 'validation_event_rate': np.mean(y_val)})
            # Quantile-binned reliability curve; empty bins ignored.
            bins = pd.qcut(pd.Series(scores).rank(method='first'), q=10, labels=False)
            curve = pd.DataFrame({'bin': bins, 'score': scores, 'label': y_val}).groupby('bin').agg(pred=('score', 'mean'), observed=('label', 'mean'))
            axis.plot(curve['pred'], curve['observed'], marker='o', label=method)
            if method in ('raw', 'platt'):
                precision, recall, _ = precision_recall_curve(y_val, scores)
                ax_pr.plot(recall, precision, label=f'{name} {method}')
            # Thresholds are explored on validation; reported costs are in-sample, not final test estimates.
            candidates = np.unique(np.r_[0.0, scores, 1.0, np.nextafter(1.0, 2.0)])
            for ratio in COST_RATIOS:
                best = None
                for threshold in candidates:
                    tp, fp, fn, tn = counts(y_val, scores, threshold)
                    cost = fp + ratio * fn
                    item = (cost, -tp, float(threshold), tp, fp, fn, tn)
                    if best is None or item < best:
                        best = item
                cost, _, threshold, tp, fp, fn, tn = best
                threshold_rows.append({'model': name, 'calibration': method, 'fn_to_fp_cost_ratio': ratio, 'threshold': threshold, 'total_cost_units': cost, 'cost_per_robot_day': cost / len(y_val), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1), 'alerts': tp + fp, 'no_alert_cost_units': ratio * int(y_val.sum())})
        axis.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        axis.set_title(name)
        axis.set_xlabel('Mean predicted probability')
        axis.set_ylabel('Observed event rate')
        axis.legend()
    fig_cal.suptitle('Validation calibration (exploratory; temporal internal calibration)')
    fig_cal.tight_layout()
    fig_cal.savefig(OUTPUT / 'W04_calibration_curves.png', dpi=160)
    plt.close(fig_cal)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Validation precision–recall curves')
    ax_pr.legend()
    fig_pr.tight_layout()
    fig_pr.savefig(OUTPUT / 'W04_precision_recall_curves.png', dpi=160)
    plt.close(fig_pr)
    metric_df = pd.DataFrame(metrics)
    cost_df = pd.DataFrame(threshold_rows)
    metric_df.to_csv(OUTPUT / 'W04_calibration_metrics.csv', index=False)
    cost_df.to_csv(OUTPUT / 'W04_threshold_sensitivity.csv', index=False)
    predictions.to_csv(OUTPUT / 'W04_calibrated_validation_predictions.csv', index=False)
    print('\nValidation calibration metrics (exploratory):')
    print(metric_df.round(5).to_string(index=False))
    print('\nCost sensitivity:')
    print(cost_df[['model', 'calibration', 'fn_to_fp_cost_ratio', 'threshold', 'total_cost_units', 'alerts', 'precision', 'recall']].round(4).to_string(index=False))
    print('\nSaved reports/W04/W04_calibration_metrics.csv, W04_threshold_sensitivity.csv, W04_calibrated_validation_predictions.csv and two PNG charts')
    print('NOTE: Cost thresholds optimized on validation are exploratory and optimistic; do not report them as test performance.')
    print('NOTE: Calibration model uses earlier training only, but models were shortlisted using validation; reported metrics are not unbiased final estimates.')


if __name__ == '__main__':
    main()
