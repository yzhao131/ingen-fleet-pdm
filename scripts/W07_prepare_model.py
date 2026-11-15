
#!/usr/bin/env python3
"""Week 7: reproduce and persist the Week 4 RF + Platt model."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT = Path("data/causal_preview/processed/W03_robot_day_features.csv")
MODEL_DIR = Path("models/W07")
REPORT_DIR = Path("reports/W07")

TARGET = "severe_fault_next_7d"
EXCLUDE = {
    "unit_id", "date", "during_maintenance",
    TARGET, "split", "commission_date"
}

MODEL_VERSION = "w07_rf_platt_v1"
FEATURE_VERSION = "W03_v1"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if df.duplicated(["unit_id", "date"]).any():
        raise ValueError("Duplicate robot-day rows")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    if train.empty or val.empty:
        raise ValueError("Missing training or validation data")

    if train["date"].max() >= val["date"].min():
        raise ValueError("Train and validation dates overlap")

    # Match Week 4's internal temporal calibration split.
    unique_dates = np.sort(train["date"].unique())
    boundary = unique_dates[int(len(unique_dates) * 0.8)]

    fit = train[
        train["date"] < boundary - pd.Timedelta(days=7)
    ].copy()

    cal = train[
        train["date"] >= boundary
    ].copy()

    if min(
        fit[TARGET].nunique(),
        cal[TARGET].nunique(),
        val[TARGET].nunique()
    ) < 2:
        raise ValueError("One split lacks a target class")

    feature_cols = [
        c for c in df.columns if c not in EXCLUDE
    ]

    categorical = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    numeric = [
        c for c in feature_cols
        if c not in categorical
    ]

    transformers = [
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )),
                ("scale", StandardScaler()),
            ]),
            numeric,
        )
    ]

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(
                        strategy="most_frequent"
                    )),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore"
                    )),
                ]),
                categorical,
            )
        )

    preprocess = ColumnTransformer(transformers)

    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])

    print("Training Random Forest...")
    pipeline.fit(
        fit[feature_cols],
        fit[TARGET].astype(int)
    )

    print("Fitting Platt calibrator...")
    cal_raw = pipeline.predict_proba(
        cal[feature_cols]
    )[:, 1]

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(
        cal_raw.reshape(-1, 1),
        cal[TARGET].astype(int)
    )

    val_raw = pipeline.predict_proba(
        val[feature_cols]
    )[:, 1]

    val_calibrated = calibrator.predict_proba(
        val_raw.reshape(-1, 1)
    )[:, 1]

    model_path = MODEL_DIR / "rf_platt.joblib"

    bundle = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "feature_columns": feature_cols,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "sklearn_version": sklearn.__version__,
    }

    joblib.dump(bundle, model_path)

    # Verify that the persisted model produces the same predictions.
    loaded = joblib.load(model_path)

    reloaded_raw = loaded["pipeline"].predict_proba(
        val[loaded["feature_columns"]]
    )[:, 1]

    reloaded_calibrated = loaded[
        "calibrator"
    ].predict_proba(
        reloaded_raw.reshape(-1, 1)
    )[:, 1]

    np.testing.assert_allclose(
        val_calibrated,
        reloaded_calibrated,
        rtol=1e-12,
        atol=1e-12,
    )

    # Compare against the original Week 4 validation output.
    reference_path = Path(
        "reports/W04/W04_calibrated_validation_predictions.csv"
    )

    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    reference = pd.read_csv(reference_path)
    reference["date"] = pd.to_datetime(
        reference["date"], utc=True
    )

    comparison = val[["unit_id", "date"]].copy()
    comparison["reproduced_probability"] = val_calibrated

    comparison = comparison.merge(
        reference[[
            "unit_id", "date", "rf_depth4_leaf50_platt"
        ]],
        on=["unit_id", "date"],
        how="left",
        validate="one_to_one",
    )

    if comparison["rf_depth4_leaf50_platt"].isna().any():
        raise ValueError("Missing Week 4 reference predictions")

    max_difference = float(
        np.max(
            np.abs(
                comparison["reproduced_probability"]
                - comparison["rf_depth4_leaf50_platt"]
            )
        )
    )

    if max_difference > 1e-8:
        raise ValueError(
            f"Week 4 predictions differ: {max_difference}"
        )

    y_val = val[TARGET].astype(int)

    metrics = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "model": "RandomForest + Platt",
        "validation_rows": int(len(val)),
        "validation_positives": int(y_val.sum()),
        "validation_ap": float(
            average_precision_score(y_val, val_calibrated)
        ),
        "validation_brier": float(
            brier_score_loss(y_val, val_calibrated)
        ),
        "max_week4_prediction_difference": max_difference,
        "sklearn_version": sklearn.__version__,
    }

    with open(
        REPORT_DIR / "W07_model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nMODEL SAVED:", model_path)
    print("Validation AP:", round(metrics["validation_ap"], 6))
    print("Validation Brier:", round(metrics["validation_brier"], 6))
    print("Max Week 4 difference:", max_difference)
    print("Reload verification: PASSED")


if __name__ == "__main__":
    main()
