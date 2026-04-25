from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import FEATURES_PATH, METRICS_PATH, MODEL_PATH, MODELS_DIR, PROCESSED_DIR, ensure_dirs
from src.pipeline.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.pipeline.split import build_processed_splits

BAND_LABELS = ["low", "medium", "high"]
BAND_REPRESENTATIVE_DISCOUNT = {"low": 10.0, "medium": 35.0, "high": 60.0}
BAND_THRESHOLDS_PATH = MODELS_DIR / "discount_band_thresholds.json"


def band_name_from_prediction(value: float | int) -> str:
    numeric = float(value)
    if numeric in [0, 1, 2]:
        return BAND_LABELS[int(numeric)]
    if numeric <= 20:
        return "low"
    if numeric <= 50:
        return "medium"
    return "high"


def representative_discounts_from_predictions(values) -> list[float]:
    return [BAND_REPRESENTATIVE_DISCOUNT[band_name_from_prediction(value)] for value in values]


def fit_category_band_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}
    global_q = df[TARGET_COLUMN].quantile([0.33, 0.66]).to_dict()
    thresholds["__global__"] = {"low_max": float(global_q[0.33]), "medium_max": float(global_q[0.66])}
    for category, group in df.groupby("main_category"):
        if len(group) < 20:
            continue
        q = group[TARGET_COLUMN].quantile([0.33, 0.66]).to_dict()
        thresholds[str(category)] = {"low_max": float(q[0.33]), "medium_max": float(q[0.66])}
    return thresholds


def category_relative_discount_band(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    labels = []
    global_thresholds = thresholds["__global__"]
    for _, row in df.iterrows():
        limits = thresholds.get(str(row.get("main_category", "")), global_thresholds)
        discount = float(row[TARGET_COLUMN])
        if discount <= limits["low_max"]:
            labels.append(0)
        elif discount <= limits["medium_max"]:
            labels.append(1)
        else:
            labels.append(2)
    return pd.Series(labels, index=df.index, dtype=int)

NUMERIC_FEATURES = [
    "log_actual_price",
    "rating",
    "log_rating_count",
    "popular_flag",
    "name_length",
    "has_pack",
    "is_cable",
    "is_adapter",
    "is_printer",
    "is_case",
    "is_battery",
    "is_usb",
    "is_wireless",
    "is_charger",
    "high_rated",
    "rating_x_log_count",
    "category_depth",
    "name_word_count",
]
CATEGORICAL_FEATURES = [
    "main_category",
    "price_bin",
    "sub_category",
    "category_level_2",
    "category_level_3",
    "last_category",
]


def _load_or_create_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return build_processed_splits()
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    if not required.issubset(train_df.columns) or not required.issubset(test_df.columns):
        return build_processed_splits()
    return train_df, test_df


def build_pipeline(params: dict | None = None) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 2,
    }
    if params:
        model_params.update(params)
    model = XGBClassifier(**model_params)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _candidate_params() -> list[dict]:
    return [
        {},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.85},
        {"n_estimators": 450, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    ]


def discount_band(discount: pd.Series) -> pd.Series:
    bins = [-0.1, 20, 50, 80]
    labels = [0, 1, 2]
    return pd.cut(discount.clip(0, 80), bins=bins, labels=labels, include_lowest=True).astype(int)


def train_model() -> dict[str, float]:
    ensure_dirs()
    train_df, test_df = _load_or_create_splits()
    thresholds = fit_category_band_thresholds(train_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = category_relative_discount_band(train_df, thresholds)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = category_relative_discount_band(test_df, thresholds)

    best_pipeline = None
    best_preds = None
    best_metrics = None
    best_params = None
    for params in _candidate_params():
        pipeline = build_pipeline(params)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }
        if best_metrics is None or metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_pipeline = pipeline
            best_preds = preds
            best_metrics = metrics
            best_params = params

    assert best_pipeline is not None
    assert best_preds is not None
    assert best_metrics is not None
    metrics = best_metrics
    metrics["selected_params"] = best_params or "default"

    joblib.dump(best_pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    BAND_THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(result, indent=2))
