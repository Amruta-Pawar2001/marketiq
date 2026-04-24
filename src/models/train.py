from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import FEATURES_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DIR, ensure_dirs
from src.pipeline.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.pipeline.split import build_processed_splits

NUMERIC_FEATURES = [
    "actual_price",
    "discounted_price",
    "price_ratio",
    "log_price",
    "rating",
    "rating_count",
    "review_count_log",
    "category_depth",
    "about_length",
    "review_length",
    "high_rated",
]
CATEGORICAL_FEATURES = ["rating_bin", "main_category"]


def _load_or_create_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return build_processed_splits()
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=2,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_model() -> dict[str, float]:
    ensure_dirs()
    train_df, test_df = _load_or_create_splits()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = np.clip(pipeline.predict(X_test), 0, 80)

    metrics = {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    joblib.dump(pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(result, indent=2))

