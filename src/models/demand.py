from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import DEMAND_METRICS_PATH, DEMAND_MODEL_PATH, ensure_dirs
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features
from src.pipeline.ingest import ingest


def demand_score(df: pd.DataFrame) -> pd.Series:
    popularity = np.log1p(df["rating_count"])
    quality = df["rating"].clip(0, 5) / 5
    value = df["discount_percentage"].clip(0, 80) / 80
    return (0.55 * popularity + 0.30 * quality + 0.15 * value).round(4)


def train_demand_model() -> dict:
    ensure_dirs()
    df = add_features(clean_data(ingest()))
    df["demand_score"] = demand_score(df)
    X = df[FEATURE_COLUMNS]
    y = df["demand_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), [c for c in FEATURE_COLUMNS if c not in ["rating_bin", "main_category"]]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["rating_bin", "main_category"]),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", XGBRegressor(objective="reg:squarederror", n_estimators=250, max_depth=3, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {"mae": float(mean_absolute_error(y_test, preds)), "r2": float(r2_score(y_test, preds))}
    joblib.dump(model, DEMAND_MODEL_PATH)
    DEMAND_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@lru_cache(maxsize=1)
def load_demand_model():
    if not DEMAND_MODEL_PATH.exists():
        train_demand_model()
    return joblib.load(DEMAND_MODEL_PATH)


def predict_demand(payload: dict) -> dict:
    from src.models.predict import prepare_input

    model = load_demand_model()
    X = prepare_input(payload)
    score = float(model.predict(X)[0])
    return {"demand_score": round(score, 3), "model_version": "xgboost-demand-v1"}


if __name__ == "__main__":
    print(json.dumps(train_demand_model(), indent=2))

