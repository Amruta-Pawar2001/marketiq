from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import METRICS_PATH, MODEL_PATH
from src.models.train import train_model
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def _default_payload(payload: dict) -> dict:
    data = {
        "product_id": payload.get("product_id", "api-product"),
        "product_name": payload.get("product_name", ""),
        "category": payload.get("category", "unknown"),
        "actual_price": payload.get("actual_price"),
        "discounted_price": payload.get("discounted_price"),
        "discount_percentage": payload.get("discount_percentage", 0),
        "rating": payload.get("rating", 0),
        "rating_count": payload.get("rating_count", 0),
        "about_product": payload.get("about_product", ""),
        "review_title": payload.get("review_title", ""),
        "review_content": payload.get("review_content", ""),
    }
    if data["discounted_price"] in [None, ""]:
        data["discounted_price"] = float(data["actual_price"]) * 0.75
    return data


def prepare_input(payload: dict) -> pd.DataFrame:
    raw = pd.DataFrame([_default_payload(payload)])
    raw["actual_price"] = pd.to_numeric(raw["actual_price"], errors="coerce")
    raw["discounted_price"] = pd.to_numeric(raw["discounted_price"], errors="coerce")
    raw["discount_percentage"] = pd.to_numeric(raw["discount_percentage"], errors="coerce").fillna(0)
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce").fillna(0)
    raw["rating_count"] = pd.to_numeric(raw["rating_count"], errors="coerce").fillna(0)
    return add_features(clean_data(raw))[FEATURE_COLUMNS]


def predict_discount(payload: dict) -> dict:
    model = load_model()
    X = prepare_input(payload)
    pred = float(np.clip(model.predict(X)[0], 0, 80))
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    confidence = max(0.0, min(1.0, 1.0 - float(metrics.get("mae", 12.0)) / 80.0))
    signals = {
        "actual_price": float(X.iloc[0]["actual_price"]),
        "rating": float(X.iloc[0]["rating"]),
        "rating_count": float(X.iloc[0]["rating_count"]),
        "price_ratio": float(X.iloc[0]["price_ratio"]),
        "main_category": str(X.iloc[0]["main_category"]),
    }
    return {
        "discount_pct": round(pred, 2),
        "confidence": round(confidence, 3),
        "model_version": "xgboost-v1",
        "feature_signals": signals,
    }

