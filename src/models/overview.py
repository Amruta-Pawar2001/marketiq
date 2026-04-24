from __future__ import annotations

import json

import numpy as np

from src.config import METRICS_PATH
from src.models.predict import load_model
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features
from src.pipeline.ingest import ingest
from src.sentiment.analyzer import analyze_text, review_text


def overview_metrics(sentiment_sample: int = 250) -> dict:
    df = add_features(clean_data(ingest()))
    model = load_model()
    predictions = np.clip(model.predict(df[FEATURE_COLUMNS]), 0, 80)

    sample = df.sort_values("rating_count", ascending=False).head(sentiment_sample).copy()
    sample["review_text"] = review_text(sample)
    sentiment_scores = [
        analyze_text(text)["sentiment_score"]
        for text in sample["review_text"].tolist()
    ]

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    return {
        "avg_predicted_discount": round(float(np.mean(predictions)), 2),
        "avg_sentiment_score": round(float(np.mean(sentiment_scores)), 3) if sentiment_scores else None,
        "model_r2": round(float(metrics["r2"]), 4) if "r2" in metrics else None,
        "model_rmse": round(float(metrics["rmse"]), 4) if "rmse" in metrics else None,
        "model_mae": round(float(metrics["mae"]), 4) if "mae" in metrics else None,
        "rag_factuality_rate": None,
        "rag_status": "Needs evaluation set",
        "products_scored": int(len(df)),
        "sentiment_sample_size": int(len(sample)),
    }
