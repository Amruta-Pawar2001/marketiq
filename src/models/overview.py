from __future__ import annotations

import json

import numpy as np

from src.config import METRICS_PATH
from src.models.demand import demand_score
from src.models.predict import load_model
from src.models.train import representative_discounts_from_predictions
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features
from src.pipeline.ingest import ingest
from src.sentiment.analyzer import analyze_text, review_text


def overview_metrics(sentiment_sample: int = 250) -> dict:
    df = add_features(clean_data(ingest()))
    model = load_model()
    predictions = representative_discounts_from_predictions(model.predict(df[FEATURE_COLUMNS]))
    traction_scores = demand_score(df)

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
        "model_r2": None,
        "model_rmse": None,
        "model_mae": None,
        "model_accuracy": round(float(metrics["accuracy"]), 4) if "accuracy" in metrics else None,
        "model_f1_macro": round(float(metrics["f1_macro"]), 4) if "f1_macro" in metrics else None,
        "rag_factuality_rate": None,
        "rag_status": "Needs evaluation set",
        "avg_market_traction": round(float(np.mean(traction_scores)), 2),
        "market_traction_basis": "rating count, rating, and discount proxy",
        "products_scored": int(len(df)),
        "sentiment_sample_size": int(len(sample)),
    }
