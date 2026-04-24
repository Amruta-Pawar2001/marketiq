from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    META_PATH,
    METRICS_PATH,
    OPENAI_MODEL,
    TEXTS_PATH,
)
from src.models.predict import load_model
from src.monitoring.drift import drift_report
from src.monitoring.metrics import snapshot
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features
from src.pipeline.ingest import ingest
from src.rag.chunker import load_chunks
from src.segments.clustering import segment_summary
from src.sentiment.analyzer import analyze_text, review_text


def discount_distribution() -> dict:
    df = add_features(clean_data(ingest()))
    preds = np.clip(load_model().predict(df[FEATURE_COLUMNS]), 0, 80)
    work = df.assign(predicted_discount=preds)
    grouped = (
        work.groupby("main_category")
        .agg(avg_predicted_discount=("predicted_discount", "mean"), products=("product_id", "count"))
        .sort_values("avg_predicted_discount", ascending=False)
        .head(8)
        .round(2)
        .reset_index()
    )
    return {"categories": grouped.to_dict(orient="records")}


def model_health() -> dict:
    metrics = {}
    if METRICS_PATH.exists():
        import json

        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "rmse": round(float(metrics.get("rmse", 0)), 4),
        "mae": round(float(metrics.get("mae", 0)), 4),
        "r2": round(float(metrics.get("r2", 0)), 4),
        "train_rows": int(metrics.get("train_rows", 0)),
        "test_rows": int(metrics.get("test_rows", 0)),
        "model": "XGBoost discount regressor",
    }


def review_insights(limit: int = 3) -> dict:
    df = clean_data(ingest())
    df["text"] = review_text(df)
    sample = df.sort_values("rating_count", ascending=False).head(250).copy()
    scored = []
    for _, row in sample.iterrows():
        sentiment = analyze_text(row["text"])
        scored.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "rating": float(row["rating"]),
                "rating_count": int(row["rating_count"]),
                "sentiment_score": float(sentiment["sentiment_score"]),
                "label": sentiment["label"],
                "snippet": str(row["review_content"])[:180],
            }
        )
    ranked = sorted(scored, key=lambda x: (abs(x["sentiment_score"]), x["rating_count"]), reverse=True)
    return {"insights": ranked[:limit]}


def segment_actions() -> dict:
    actions = []
    for segment in segment_summary().get("segments", []):
        label = segment.get("label", "segment")
        avg_discount = float(segment.get("avg_discount", 0))
        avg_rating = float(segment.get("avg_rating", 0))
        if "high-trust" in label:
            action = "Protect margin and feature these products in premium placements."
        elif "deal-sensitive" in label or avg_discount >= 50:
            action = "Use controlled promotions and monitor discount fatigue."
        elif avg_rating < 4:
            action = "Review product quality signals before increasing spend."
        else:
            action = "Test small discount lifts and compare conversion response."
        actions.append(
            {
                "segment_id": segment.get("segment_id"),
                "label": label,
                "products": segment.get("products"),
                "avg_discount": avg_discount,
                "avg_rating": avg_rating,
                "action": action,
            }
        )
    return {"actions": actions}


def vector_index_health() -> dict:
    start = time.perf_counter()
    chunks = load_chunks()
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    meta = pd.read_pickle(META_PATH) if META_PATH.exists() else pd.DataFrame()
    files = {
        "index": FAISS_INDEX_PATH,
        "embeddings": EMBEDDINGS_PATH,
        "texts": TEXTS_PATH,
        "metadata": META_PATH,
        "chunks": CHUNKS_PATH,
    }
    return {
        "total_chunks": len(chunks),
        "products_indexed": int(meta["product_id"].nunique()) if "product_id" in meta else 0,
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": "FAISS",
        "generation_model": OPENAI_MODEL,
        "load_ms": elapsed_ms,
        "artifact_sizes_mb": {
            name: round(Path(path).stat().st_size / (1024 * 1024), 2) if Path(path).exists() else 0
            for name, path in files.items()
        },
    }


def monitoring_dashboard() -> dict:
    drift = drift_report()
    metrics = snapshot()
    endpoint_rows = []
    for key, value in metrics.get("latency", {}).items():
        endpoint = key.replace("_latency_seconds", "")
        endpoint_rows.append(
            {
                "endpoint": f"/{endpoint}",
                "avg_latency_ms": round(float(value.get("avg", 0)) * 1000, 2),
                "max_latency_ms": round(float(value.get("max", 0)) * 1000, 2),
                "count": int(value.get("count", 0)),
            }
        )
    return {
        "drift": drift,
        "metrics": metrics,
        "endpoint_rows": endpoint_rows,
        "retrain_policy": {
            "drift_threshold": drift["threshold"],
            "recommendation": drift["recommendation"],
            "automation_status": "planned",
            "trigger_rule": "Retrain when drift remains above threshold for repeated checks.",
        },
    }
