from __future__ import annotations

from fastapi import APIRouter

from src.config import (
    DEMAND_MODEL_PATH,
    FAISS_INDEX_PATH,
    MODEL_PATH,
    OPENAI_API_KEY,
    RAW_DATA_PATH,
    SEGMENT_MODEL_PATH,
    SENTIMENT_MODEL_PATH,
)
from src.monitoring.metrics import snapshot

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "dataset": RAW_DATA_PATH.exists(),
        "xgboost_model": MODEL_PATH.exists(),
        "demand_model": DEMAND_MODEL_PATH.exists(),
        "sentiment_model": SENTIMENT_MODEL_PATH.exists(),
        "segment_model": SEGMENT_MODEL_PATH.exists(),
        "faiss_index": FAISS_INDEX_PATH.exists(),
        "openai_configured": bool(OPENAI_API_KEY),
    }


@router.get("/metrics")
def metrics() -> dict:
    return snapshot()
