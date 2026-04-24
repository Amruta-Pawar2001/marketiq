from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.monitoring.drift import drift_report
from src.segments.clustering import segment_summary
from src.sentiment.analyzer import analyze_product

router = APIRouter(tags=["insights"])


@router.get("/sentiment/{product_id}")
def sentiment(product_id: str) -> dict:
    try:
        return analyze_product(product_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.get("/segments")
def segments() -> dict:
    return segment_summary()


@router.get("/drift")
def drift() -> dict:
    return drift_report()
