from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.models.dashboard import (
    discount_distribution,
    model_health,
    monitoring_dashboard,
    review_insights,
    segment_actions,
    vector_index_health,
)
from src.monitoring.drift import drift_report
from src.models.overview import overview_metrics
from src.models.top_products import top_products
from src.segments.clustering import segment_summary
from src.sentiment.analyzer import analyze_product, sentiment_heatmap

router = APIRouter(tags=["insights"])


@router.get("/sentiment/{product_id}")
def sentiment(product_id: str) -> dict:
    try:
        return analyze_product(product_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.get("/sentiment_heatmap")
def heatmap(top_n: int = 5) -> dict:
    return sentiment_heatmap(top_n=top_n)


@router.get("/top_products")
def top_products_route(limit: int = 5) -> dict:
    return top_products(limit=limit)


@router.get("/overview_metrics")
def overview_metrics_route() -> dict:
    return overview_metrics()


@router.get("/discount_distribution")
def discount_distribution_route() -> dict:
    return discount_distribution()


@router.get("/model_health")
def model_health_route() -> dict:
    return model_health()


@router.get("/review_insights")
def review_insights_route(limit: int = 3) -> dict:
    return review_insights(limit=limit)


@router.get("/segment_actions")
def segment_actions_route() -> dict:
    return segment_actions()


@router.get("/vector_index_health")
def vector_index_health_route() -> dict:
    return vector_index_health()


@router.get("/monitoring_dashboard")
def monitoring_dashboard_route() -> dict:
    return monitoring_dashboard()


@router.get("/segments")
def segments() -> dict:
    return segment_summary()


@router.get("/drift")
def drift() -> dict:
    return drift_report()
