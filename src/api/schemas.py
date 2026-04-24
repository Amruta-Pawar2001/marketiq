from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    category: str = Field(..., examples=["electronics"])
    actual_price: float = Field(..., gt=0)
    discounted_price: float | None = Field(default=None, gt=0)
    rating: float = Field(..., ge=0, le=5)
    rating_count: int = Field(..., ge=0)
    product_name: str = ""
    about_product: str = ""
    review_title: str = ""
    review_content: str = ""


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    discount_pct: float
    confidence: float
    model_version: str
    feature_signals: dict


class DemandResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    demand_score: float
    model_version: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    contexts: list[dict]
    filters_used: dict
    factuality_score: float
