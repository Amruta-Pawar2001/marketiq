from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import DemandResponse, PredictRequest, PredictResponse
from src.models.demand import predict_demand
from src.models.predict import predict_discount
from src.monitoring.metrics import track

router = APIRouter(tags=["prediction"])


@router.post("/predict_discount", response_model=PredictResponse)
def predict_discount_route(payload: PredictRequest) -> dict:
    with track("predict_discount"):
        return predict_discount(payload.model_dump())


@router.post("/predict_demand", response_model=DemandResponse)
def predict_demand_route(payload: PredictRequest) -> dict:
    with track("predict_demand"):
        return predict_demand(payload.model_dump())
