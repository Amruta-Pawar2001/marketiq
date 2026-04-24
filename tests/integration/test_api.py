from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import answer as answer_route
from src.api.routes import insights as insights_route
from src.api.routes import predict as predict_route


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_discount(monkeypatch):
    monkeypatch.setattr(
        predict_route,
        "predict_discount",
        lambda payload: {
            "discount_pct": 34.5,
            "confidence": 0.9,
            "model_version": "xgboost-v1",
            "feature_signals": {},
        },
    )
    response = client.post(
        "/predict_discount",
        json={"category": "electronics", "actual_price": 2499, "rating": 4.2, "rating_count": 1240},
    )
    assert response.status_code == 200
    assert response.json()["discount_pct"] == 34.5


def test_predict_demand(monkeypatch):
    monkeypatch.setattr(
        predict_route,
        "predict_demand",
        lambda payload: {"demand_score": 4.2, "model_version": "xgboost-demand-v1"},
    )
    response = client.post(
        "/predict_demand",
        json={"category": "electronics", "actual_price": 2499, "rating": 4.2, "rating_count": 1240},
    )
    assert response.status_code == 200
    assert response.json()["demand_score"] == 4.2


def test_answer_question(monkeypatch):
    monkeypatch.setattr(
        answer_route,
        "answer_question",
        lambda question, top_k=5: {
            "answer": "A cable is relevant.",
            "sources": [{"product_id": "p1"}],
            "contexts": [],
            "filters_used": {},
            "factuality_score": 0.8,
        },
    )
    response = client.post("/answer_question", json={"question": "What cable should I buy?", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["sources"][0]["product_id"] == "p1"


def test_top_products(monkeypatch):
    monkeypatch.setattr(
        insights_route,
        "top_products",
        lambda limit=5: {
            "rows": [
                {
                    "product_id": "p1",
                    "product_name": "Cable",
                    "actual_discount": 40,
                    "predicted_discount": 35,
                    "gap": 5,
                    "sentiment_score": 0.6,
                    "sentiment_label": "positive",
                    "flag": "Good",
                    "flag_color": "green",
                }
            ]
        },
    )
    response = client.get("/top_products?limit=1")
    assert response.status_code == 200
    assert response.json()["rows"][0]["product_name"] == "Cable"
