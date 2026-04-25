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
        lambda question, top_k=5, history=None: {
            "answer": "A cable is relevant.",
            "sources": [{"product_id": "p1"}],
            "contexts": [],
            "filters_used": {"history_count": len(history or [])},
            "factuality_score": 0.8,
        },
    )
    response = client.post(
        "/answer_question",
        json={
            "question": "What about under 1000?",
            "top_k": 3,
            "history": [{"role": "user", "content": "What cable should I buy?"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["sources"][0]["product_id"] == "p1"
    assert response.json()["filters_used"]["history_count"] == 1


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


def test_overview_metrics(monkeypatch):
    monkeypatch.setattr(
        insights_route,
        "overview_metrics",
        lambda: {
            "avg_predicted_discount": 30.0,
            "avg_sentiment_score": 0.5,
            "model_r2": 0.9,
            "rag_factuality_rate": None,
            "rag_status": "Needs evaluation set",
            "products_scored": 10,
            "sentiment_sample_size": 10,
        },
    )
    response = client.get("/overview_metrics")
    assert response.status_code == 200
    assert response.json()["avg_predicted_discount"] == 30.0


def test_dashboard_endpoints(monkeypatch):
    monkeypatch.setattr(insights_route, "discount_distribution", lambda: {"categories": []})
    monkeypatch.setattr(insights_route, "model_health", lambda: {"rmse": 1, "mae": 1, "r2": 0.9})
    monkeypatch.setattr(insights_route, "review_insights", lambda limit=3: {"insights": []})
    monkeypatch.setattr(insights_route, "segment_actions", lambda: {"actions": []})
    monkeypatch.setattr(insights_route, "vector_index_health", lambda: {"total_chunks": 0})
    monkeypatch.setattr(insights_route, "monitoring_dashboard", lambda: {"drift": {}, "metrics": {}, "endpoint_rows": []})
    for path in [
        "/discount_distribution",
        "/model_health",
        "/review_insights",
        "/segment_actions",
        "/vector_index_health",
        "/monitoring_dashboard",
    ]:
        assert client.get(path).status_code == 200
