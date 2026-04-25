from src.models.predict import prepare_input
from src.models.overview import overview_metrics
from src.models.top_products import _flag
from src.pipeline.features import FEATURE_COLUMNS


def test_prepare_input_returns_feature_frame():
    frame = prepare_input(
        {
            "category": "electronics",
            "actual_price": 2499,
            "discounted_price": 1699,
            "rating": 4.2,
            "rating_count": 1240,
        }
    )
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.iloc[0]["high_rated"] == 1


def test_demand_score_increases_with_popularity():
    import pandas as pd

    from src.models.demand import demand_score

    df = pd.DataFrame(
        {
            "rating_count": [10, 10000],
            "rating": [4.0, 4.0],
            "discount_percentage": [20, 20],
        }
    )
    scores = demand_score(df)
    assert scores.iloc[1] > scores.iloc[0]


def test_top_product_flag_rules():
    assert _flag(2, 0.4) == ("Good", "green")
    assert _flag(20, 0.4) == ("Review", "red")
    assert _flag(8, 0.1) == ("Watch", "yellow")


def test_overview_metrics_shape(monkeypatch):
    import pandas as pd

    from src.models import overview as overview_module

    df = pd.DataFrame(
        {
            "product_id": ["p1"],
            "product_name": ["Cable"],
            "category": ["electronics"],
            "actual_price": [100.0],
            "discounted_price": [80.0],
            "discount_percentage": [20.0],
            "rating": [4.2],
            "rating_count": [10],
            "about_product": ["Good quality cable"],
            "review_title": ["Good"],
            "review_content": ["Good value"],
        }
    )

    class FakeModel:
        def predict(self, X):
            return [25.0]

    monkeypatch.setattr(overview_module, "ingest", lambda: df)
    monkeypatch.setattr(overview_module, "load_model", lambda: FakeModel())
    monkeypatch.setattr(overview_module, "analyze_text", lambda text: {"sentiment_score": 0.5})
    metrics = overview_metrics(sentiment_sample=1)
    assert metrics["avg_predicted_discount"] == 35.0
    assert metrics["avg_sentiment_score"] == 0.5


def test_dashboard_discount_distribution_shape(monkeypatch):
    import pandas as pd

    from src.models import dashboard as dashboard_module

    df = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_name": ["Cable", "Stand"],
            "category": ["electronics", "office"],
            "actual_price": [100.0, 200.0],
            "discounted_price": [80.0, 150.0],
            "discount_percentage": [20.0, 25.0],
            "rating": [4.2, 4.0],
            "rating_count": [10, 20],
            "about_product": ["Good quality cable", "Good stand"],
            "review_title": ["Good", "Fine"],
            "review_content": ["Good value", "Works"],
        }
    )

    class FakeModel:
        def predict(self, X):
            return [25.0, 35.0]

    monkeypatch.setattr(dashboard_module, "ingest", lambda: df)
    monkeypatch.setattr(dashboard_module, "load_model", lambda: FakeModel())
    result = dashboard_module.discount_distribution()
    assert result["categories"][0]["avg_predicted_discount"] >= 25
