from src.models.predict import prepare_input
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
