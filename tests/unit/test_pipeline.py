import pandas as pd

from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features, model_frame
from src.pipeline.ingest import parse_numeric


def test_parse_numeric_handles_currency_percent_and_counts():
    assert parse_numeric("â‚¹1,299") == 1299
    assert parse_numeric("64%") == 64
    assert parse_numeric("24,269") == 24269


def test_clean_and_features_create_model_columns():
    df = pd.DataFrame(
        [
            {
                "product_id": "p1",
                "product_name": "Cable",
                "category": "Computers&Accessories|Cables",
                "discounted_price": 399,
                "actual_price": 1099,
                "discount_percentage": 64,
                "rating": 4.2,
                "rating_count": 24269,
                "about_product": "Fast cable",
                "review_title": "Good",
                "review_content": "Good value",
            }
        ]
    )
    clean = clean_data(df)
    featured = add_features(clean)
    X, y = model_frame(featured)
    assert set(FEATURE_COLUMNS).issubset(X.columns)
    assert y.iloc[0] == 64
    assert featured["main_category"].iloc[0] == "computers&accessories"

