from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "actual_price",
    "discounted_price",
    "price_ratio",
    "log_price",
    "rating",
    "rating_count",
    "review_count_log",
    "category_depth",
    "about_length",
    "review_length",
    "high_rated",
    "rating_bin",
    "main_category",
]
TARGET_COLUMN = "discount_percentage"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["price_ratio"] = (data["discounted_price"] / data["actual_price"]).replace(
        [np.inf, -np.inf], np.nan
    )
    data["price_ratio"] = data["price_ratio"].fillna(data["price_ratio"].median())
    data["log_price"] = np.log1p(data["actual_price"])
    data["review_count_log"] = np.log1p(data["rating_count"])
    data["category_depth"] = data["category"].fillna("").astype(str).str.count(">") + 1
    data["about_length"] = data["about_product"].fillna("").astype(str).str.len()
    data["review_length"] = data["review_content"].fillna("").astype(str).str.len()
    data["high_rated"] = (data["rating"] >= 4.0).astype(int)
    data["rating_bin"] = pd.cut(
        data["rating"],
        bins=[0, 3, 4, 4.5, 5],
        labels=["low", "med", "high", "top"],
        include_lowest=True,
    ).astype(str)
    data["rating_bin"] = data["rating_bin"].replace("nan", "unknown")
    return data


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = add_features(df)
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].copy()

