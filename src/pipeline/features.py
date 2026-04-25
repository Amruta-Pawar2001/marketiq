from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_actual_price",
    "main_category",
    "price_bin",
    "log_rating_count",
    "rating",
    "popular_flag",
    "sub_category",
    "category_level_2",
    "category_level_3",
    "last_category",
    "name_length",
    "has_pack",
    "is_cable",
    "is_adapter",
    "is_printer",
    "is_case",
    "is_battery",
    "is_usb",
    "is_wireless",
    "is_charger",
    "high_rated",
    "rating_x_log_count",
    "category_depth",
    "name_word_count",
]
TARGET_COLUMN = "discount_percentage"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["price_ratio"] = (data["discounted_price"] / data["actual_price"]).replace(
        [np.inf, -np.inf], np.nan
    )
    data["price_ratio"] = data["price_ratio"].fillna(data["price_ratio"].median())
    data["log_price"] = np.log1p(data["actual_price"])
    data["log_actual_price"] = data["log_price"]
    data["review_count_log"] = np.log1p(data["rating_count"])
    data["log_rating_count"] = data["review_count_log"]
    data["category_depth"] = data["category"].fillna("").astype(str).str.count(">") + 1
    category_parts = data["category"].fillna("unknown").astype(str).str.split(">")
    data["sub_category"] = category_parts.str[1].fillna("unknown").str.strip()
    data["category_level_2"] = data["sub_category"]
    data["category_level_3"] = category_parts.str[2].fillna("unknown").str.strip()
    data["last_category"] = category_parts.str[-1].fillna("unknown").str.strip()
    data["about_length"] = data["about_product"].fillna("").astype(str).str.len()
    data["review_length"] = data["review_content"].fillna("").astype(str).str.len()
    product_name = data["product_name"].fillna("").astype(str)
    product_name_lower = product_name.str.lower()
    data["name_length"] = product_name.str.len()
    data["name_word_count"] = product_name.str.split().str.len().fillna(0)
    data["has_pack"] = product_name.str.contains("pack|combo|set|kit", case=False, regex=True).astype(int)
    data["is_cable"] = product_name_lower.str.contains("cable|cord", regex=True).astype(int)
    data["is_adapter"] = product_name_lower.str.contains("adapter|adaptor|converter|hub", regex=True).astype(int)
    data["is_printer"] = product_name_lower.str.contains("printer|ink|toner", regex=True).astype(int)
    data["is_case"] = product_name_lower.str.contains("case|cover|sleeve", regex=True).astype(int)
    data["is_battery"] = product_name_lower.str.contains("battery|batteries|cell", regex=True).astype(int)
    data["is_usb"] = product_name_lower.str.contains("usb|type c|type-c|lightning", regex=True).astype(int)
    data["is_wireless"] = product_name_lower.str.contains("wireless|bluetooth|wifi|wi-fi", regex=True).astype(int)
    data["is_charger"] = product_name_lower.str.contains("charger|charging|power bank", regex=True).astype(int)
    data["high_rated"] = (data["rating"] >= 4.0).astype(int)
    data["popular_flag"] = (data["rating_count"] > data["rating_count"].median()).astype(int)
    data["rating_x_log_count"] = data["rating"] * data["log_rating_count"]
    if len(data) < 5:
        data["price_bin"] = "mid"
    else:
        data["price_bin"] = pd.qcut(
            data["actual_price"].rank(method="first"),
            q=5,
            labels=["very_low", "low", "mid", "high", "premium"],
            duplicates="drop",
        ).astype(str)
    data["price_bin"] = data["price_bin"].replace("nan", "unknown")
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
