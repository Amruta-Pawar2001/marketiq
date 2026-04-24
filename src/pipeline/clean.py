from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.drop_duplicates(subset=["product_id"], keep="first")
    clean = clean[clean["actual_price"].notna() & (clean["actual_price"] > 0)]
    clean = clean[clean["discount_percentage"].notna()]

    clean["category"] = (
        clean["category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.replace("|", ">", regex=False)
        .str.lower()
    )
    clean["main_category"] = clean["category"].str.split(">").str[0].str.strip()

    rating_median = clean.groupby("main_category")["rating"].transform("median")
    clean["rating"] = clean["rating"].fillna(rating_median).fillna(clean["rating"].median())
    clean["rating_count"] = clean["rating_count"].fillna(0).clip(lower=0)
    clean["discount_percentage"] = clean["discount_percentage"].clip(lower=0, upper=80)
    clean["discounted_price"] = clean["discounted_price"].fillna(
        clean["actual_price"] * (1 - clean["discount_percentage"] / 100)
    )

    text_cols = ["product_name", "about_product", "review_title", "review_content"]
    for col in text_cols:
        clean[col] = clean[col].fillna("").astype(str)

    return clean.reset_index(drop=True)

