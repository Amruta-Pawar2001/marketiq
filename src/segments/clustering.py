from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SEGMENT_MODEL_PATH, SEGMENT_SUMMARY_PATH, ensure_dirs
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features
from src.pipeline.ingest import ingest

SEGMENT_FEATURES = [
    "actual_price",
    "discounted_price",
    "price_ratio",
    "rating",
    "rating_count",
    "review_count_log",
    "discount_percentage",
    "main_category",
]


def _frame() -> pd.DataFrame:
    return add_features(clean_data(ingest()))


def train_segments(n_clusters: int = 4) -> dict:
    ensure_dirs()
    df = _frame()
    features = df[SEGMENT_FEATURES]
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), [c for c in SEGMENT_FEATURES if c != "main_category"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["main_category"]),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ]
    )
    labels = model.fit_predict(features)
    df["segment_id"] = labels
    summary = summarize_segments(df)
    joblib.dump(model, SEGMENT_MODEL_PATH)
    SEGMENT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_segments(df: pd.DataFrame) -> dict:
    grouped = (
        df.groupby("segment_id")
        .agg(
            products=("product_id", "count"),
            avg_price=("actual_price", "mean"),
            avg_discount=("discount_percentage", "mean"),
            avg_rating=("rating", "mean"),
            avg_rating_count=("rating_count", "mean"),
        )
        .round(2)
        .reset_index()
    )
    labels = {}
    for _, row in grouped.iterrows():
        sid = int(row["segment_id"])
        if row["avg_rating"] >= 4.1 and row["avg_rating_count"] >= grouped["avg_rating_count"].median():
            labels[sid] = "high-trust winners"
        elif row["avg_discount"] >= grouped["avg_discount"].median():
            labels[sid] = "deal-sensitive products"
        elif row["avg_price"] >= grouped["avg_price"].median():
            labels[sid] = "premium niche"
        else:
            labels[sid] = "budget long-tail"
    records = grouped.to_dict(orient="records")
    for record in records:
        record["label"] = labels[int(record["segment_id"])]
    return {"segments": records}


@lru_cache(maxsize=1)
def load_segment_model():
    if not SEGMENT_MODEL_PATH.exists() or not SEGMENT_SUMMARY_PATH.exists():
        train_segments()
    return joblib.load(SEGMENT_MODEL_PATH)


def segment_summary() -> dict:
    if not SEGMENT_SUMMARY_PATH.exists():
        return train_segments()
    return json.loads(SEGMENT_SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(train_segments(), indent=2))

