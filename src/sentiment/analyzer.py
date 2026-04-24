from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import SENTIMENT_MODEL_PATH, ensure_dirs
from src.pipeline.clean import clean_data
from src.pipeline.ingest import ingest


def review_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["review_title"].fillna("").astype(str)
        + " "
        + df["review_content"].fillna("").astype(str)
        + " "
        + df["about_product"].fillna("").astype(str)
    )


def sentiment_label(rating: float) -> str:
    if rating >= 4.1:
        return "positive"
    if rating <= 3.5:
        return "negative"
    return "neutral"


def train_sentiment_model() -> dict:
    ensure_dirs()
    df = clean_data(ingest())
    X = review_text(df)
    y = df["rating"].map(sentiment_label)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, preds)), "rows": int(len(df))}
    joblib.dump({"model": model, "metrics": metrics}, SENTIMENT_MODEL_PATH)
    return metrics


@lru_cache(maxsize=1)
def load_sentiment_model():
    if not SENTIMENT_MODEL_PATH.exists():
        train_sentiment_model()
    return joblib.load(SENTIMENT_MODEL_PATH)


def analyze_text(text: str) -> dict:
    bundle = load_sentiment_model()
    model = bundle["model"]
    label = str(model.predict([text])[0])
    probabilities = {}
    if hasattr(model.named_steps["clf"], "predict_proba"):
        probs = model.predict_proba([text])[0]
        probabilities = {
            cls: round(float(prob), 3)
            for cls, prob in zip(model.named_steps["clf"].classes_, probs)
        }
    score = probabilities.get("positive", 0.0) - probabilities.get("negative", 0.0)
    return {
        "label": label,
        "sentiment_score": round(float(score), 3),
        "probabilities": probabilities,
        "model_metrics": bundle.get("metrics", {}),
    }


def analyze_product(product_id: str) -> dict:
    df = clean_data(ingest())
    row = df[df["product_id"] == product_id]
    if row.empty:
        raise KeyError(product_id)
    text = review_text(row).iloc[0]
    result = analyze_text(text)
    result["product_id"] = product_id
    result["product_name"] = row.iloc[0]["product_name"]
    return result


if __name__ == "__main__":
    print(train_sentiment_model())

