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

ASPECT_KEYWORDS = {
    "Quality": ["quality", "sturdy", "durable", "material", "build", "strong", "premium"],
    "Value": ["value", "price", "worth", "cheap", "budget", "money", "cost"],
    "Delivery": ["delivery", "delivered", "shipping", "package", "arrived", "fast"],
    "Support": ["support", "service", "warranty", "replace", "replacement", "return"],
    "Durability": ["durable", "durability", "long", "months", "break", "broken", "stopped", "life"],
}
POSITIVE_WORDS = {"good", "great", "excellent", "best", "value", "fast", "quality", "durable", "sturdy", "nice", "perfect"}
NEGATIVE_WORDS = {"bad", "poor", "slow", "issue", "problem", "not", "broken", "stopped", "worst", "damaged", "complaint"}


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


def _aspect_score(text: str, rating: float, keywords: list[str]) -> float:
    lower = text.lower()
    sentences = [part.strip() for part in lower.replace("|", ".").split(".") if part.strip()]
    relevant = [sentence for sentence in sentences if any(keyword in sentence for keyword in keywords)]
    if not relevant:
        return round(min(1.0, max(0.0, float(rating) / 5)), 2)

    hits = " ".join(relevant).split()
    positive = sum(word.strip(".,!?;:") in POSITIVE_WORDS for word in hits)
    negative = sum(word.strip(".,!?;:") in NEGATIVE_WORDS for word in hits)
    lexical = (positive + 1) / (positive + negative + 2)
    rating_prior = min(1.0, max(0.0, float(rating) / 5))
    return round((0.65 * lexical) + (0.35 * rating_prior), 2)


def sentiment_heatmap(top_n: int = 5) -> dict:
    df = clean_data(ingest())
    df["text"] = review_text(df)
    top = (
        df.sort_values(["rating_count", "rating"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    rows = []
    for _, row in top.iterrows():
        text = row["text"]
        scores = {
            aspect: _aspect_score(text, float(row["rating"]), keywords)
            for aspect, keywords in ASPECT_KEYWORDS.items()
        }
        rows.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "short_name": str(row["product_name"])[:28],
                "rating": float(row["rating"]),
                "rating_count": int(row["rating_count"]),
                "scores": scores,
            }
        )
    return {"aspects": list(ASPECT_KEYWORDS.keys()), "rows": rows}


if __name__ == "__main__":
    print(train_sentiment_model())
