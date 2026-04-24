from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.predict import load_model
from src.pipeline.clean import clean_data
from src.pipeline.features import FEATURE_COLUMNS, add_features
from src.pipeline.ingest import ingest
from src.sentiment.analyzer import analyze_text, review_text


def _flag(gap: float, sentiment_score: float) -> tuple[str, str]:
    if abs(gap) <= 5 and sentiment_score >= 0:
        return "Good", "green"
    if abs(gap) >= 15 or sentiment_score < -0.2:
        return "Review", "red"
    return "Watch", "yellow"


def top_products(limit: int = 5) -> dict:
    df = add_features(clean_data(ingest()))
    model = load_model()
    preds = np.clip(model.predict(df[FEATURE_COLUMNS]), 0, 80)
    df["predicted_discount"] = preds
    df["discount_gap"] = df["discount_percentage"] - df["predicted_discount"]
    df["review_text"] = review_text(df)

    candidates = (
        df.assign(priority=df["discount_gap"].abs() * np.log1p(df["rating_count"]))
        .sort_values("priority", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )

    rows = []
    for _, row in candidates.iterrows():
        sentiment = analyze_text(row["review_text"])
        label, color = _flag(float(row["discount_gap"]), float(sentiment["sentiment_score"]))
        rows.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "actual_discount": round(float(row["discount_percentage"]), 2),
                "predicted_discount": round(float(row["predicted_discount"]), 2),
                "gap": round(float(row["discount_gap"]), 2),
                "sentiment_score": round(float(sentiment["sentiment_score"]), 3),
                "sentiment_label": sentiment["label"],
                "flag": label,
                "flag_color": color,
            }
        )
    return {"rows": rows}
