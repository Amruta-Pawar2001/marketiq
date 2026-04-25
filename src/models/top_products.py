from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.demand import demand_score
from src.models.predict import load_model
from src.models.train import band_name_from_prediction, representative_discounts_from_predictions
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
    raw_predictions = model.predict(df[FEATURE_COLUMNS])
    bands = [band_name_from_prediction(label) for label in raw_predictions]
    preds = representative_discounts_from_predictions(raw_predictions)
    df["predicted_discount"] = preds
    df["predicted_discount_band"] = bands
    df["discount_gap"] = df["discount_percentage"] - df["predicted_discount"]
    df["review_text"] = review_text(df)
    df["market_traction_score"] = demand_score(df)

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
                "predicted_discount_band": row["predicted_discount_band"],
                "gap": round(float(row["discount_gap"]), 2),
                "sentiment_score": round(float(sentiment["sentiment_score"]), 3),
                "sentiment_label": sentiment["label"],
                "market_traction_score": round(float(row["market_traction_score"]), 2),
                "market_traction_label": _traction_label(float(row["market_traction_score"]), df["market_traction_score"]),
                "flag": label,
                "flag_color": color,
            }
        )
    return {"rows": rows}


def _traction_label(score: float, all_scores: pd.Series) -> str:
    low, high = all_scores.quantile([0.33, 0.67])
    if score >= high:
        return "High"
    if score <= low:
        return "Low"
    return "Med"
