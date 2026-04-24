from __future__ import annotations

from src.models.predict import prepare_input


def explain_prediction(payload: dict) -> dict[str, float | str]:
    """Lightweight explanation fallback for API responses and tests."""
    row = prepare_input(payload).iloc[0]
    return {
        "main_category": str(row["main_category"]),
        "price_ratio": round(float(row["price_ratio"]), 4),
        "rating": round(float(row["rating"]), 2),
        "review_count_log": round(float(row["review_count_log"]), 4),
        "high_rated": int(row["high_rated"]),
    }
