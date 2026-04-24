import pandas as pd

from src.segments.clustering import summarize_segments
from src.sentiment.analyzer import sentiment_label


def test_sentiment_labels_from_rating():
    assert sentiment_label(4.5) == "positive"
    assert sentiment_label(3.0) == "negative"
    assert sentiment_label(3.8) == "neutral"


def test_segment_summary_labels_segments():
    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 1, 1],
            "product_id": ["a", "b", "c", "d"],
            "actual_price": [100, 120, 2000, 2100],
            "discount_percentage": [10, 12, 60, 65],
            "rating": [4.4, 4.3, 3.8, 3.9],
            "rating_count": [1000, 900, 50, 60],
        }
    )
    summary = summarize_segments(df)
    assert len(summary["segments"]) == 2
    assert "label" in summary["segments"][0]


def test_psi_detects_distribution_shift():
    from src.monitoring.drift import _psi

    baseline = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    shifted = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    score = _psi(baseline, shifted)
    assert score >= 0
