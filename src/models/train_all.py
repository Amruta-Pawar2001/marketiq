from __future__ import annotations

import json

from src.models.demand import train_demand_model
from src.models.train import train_model
from src.monitoring.drift import drift_report
from src.segments.clustering import train_segments
from src.sentiment.analyzer import train_sentiment_model


def train_all() -> dict:
    return {
        "discount": train_model(),
        "sentiment": train_sentiment_model(),
        "segments": train_segments(),
        "demand": train_demand_model(),
        "drift": drift_report(),
    }


if __name__ == "__main__":
    print(json.dumps(train_all(), indent=2))
