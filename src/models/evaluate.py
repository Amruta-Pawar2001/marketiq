from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import METRICS_PATH, MODEL_PATH, PROCESSED_DIR
from src.models.train import train_model
from src.pipeline.features import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate_model() -> dict[str, float]:
    if not MODEL_PATH.exists():
        train_model()
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    preds = np.clip(model.predict(test_df[FEATURE_COLUMNS]), 0, 80)
    y_test = test_df[TARGET_COLUMN]
    metrics = {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate_model(), indent=2))

