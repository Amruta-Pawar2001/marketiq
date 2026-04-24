from __future__ import annotations

import json

import pandas as pd

from src.config import DRIFT_REPORT_PATH, PROCESSED_DIR, ensure_dirs
from src.pipeline.clean import clean_data
from src.pipeline.features import add_features
from src.pipeline.ingest import ingest
from src.pipeline.split import build_processed_splits

DRIFT_COLUMNS = ["actual_price", "discount_percentage", "rating", "rating_count", "price_ratio"]


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    _, edges = pd.qcut(expected.rank(method="first"), q=bins, retbins=True, duplicates="drop")
    expected_bins = pd.cut(expected.rank(method="first"), bins=edges, include_lowest=True).value_counts(normalize=True)
    actual_bins = pd.cut(actual.rank(method="first"), bins=edges, include_lowest=True).value_counts(normalize=True)
    all_bins = expected_bins.index.union(actual_bins.index)
    e = expected_bins.reindex(all_bins, fill_value=0.001)
    a = actual_bins.reindex(all_bins, fill_value=0.001)
    return float(((a - e) * ((a / e).map(lambda x: 0 if x <= 0 else __import__("math").log(x)))).sum())


def drift_report(threshold: float = 0.15) -> dict:
    ensure_dirs()
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        build_processed_splits()
    baseline = pd.read_csv(train_path)
    current = add_features(clean_data(ingest()))
    feature_scores = {
        col: round(_psi(baseline[col], current[col]), 4)
        for col in DRIFT_COLUMNS
        if col in baseline.columns and col in current.columns
    }
    overall = max(feature_scores.values(), default=0.0)
    report = {
        "threshold": threshold,
        "feature_scores": feature_scores,
        "overall_drift_score": overall,
        "drift_detected": overall >= threshold,
        "recommendation": "retrain" if overall >= threshold else "monitor",
    }
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(drift_report(), indent=2))

