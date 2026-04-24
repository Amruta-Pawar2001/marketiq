from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH


def parse_numeric(value: object) -> float:
    """Parse currency, percentage, counts, and mojibake rupee strings."""
    if pd.isna(value):
        return float("nan")
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in {"", ".", "-", "-."}:
        return float("nan")
    return float(cleaned)


def load_raw_csv(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def ingest(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = load_raw_csv(path).copy()
    df["actual_price"] = df["actual_price"].map(parse_numeric)
    df["discounted_price"] = df["discounted_price"].map(parse_numeric)
    df["discount_percentage"] = df["discount_percentage"].map(parse_numeric)
    df["rating"] = df["rating"].map(parse_numeric)
    df["rating_count"] = df["rating_count"].map(parse_numeric)
    return df


if __name__ == "__main__":
    data = ingest()
    print(data.info())
    print(data.head(3))

