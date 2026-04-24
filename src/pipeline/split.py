from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, ensure_dirs
from src.pipeline.clean import clean_data
from src.pipeline.features import add_features
from src.pipeline.ingest import ingest


def build_processed_splits(test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    df = add_features(clean_data(ingest()))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)
    return train_df, test_df


if __name__ == "__main__":
    train, test = build_processed_splits()
    print(f"Saved train={len(train)} rows, test={len(test)} rows")

