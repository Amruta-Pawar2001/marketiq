from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import CHUNKS_PATH, ensure_dirs
from src.pipeline.clean import clean_data
from src.pipeline.ingest import ingest


def build_product_text(row: pd.Series) -> str:
    parts = [
        f"Product: {row.get('product_name', '')}",
        f"Category: {row.get('category', '')}",
        f"Actual price: {row.get('actual_price', '')}",
        f"Discount: {row.get('discount_percentage', '')}%",
        f"Rating: {row.get('rating', '')}",
        f"Description: {row.get('about_product', '')}",
        f"Review title: {row.get('review_title', '')}",
        f"Review content: {row.get('review_content', '')}",
    ]
    return " | ".join([p for p in parts if p.strip()])


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = " ".join(str(text).split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(output_path: str | Path = CHUNKS_PATH) -> list[dict]:
    ensure_dirs()
    df = clean_data(ingest())
    records: list[dict] = []
    for _, row in df.iterrows():
        full_text = build_product_text(row)
        for idx, chunk in enumerate(chunk_text(full_text)):
            records.append(
                {
                    "id": f"{row.get('product_id', 'unknown')}-{idx}",
                    "text": chunk,
                    "metadata": {
                        "product_id": row.get("product_id", ""),
                        "product_name": row.get("product_name", ""),
                        "main_category": row.get("main_category", ""),
                        "actual_price": float(row.get("actual_price", 0) or 0),
                        "discount_percentage": float(row.get("discount_percentage", 0) or 0),
                        "rating": float(row.get("rating", 0) or 0),
                    },
                }
            )
    with Path(output_path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def load_chunks(path: str | Path = CHUNKS_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return build_chunks(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

