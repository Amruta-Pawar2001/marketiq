from __future__ import annotations

import pickle
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd

from src.config import EMBEDDINGS_PATH, FAISS_INDEX_PATH, META_PATH, TEXTS_PATH
from src.rag.embedder import build_index, load_embedder


@lru_cache(maxsize=1)
def load_vector_store():
    if not FAISS_INDEX_PATH.exists() or not TEXTS_PATH.exists() or not META_PATH.exists():
        build_index()
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with TEXTS_PATH.open("rb") as f:
        texts = pickle.load(f)
    with EMBEDDINGS_PATH.open("rb") as f:
        embeddings = pickle.load(f)
    meta = pd.read_pickle(META_PATH)
    embedder = load_embedder()
    return index, texts, embeddings, meta, embedder


def search(query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
    _, texts, embeddings, meta, embedder = load_vector_store()
    candidate_indices = np.arange(len(texts))
    filters = filters or {}

    if filters:
        mask = pd.Series([True] * len(meta))
        if "category" in filters:
            mask &= meta["main_category"].astype(str).str.contains(filters["category"], case=False, na=False)
        if "max_price" in filters:
            mask &= meta["actual_price"] <= filters["max_price"]
        if "min_rating" in filters:
            mask &= meta["rating"] >= filters["min_rating"]
        if "min_discount" in filters:
            mask &= meta["discount_percentage"] >= filters["min_discount"]
        filtered = np.where(mask.values)[0]
        if len(filtered) >= 1:
            candidate_indices = filtered

    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores = (embeddings[candidate_indices] @ query_vec.T).squeeze()
    scores = np.atleast_1d(scores)
    order = np.argsort(scores)[::-1][:top_k]
    global_indices = candidate_indices[order]

    results = []
    for score, idx in zip(scores[order], global_indices):
        item = meta.iloc[int(idx)].to_dict()
        results.append(
            {
                "score": float(score),
                "text": texts[int(idx)],
                "metadata": item,
            }
        )
    return results

