from __future__ import annotations

import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    META_PATH,
    TEXTS_PATH,
    ensure_dirs,
)
from src.rag.chunker import load_chunks


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def build_index() -> dict[str, int]:
    ensure_dirs()
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]

    embedder = load_embedder()
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with TEXTS_PATH.open("wb") as f:
        pickle.dump(texts, f)
    with EMBEDDINGS_PATH.open("wb") as f:
        pickle.dump(embeddings, f)
    pd.DataFrame(metadata).to_pickle(META_PATH)
    return {"chunks": len(texts), "dimension": int(embeddings.shape[1])}


if __name__ == "__main__":
    print(build_index())

