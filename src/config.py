from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "amazon.csv"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

MODEL_PATH = MODELS_DIR / "discount_model.joblib"
METRICS_PATH = MODELS_DIR / "discount_metrics.json"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
SEGMENT_MODEL_PATH = MODELS_DIR / "segment_model.joblib"
SEGMENT_SUMMARY_PATH = MODELS_DIR / "segment_summary.json"
DEMAND_MODEL_PATH = MODELS_DIR / "demand_model.joblib"
DEMAND_METRICS_PATH = MODELS_DIR / "demand_metrics.json"
DRIFT_REPORT_PATH = MODELS_DIR / "drift_report.json"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "product_index.faiss"
TEXTS_PATH = EMBEDDINGS_DIR / "product_texts.pkl"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "product_embeddings.pkl"
META_PATH = EMBEDDINGS_DIR / "product_meta.pkl"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.jsonl"


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
