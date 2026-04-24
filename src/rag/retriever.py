from __future__ import annotations

import re

from src.rag.vector_store import load_vector_store, search
from src.config import EMBEDDING_MODEL, OPENAI_MODEL


def parse_filters(question: str) -> dict:
    q = question.lower()
    filters: dict[str, float | str] = {}
    _, _, _, meta, _ = load_vector_store()
    for cat in sorted(meta["main_category"].dropna().astype(str).unique(), key=len, reverse=True):
        if cat and cat.lower() in q:
            filters["category"] = cat.lower()
            break

    price_match = re.search(r"(?:under|below|less than|within|max|upto|up to)\s*(?:rs|inr|rupees)?\s*(\d[\d,]*)", q)
    if price_match:
        filters["max_price"] = float(price_match.group(1).replace(",", ""))

    rating_match = re.search(r"(?:above|at least|minimum|over|more than)\s*(\d(?:\.\d)?)\s*(?:star|rating)?", q)
    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))
    elif any(term in q for term in ["highly rated", "top rated", "best rated"]):
        filters["min_rating"] = 4.0

    discount_match = re.search(r"(\d+)\s*%\s*(?:off|discount)", q)
    if discount_match:
        filters["min_discount"] = float(discount_match.group(1))
    elif any(term in q for term in ["high discount", "big discount", "best deal", "on sale"]):
        filters["min_discount"] = 30.0

    return filters


def retrieve(question: str, top_k: int = 5) -> tuple[list[dict], dict]:
    filters = parse_filters(question)
    return search(question, top_k=top_k, filters=filters), filters


def trace_question(question: str, top_k: int = 5) -> dict:
    contexts, filters = retrieve(question, top_k=top_k)
    retrieved = []
    for rank, ctx in enumerate(contexts, start=1):
        meta = ctx.get("metadata", {})
        retrieved.append(
            {
                "rank": rank,
                "score": round(float(ctx.get("score", 0.0)), 4),
                "product_id": meta.get("product_id", ""),
                "product_name": meta.get("product_name", ""),
                "category": meta.get("main_category", ""),
                "rating": meta.get("rating", None),
                "discount_percentage": meta.get("discount_percentage", None),
                "snippet": str(ctx.get("text", ""))[:320],
            }
        )
    return {
        "query": question,
        "filters_used": filters,
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": "FAISS IndexFlatIP over normalized MiniLM embeddings",
        "generation_model": OPENAI_MODEL,
        "top_k": top_k,
        "retrieved_count": len(retrieved),
        "min_similarity": min([item["score"] for item in retrieved], default=0.0),
        "retrieved": retrieved,
    }
