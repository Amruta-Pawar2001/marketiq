from __future__ import annotations

import math
import re

import pandas as pd
from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL, RAG_TOP_K
from src.pipeline.clean import clean_data
from src.pipeline.ingest import ingest
from src.rag.retriever import retrieve


def _product_label(ctx: dict) -> str:
    metadata = ctx.get("metadata", {})
    name = metadata.get("product_name") or "Unknown product"
    product_id = metadata.get("product_id") or "unknown-id"
    return f"{name} ({product_id})"


def _load_catalog() -> pd.DataFrame:
    return clean_data(ingest())


def _source_from_row(row: pd.Series, **extra: object) -> dict:
    source = {
        "product_id": row.get("product_id", ""),
        "product_name": row.get("product_name", ""),
        "main_category": row.get("main_category", ""),
        "actual_price": float(row.get("actual_price", 0) or 0),
        "discounted_price": float(row.get("discounted_price", 0) or 0),
        "discount_percentage": float(row.get("discount_percentage", 0) or 0),
        "rating": float(row.get("rating", 0) or 0),
        "rating_count": int(row.get("rating_count", 0) or 0),
    }
    source.update(extra)
    return source


def _extract_category(question: str, df: pd.DataFrame) -> str | None:
    q = question.lower()
    categories = sorted(df["main_category"].dropna().astype(str).unique(), key=len, reverse=True)
    for category in categories:
        if category and category.lower() in q:
            return category
    return None


def _extract_max_price(question: str) -> float | None:
    q = question.lower()
    match = re.search(r"(?:under|below|less than|within|max|upto|up to)\s*(?:rs|inr|rupees|₹)?\s*(\d[\d,]*)", q)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _recent_history_text(history: list[dict[str, str]], limit: int = 6) -> str:
    turns = []
    for turn in history[-limit:]:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            turns.append(f"{role}: {content}")
    return "\n".join(turns)


def _contextualize_question(question: str, history: list[dict[str, str]] | None = None) -> str:
    history = history or []
    if not history:
        return question

    q = question.strip()
    q_lower = q.lower()
    looks_like_followup = (
        len(q.split()) <= 8
        or q_lower.startswith(("what about", "how about", "and ", "also ", "under ", "below "))
        or any(term in q_lower for term in ["this", "that", "those", "same", "it", "them"])
    )
    if not looks_like_followup:
        return question

    prior_user_questions = [
        str(turn.get("content", "")).strip()
        for turn in history
        if str(turn.get("role", "")).lower() == "user" and str(turn.get("content", "")).strip()
    ]
    if not prior_user_questions:
        return question
    context = " ".join(prior_user_questions[-2:])
    return f"{context}. Follow-up: {question}"


def _small_talk_answer(question: str) -> dict | None:
    q = " ".join(question.lower().strip().split())
    thanks_terms = ["thanks", "thank you", "okay thanks", "ok thanks", "thx", "got it", "sounds good"]
    greeting_terms = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    bye_terms = ["bye", "goodbye", "see you", "see ya"]

    if q in thanks_terms or any(q.startswith(f"{term} ") for term in thanks_terms):
        answer = "You're welcome. Ask me anytime about product recommendations, discounts, reviews, or category insights."
    elif q in greeting_terms:
        answer = "Hi, I'm ready. You can ask me about product recommendations, best deals, weak products, or review insights."
    elif q in bye_terms:
        answer = "Goodbye. Your current dashboard session will keep the visible chat until you refresh the page."
    else:
        return None

    return {
        "answer": answer,
        "sources": [],
        "contexts": [],
        "filters_used": {"intent": "small_talk"},
        "factuality_score": 1.0,
    }


def _out_of_scope_answer(question: str) -> dict | None:
    q = question.lower()
    market_terms = {
        "amazon",
        "product",
        "products",
        "price",
        "pricing",
        "discount",
        "deal",
        "rating",
        "ratings",
        "review",
        "reviews",
        "sentiment",
        "category",
        "categories",
        "electronics",
        "home",
        "kitchen",
        "computer",
        "computers",
        "cable",
        "usb",
        "buy",
        "recommend",
        "recommendation",
        "compare",
        "comparison",
        "best",
        "worst",
        "highest",
        "lowest",
        "under",
        "above",
        "market",
        "sales",
        "demand",
        "traction",
        "segment",
        "segments",
    }
    if any(term in q for term in market_terms):
        return None

    return {
        "answer": (
            "I can only answer questions about the MarketIQ Amazon product dataset, "
            "such as recommendations, discounts, ratings, reviews, categories, and product comparisons."
        ),
        "sources": [],
        "contexts": [],
        "filters_used": {"intent": "out_of_scope"},
        "factuality_score": 1.0,
    }


def _sentiment_score(row: pd.Series) -> float:
    text = " ".join(
        [
            str(row.get("about_product", "")),
            str(row.get("review_title", "")),
            str(row.get("review_content", "")),
        ]
    ).lower()
    positive_terms = ["good", "great", "best", "excellent", "fast", "easy", "quality", "value", "perfect", "nice"]
    negative_terms = ["bad", "poor", "worst", "broken", "slow", "issue", "problem", "waste", "return", "defect"]
    positive = sum(text.count(term) for term in positive_terms)
    negative = sum(text.count(term) for term in negative_terms)
    if positive + negative == 0:
        return float(row.get("rating", 0) or 0) / 5
    return max(0.0, min(1.0, (positive + 1) / (positive + negative + 2)))


def _with_recommendation_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    max_rating_count = max(float(scored["rating_count"].max() or 0), 1.0)
    scored["rating_signal"] = scored["rating"].clip(0, 5) / 5
    scored["review_signal"] = scored["rating_count"].apply(lambda x: math.log1p(max(float(x), 0)) / math.log1p(max_rating_count))
    scored["discount_signal"] = scored["discount_percentage"].clip(0, 80) / 80
    scored["sentiment_signal"] = scored.apply(_sentiment_score, axis=1)
    scored["recommendation_score"] = (
        0.45 * scored["rating_signal"]
        + 0.30 * scored["review_signal"]
        + 0.15 * scored["discount_signal"]
        + 0.10 * scored["sentiment_signal"]
    )
    return scored


def _worst_or_best_product_answer(question: str, df: pd.DataFrame | None = None) -> dict | None:
    q = question.lower()
    if not any(term in q for term in ["worst", "lowest rated", "lowest-rated", "best rated", "top rated", "highest rated"]):
        return None

    df = _load_catalog() if df is None else df
    category = _extract_category(question, df)
    if category:
        df = df[df["main_category"] == category]
    if df.empty:
        return None

    if "worst" in q or "lowest" in q:
        ranked = df.sort_values(["rating", "rating_count"], ascending=[True, False])
        direction = "worst-rated"
    else:
        ranked = df.sort_values(["rating", "rating_count"], ascending=[False, False])
        direction = "best-rated"

    row = ranked.iloc[0]
    answer = (
        f"The {direction} product in the dataset is {row['product_name']} "
        f"with a rating of {float(row['rating']):.1f} from {int(row['rating_count'])} ratings. "
        f"It belongs to {row['main_category']} and has a listed discount of "
        f"{float(row['discount_percentage']):.0f}%."
    )
    source = _source_from_row(row)
    return {
        "answer": answer,
        "sources": [source],
        "contexts": [],
        "filters_used": {"intent": "product_ranking", "ranking": direction, "category": category, "sort": "rating_then_rating_count"},
        "factuality_score": 1.0,
    }


def _deal_answer(question: str, df: pd.DataFrame) -> dict | None:
    q = question.lower()
    if not any(term in q for term in ["highest discount", "biggest discount", "best deal", "most discount", "maximum discount"]):
        return None

    category = _extract_category(question, df)
    max_price = _extract_max_price(question)
    filtered = df.copy()
    if category:
        filtered = filtered[filtered["main_category"] == category]
    if max_price is not None:
        filtered = filtered[filtered["actual_price"] <= max_price]
    if filtered.empty:
        return {
            "answer": "I could not find matching products for that deal filter in the dataset.",
            "sources": [],
            "contexts": [],
            "filters_used": {"intent": "deal_ranking", "category": category, "max_price": max_price},
            "factuality_score": 0.3,
        }

    ranked = filtered.sort_values(["discount_percentage", "rating", "rating_count"], ascending=[False, False, False]).head(3)
    best = ranked.iloc[0]
    answer = (
        f"The strongest deal I found is {best['product_name']} with "
        f"{float(best['discount_percentage']):.0f}% off, a rating of {float(best['rating']):.1f}, "
        f"and {int(best['rating_count'])} ratings. "
        f"I also checked rating and review count so the answer is not based on discount alone."
    )
    return {
        "answer": answer,
        "sources": [_source_from_row(row, rank=idx + 1) for idx, (_, row) in enumerate(ranked.iterrows())],
        "contexts": [],
        "filters_used": {"intent": "deal_ranking", "category": category, "max_price": max_price, "sort": "discount_then_rating"},
        "factuality_score": 0.95,
    }


def _recommendation_answer(question: str, df: pd.DataFrame) -> dict | None:
    q = question.lower()
    if not any(term in q for term in ["should i buy", "recommend", "suggest", "best product", "what should i buy", "which product should i buy"]):
        return None

    category = _extract_category(question, df)
    max_price = _extract_max_price(question)
    filtered = df.copy()
    if category:
        filtered = filtered[filtered["main_category"] == category]
    if max_price is not None:
        filtered = filtered[filtered["actual_price"] <= max_price]
    if filtered.empty:
        return {
            "answer": "I could not find products matching that recommendation filter in the dataset.",
            "sources": [],
            "contexts": [],
            "filters_used": {"intent": "recommendation", "category": category, "max_price": max_price},
            "factuality_score": 0.3,
        }

    scored = _with_recommendation_score(filtered)
    ranked = scored.sort_values("recommendation_score", ascending=False).head(3)
    best = ranked.iloc[0]
    category_text = f" in {category}" if category else ""
    answer = (
        f"I would recommend {best['product_name']}{category_text}. "
        f"It has a {float(best['rating']):.1f} rating from {int(best['rating_count'])} ratings, "
        f"{float(best['discount_percentage']):.0f}% discount, and a recommendation score of "
        f"{float(best['recommendation_score']) * 100:.0f}/100. "
        "The score combines rating quality, review-count confidence, discount value, and review sentiment signals."
    )
    return {
        "answer": answer,
        "sources": [
            _source_from_row(row, rank=idx + 1, recommendation_score=round(float(row["recommendation_score"]), 3))
            for idx, (_, row) in enumerate(ranked.iterrows())
        ],
        "contexts": [],
        "filters_used": {
            "intent": "recommendation",
            "category": category,
            "max_price": max_price,
            "score": "0.45*rating + 0.30*review_count + 0.15*discount + 0.10*sentiment",
        },
        "factuality_score": 0.95,
    }


def _category_analytics_answer(question: str, df: pd.DataFrame) -> dict | None:
    q = question.lower()
    if "category" not in q or not any(term in q for term in ["highest discount", "lowest discount", "best rated", "worst rated", "average discount"]):
        return None

    grouped = (
        df.groupby("main_category")
        .agg(
            avg_discount=("discount_percentage", "mean"),
            avg_rating=("rating", "mean"),
            product_count=("product_id", "count"),
            total_ratings=("rating_count", "sum"),
        )
        .reset_index()
    )
    if "lowest discount" in q:
        row = grouped.sort_values("avg_discount", ascending=True).iloc[0]
        metric = "lowest average discount"
        value = f"{float(row['avg_discount']):.1f}%"
    elif "best rated" in q or "highest rated" in q:
        row = grouped.sort_values("avg_rating", ascending=False).iloc[0]
        metric = "highest average rating"
        value = f"{float(row['avg_rating']):.2f}"
    elif "worst rated" in q:
        row = grouped.sort_values("avg_rating", ascending=True).iloc[0]
        metric = "lowest average rating"
        value = f"{float(row['avg_rating']):.2f}"
    else:
        row = grouped.sort_values("avg_discount", ascending=False).iloc[0]
        metric = "highest average discount"
        value = f"{float(row['avg_discount']):.1f}%"

    answer = (
        f"The category with the {metric} is {row['main_category']} at {value}. "
        f"This is calculated across {int(row['product_count'])} products in the cleaned Amazon dataset."
    )
    return {
        "answer": answer,
        "sources": [row.to_dict()],
        "contexts": [],
        "filters_used": {"intent": "category_analytics", "metric": metric},
        "factuality_score": 0.95,
    }


def _structured_answer(question: str, top_k: int = RAG_TOP_K) -> dict | None:
    df = _load_catalog()
    for handler in (
        _worst_or_best_product_answer,
        _category_analytics_answer,
        _deal_answer,
        _recommendation_answer,
    ):
        result = handler(question, df)
        if result is not None:
            return result
    return None


def build_prompt(question: str, contexts: list[dict], filters: dict, history: list[dict[str, str]] | None = None) -> list[dict]:
    context_text = "\n\n".join(
        f"[Source {idx + 1} | {_product_label(ctx)}]\n{ctx['text']}"
        for idx, ctx in enumerate(contexts)
    )
    system = (
        "You are MarketIQ, a marketing data intelligence assistant. "
        "Answer only from the provided Amazon product and review context. "
        "When naming a product, use the product_name field, not only the product_id. "
        "Use the conversation history only to understand follow-up references; do not invent facts from it. "
        "If the context is insufficient, say you do not have enough information. "
        "Be concise, analytical, and cite source numbers when useful."
    )
    history_text = _recent_history_text(history or [])
    user = (
        f"Conversation history:\n{history_text or 'none'}\n\n"
        f"Filters applied: {filters or 'none'}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"User question: {question}\n\n"
        "Ground the answer in the retrieved context."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_question(question: str, top_k: int = RAG_TOP_K, history: list[dict[str, str]] | None = None) -> dict:
    small_talk = _small_talk_answer(question)
    if small_talk is not None:
        return small_talk

    contextual_question = _contextualize_question(question, history)
    out_of_scope = _out_of_scope_answer(contextual_question)
    if out_of_scope is not None:
        return out_of_scope

    structured = _structured_answer(contextual_question, top_k=top_k)
    if structured is not None:
        structured["filters_used"] = {
            **structured.get("filters_used", {}),
            "memory_used": contextual_question != question,
            "original_question": question,
        }
        return structured

    contexts, filters = retrieve(contextual_question, top_k=top_k)
    if not OPENAI_API_KEY:
        return {
            "answer": "OPENAI_API_KEY is not configured. Retrieved sources are returned for review.",
            "sources": [ctx["metadata"] for ctx in contexts],
            "contexts": contexts,
            "filters_used": {**filters, "memory_used": contextual_question != question, "original_question": question},
            "factuality_score": 0.0,
        }

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=build_prompt(contextual_question, contexts, filters, history=history),
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer,
        "sources": [ctx["metadata"] for ctx in contexts],
        "contexts": contexts,
        "filters_used": {**filters, "memory_used": contextual_question != question, "original_question": question},
        "factuality_score": round(max([ctx["score"] for ctx in contexts], default=0.0), 3),
    }
