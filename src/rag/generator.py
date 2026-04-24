from __future__ import annotations

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL, RAG_TOP_K
from src.rag.retriever import retrieve


def build_prompt(question: str, contexts: list[dict], filters: dict) -> list[dict]:
    context_text = "\n\n".join(
        f"[Source {idx + 1} | {ctx['metadata'].get('product_id', '')}]\n{ctx['text']}"
        for idx, ctx in enumerate(contexts)
    )
    system = (
        "You are MarketIQ, a marketing data intelligence assistant. "
        "Answer only from the provided Amazon product and review context. "
        "If the context is insufficient, say you do not have enough information. "
        "Be concise, analytical, and cite source numbers when useful."
    )
    user = (
        f"Filters applied: {filters or 'none'}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"User question: {question}\n\n"
        "Ground the answer in the retrieved context."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_question(question: str, top_k: int = RAG_TOP_K) -> dict:
    contexts, filters = retrieve(question, top_k=top_k)
    if not OPENAI_API_KEY:
        return {
            "answer": "OPENAI_API_KEY is not configured. Retrieved sources are returned for review.",
            "sources": [ctx["metadata"] for ctx in contexts],
            "contexts": contexts,
            "filters_used": filters,
            "factuality_score": 0.0,
        }

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=build_prompt(question, contexts, filters),
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer,
        "sources": [ctx["metadata"] for ctx in contexts],
        "contexts": contexts,
        "filters_used": filters,
        "factuality_score": round(max([ctx["score"] for ctx in contexts], default=0.0), 3),
    }
