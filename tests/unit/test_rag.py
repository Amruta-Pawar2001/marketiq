from src.rag.chunker import chunk_text
from src.rag.generator import (
    _contextualize_question,
    _out_of_scope_answer,
    _small_talk_answer,
    _structured_answer,
    _worst_or_best_product_answer,
    build_prompt,
)
from src.rag.retriever import trace_question


def test_chunk_text_splits_long_content():
    chunks = chunk_text("a" * 1200, chunk_size=500, overlap=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_prompt_instructs_grounded_answering():
    messages = build_prompt(
        "Which cable is good?",
        [{"text": "Product: Cable | Rating: 4.2", "metadata": {"product_id": "p1"}, "score": 0.9}],
        {},
    )
    assert "Answer only from the provided" in messages[0]["content"]
    assert "Product: Cable" in messages[1]["content"]


def test_trace_question_shape(monkeypatch):
    monkeypatch.setattr(
        "src.rag.retriever.retrieve",
        lambda question, top_k=5: (
            [
                {
                    "score": 0.91,
                    "text": "Product: Cable | Rating: 4.2",
                    "metadata": {"product_id": "p1", "product_name": "Cable", "main_category": "electronics"},
                }
            ],
            {"max_price": 500.0},
        ),
    )
    trace = trace_question("USB cable under 500")
    assert trace["filters_used"]["max_price"] == 500.0
    assert trace["retrieved"][0]["score"] == 0.91


def test_worst_product_answer_uses_product_name(monkeypatch):
    import pandas as pd

    from src.rag import generator

    df = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_name": ["Good Cable", "Bad Speaker"],
            "category": ["electronics", "electronics"],
            "actual_price": [100, 200],
            "discounted_price": [80, 150],
            "discount_percentage": [20, 25],
            "rating": [4.5, 2.5],
            "rating_count": [10, 99],
            "about_product": ["", ""],
            "review_title": ["", ""],
            "review_content": ["", ""],
        }
    )
    monkeypatch.setattr(generator, "ingest", lambda: df)
    result = _worst_or_best_product_answer("What is our worst-rated product?")
    assert "Bad Speaker" in result["answer"]
    assert result["sources"][0]["product_name"] == "Bad Speaker"


def test_recommendation_answer_ranks_category_products(monkeypatch):
    import pandas as pd

    from src.rag import generator

    df = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["Weak Cable", "Trusted Headphones", "Kitchen Bowl"],
            "category": ["electronics", "electronics", "home&kitchen"],
            "actual_price": [200, 999, 300],
            "discounted_price": [180, 699, 250],
            "discount_percentage": [10, 30, 15],
            "rating": [3.2, 4.7, 4.9],
            "rating_count": [5, 5000, 200],
            "about_product": ["bad issue", "excellent quality and great value", "good"],
            "review_title": ["", "", ""],
            "review_content": ["", "", ""],
        }
    )
    monkeypatch.setattr(generator, "ingest", lambda: df)
    result = _structured_answer("Which product should I buy in electronics now?")
    assert "Trusted Headphones" in result["answer"]
    assert result["filters_used"]["intent"] == "recommendation"
    assert result["sources"][0]["main_category"] == "electronics"


def test_category_analytics_answer_uses_aggregates(monkeypatch):
    import pandas as pd

    from src.rag import generator

    df = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["A", "B", "C"],
            "category": ["electronics", "electronics", "books"],
            "actual_price": [100, 200, 300],
            "discounted_price": [80, 160, 270],
            "discount_percentage": [20, 20, 10],
            "rating": [4.0, 4.5, 3.5],
            "rating_count": [10, 20, 30],
            "about_product": ["", "", ""],
            "review_title": ["", "", ""],
            "review_content": ["", "", ""],
        }
    )
    monkeypatch.setattr(generator, "ingest", lambda: df)
    result = _structured_answer("Which category has highest discount?")
    assert "electronics" in result["answer"]
    assert result["filters_used"]["intent"] == "category_analytics"


def test_contextualize_question_uses_recent_user_history():
    history = [
        {"role": "user", "content": "Which product should I buy in electronics?"},
        {"role": "assistant", "content": "I recommend a cable."},
    ]
    contextual = _contextualize_question("What about under 1000?", history)
    assert "electronics" in contextual
    assert "Follow-up: What about under 1000?" in contextual


def test_small_talk_does_not_trigger_product_memory():
    result = _small_talk_answer("Okay thanks")
    assert result["filters_used"]["intent"] == "small_talk"
    assert "welcome" in result["answer"].lower()
    assert result["sources"] == []


def test_out_of_scope_refuses_unrelated_questions():
    result = _out_of_scope_answer("Who won the football match yesterday?")
    assert result["filters_used"]["intent"] == "out_of_scope"
    assert "MarketIQ Amazon product dataset" in result["answer"]


def test_out_of_scope_allows_market_followup_after_memory():
    history = [{"role": "user", "content": "Which product should I buy in electronics?"}]
    contextual = _contextualize_question("What about under 1000?", history)
    assert _out_of_scope_answer(contextual) is None
