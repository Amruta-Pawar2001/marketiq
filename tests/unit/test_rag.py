from src.rag.chunker import chunk_text
from src.rag.generator import build_prompt
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
