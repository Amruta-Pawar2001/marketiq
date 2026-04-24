from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import AskRequest, AskResponse
from src.monitoring.metrics import track
from src.rag.generator import answer_question

router = APIRouter(tags=["assistant"])


@router.post("/answer_question", response_model=AskResponse)
def answer_question_route(payload: AskRequest) -> dict:
    with track("answer_question"):
        return answer_question(payload.question, top_k=payload.top_k)

