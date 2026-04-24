from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.routes import answer, health, insights, predict
from src.config import FRONTEND_DIR

app = FastAPI(
    title="MarketIQ Marketing Data Intelligence Platform",
    version="1.0.0",
    description="XGBoost discount prediction and OpenAI-powered RAG over Amazon product data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(answer.router)
app.include_router(insights.router)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "marketing_intelligence_ui.html")
