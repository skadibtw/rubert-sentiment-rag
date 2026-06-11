from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.api.demo_page import DEMO_PAGE_HTML
from src.inference import SentimentPredictor, SklearnSentimentPredictor, load_predictor
from src.rag import ReviewRAG, load_review_rag

DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/baseline"))
DEFAULT_RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "artifacts/rag"))


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    text: str
    label_id: int
    label: str
    scores: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    sentiment_focus: str | None = Field(default=None)
    generation_mode: str | None = Field(default=None)


class RetrievedReviewResponse(BaseModel):
    text: str
    label_id: int
    label: str
    split: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sentiment_focus: str | None
    generation_mode: str
    llm_used: bool
    keyphrases: list[str]
    label_distribution: dict[str, int]
    contexts: list[RetrievedReviewResponse]


app = FastAPI(
    title="NLP Pet Project API",
    version="0.3.0",
    description=(
        "API for Russian sentiment analysis and retrieval QA over the RuReviews "
        "dataset."
    ),
)


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor | SklearnSentimentPredictor:
    if not DEFAULT_MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {DEFAULT_MODEL_DIR}")
    return load_predictor(DEFAULT_MODEL_DIR)


@lru_cache(maxsize=1)
def get_rag_pipeline() -> ReviewRAG:
    if not DEFAULT_RAG_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"RAG index directory does not exist: {DEFAULT_RAG_INDEX_DIR}"
        )
    return load_review_rag(DEFAULT_RAG_INDEX_DIR)


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return DEMO_PAGE_HTML


@app.get("/health")
def health() -> dict[str, str | bool]:
    model_available = DEFAULT_MODEL_DIR.exists()
    rag_index_available = DEFAULT_RAG_INDEX_DIR.exists()
    return {
        "status": "ok",
        "model_dir": str(DEFAULT_MODEL_DIR),
        "model_available": model_available,
        "rag_index_dir": str(DEFAULT_RAG_INDEX_DIR),
        "rag_index_available": rag_index_available,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    prediction = predictor.predict([payload.text])[0]
    return PredictResponse(
        text=prediction.text,
        label_id=prediction.label_id,
        label=prediction.label,
        scores=prediction.scores,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        rag = get_rag_pipeline()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    response = rag.answer(
        payload.question,
        top_k=payload.top_k,
        sentiment_focus=payload.sentiment_focus,
        generation_mode=payload.generation_mode,
    )
    return AskResponse(
        question=response.question,
        answer=response.answer,
        sentiment_focus=response.sentiment_focus,
        generation_mode=response.generation_mode,
        llm_used=response.llm_used,
        keyphrases=response.keyphrases,
        label_distribution=response.label_distribution,
        contexts=[
            RetrievedReviewResponse(
                text=context.text,
                label_id=context.label_id,
                label=context.label,
                split=context.split,
                score=context.score,
            )
            for context in response.contexts
        ],
    )
