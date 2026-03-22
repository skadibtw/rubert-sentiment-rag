from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import SentimentPredictor, load_predictor


DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/bert_custom"))


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    text: str
    label_id: int
    label: str
    scores: dict[str, float]


app = FastAPI(title="NLP Pet Project API", version="0.1.0")


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    if not DEFAULT_MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {DEFAULT_MODEL_DIR}")
    return load_predictor(DEFAULT_MODEL_DIR)


@app.get("/health")
def health() -> dict[str, str | bool]:
    model_available = DEFAULT_MODEL_DIR.exists()
    return {
        "status": "ok",
        "model_dir": str(DEFAULT_MODEL_DIR),
        "model_available": model_available,
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
