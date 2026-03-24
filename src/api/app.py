from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.inference import SentimentPredictor, load_predictor
from src.rag import ReviewRAG, load_review_rag

DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/bert_custom"))
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
def get_predictor() -> SentimentPredictor:
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
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NLP Pet Project Demo</title>
    <style>
      :root {
        --bg: #f4efe6;
        --panel: rgba(255, 250, 242, 0.9);
        --ink: #1f2933;
        --muted: #52606d;
        --accent: #ad4f2d;
        --accent-soft: #f6d6c5;
        --border: rgba(31, 41, 51, 0.12);
        --shadow: 0 20px 60px rgba(75, 85, 99, 0.15);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(173, 79, 45, 0.18), transparent 34%),
          radial-gradient(circle at bottom right, rgba(53, 122, 93, 0.16), transparent 30%),
          linear-gradient(135deg, #efe7da 0%, #f8f4ec 45%, #efe3d6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
      }

      .shell {
        width: min(960px, 100%);
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 28px;
        box-shadow: var(--shadow);
        overflow: hidden;
        backdrop-filter: blur(16px);
      }

      .hero {
        padding: 32px 32px 12px;
      }

      .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: var(--accent);
        margin: 0 0 12px;
      }

      h1 {
        margin: 0;
        font-size: clamp(32px, 5vw, 56px);
        line-height: 0.96;
        max-width: 10ch;
      }

      .subtitle {
        margin: 16px 0 0;
        max-width: 58ch;
        font-size: 18px;
        line-height: 1.6;
        color: var(--muted);
      }

      .content {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 20px;
        padding: 20px 32px 32px;
      }

      .card {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 22px;
      }

      label {
        display: block;
        font-size: 14px;
        margin-bottom: 10px;
        color: var(--muted);
      }

      textarea {
        width: 100%;
        min-height: 210px;
        resize: vertical;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        font: inherit;
        font-size: 16px;
        background: rgba(255, 255, 255, 0.95);
        color: var(--ink);
      }

      button {
        margin-top: 16px;
        border: 0;
        border-radius: 999px;
        padding: 14px 20px;
        font: inherit;
        font-size: 15px;
        cursor: pointer;
        color: #fff;
        background: linear-gradient(135deg, #ad4f2d 0%, #8c3c1d 100%);
        box-shadow: 0 12px 24px rgba(173, 79, 45, 0.24);
      }

      .hint,
      .metric {
        font-size: 14px;
        color: var(--muted);
      }

      .result-label {
        margin: 0;
        font-size: 34px;
      }

      .scores {
        display: grid;
        gap: 12px;
        margin-top: 18px;
      }

      .score-row {
        display: grid;
        grid-template-columns: 92px 1fr auto;
        gap: 12px;
        align-items: center;
      }

      .bar {
        height: 10px;
        border-radius: 999px;
        background: #eee4d8;
        overflow: hidden;
      }

      .bar > span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, var(--accent-soft), var(--accent));
      }

      .meta {
        display: grid;
        gap: 10px;
        margin-top: 22px;
      }

      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 999px;
        background: rgba(53, 122, 93, 0.12);
        color: #22543d;
        font-size: 14px;
      }

      .status.error {
        background: rgba(173, 79, 45, 0.14);
        color: #8c3c1d;
      }

      @media (max-width: 800px) {
        .content {
          grid-template-columns: 1fr;
          padding: 16px 20px 20px;
        }

        .hero {
          padding: 24px 20px 8px;
        }

        h1 {
          max-width: none;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">Russian Sentiment Analysis</p>
        <h1>Inspect review tone in one request.</h1>
        <p class="subtitle">
          This demo uses a fine-tuned ruBERT classifier trained on RuReviews.
          Paste a review, run inference, and inspect class probabilities.
        </p>
      </section>

      <section class="content">
        <article class="card">
          <label for="review">Review text</label>
          <textarea id="review">Приложение удобное, но после последнего обновления стало заметно медленнее.</textarea>
          <button id="predict-button" type="button">Analyze sentiment</button>
          <p class="hint">API docs are available at <code>/docs</code>; review QA uses <code>/ask</code>.</p>
        </article>

        <aside class="card">
          <div id="health" class="status">Checking model availability...</div>
          <div class="meta">
            <div>
              <p class="metric">Predicted label</p>
              <h2 id="label" class="result-label">-</h2>
            </div>
            <div>
              <p class="metric">Original text</p>
              <p id="echo" class="hint">-</p>
            </div>
          </div>
          <div id="scores" class="scores"></div>
        </aside>
      </section>
    </main>

    <script>
      const healthNode = document.getElementById("health");
      const labelNode = document.getElementById("label");
      const echoNode = document.getElementById("echo");
      const scoresNode = document.getElementById("scores");
      const reviewNode = document.getElementById("review");
      const buttonNode = document.getElementById("predict-button");

      function renderScores(scores) {
        const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        scoresNode.innerHTML = entries
          .map(([label, value]) => {
            const percent = (value * 100).toFixed(2);
            return `
              <div class="score-row">
                <strong>${label}</strong>
                <div class="bar"><span style="width: ${percent}%"></span></div>
                <span>${percent}%</span>
              </div>
            `;
          })
          .join("");
      }

      async function checkHealth() {
        try {
          const response = await fetch("/health");
          const payload = await response.json();
          if (payload.model_available) {
            healthNode.textContent = `Model ready: ${payload.model_dir}`;
            healthNode.className = "status";
          } else {
            healthNode.textContent = `Model missing: ${payload.model_dir}`;
            healthNode.className = "status error";
          }
        } catch (error) {
          healthNode.textContent = "Health check failed";
          healthNode.className = "status error";
        }
      }

      async function predict() {
        const text = reviewNode.value.trim();
        if (!text) {
          return;
        }

        buttonNode.disabled = true;
        buttonNode.textContent = "Analyzing...";

        try {
          const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.detail || "Prediction failed");
          }

          labelNode.textContent = payload.label;
          echoNode.textContent = payload.text;
          renderScores(payload.scores);
        } catch (error) {
          labelNode.textContent = "error";
          echoNode.textContent = String(error.message || error);
          scoresNode.innerHTML = "";
        } finally {
          buttonNode.disabled = false;
          buttonNode.textContent = "Analyze sentiment";
        }
      }

      buttonNode.addEventListener("click", predict);
      checkHealth();
      predict();
    </script>
  </body>
</html>
"""


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
    )
    return AskResponse(
        question=response.question,
        answer=response.answer,
        sentiment_focus=response.sentiment_focus,
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
