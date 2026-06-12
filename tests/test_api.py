from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

from fastapi.testclient import TestClient
from joblib import dump

api_module = importlib.import_module("src.api.app")


@dataclass
class FakePrediction:
    text: str
    label_id: int
    label: str
    scores: dict[str, float]


class FakePredictor:
    def predict(self, texts: list[str]) -> list[FakePrediction]:
        return [
            FakePrediction(
                text=texts[0],
                label_id=2,
                label="positive",
                scores={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
            )
        ]


class FakeSklearnPipeline:
    def predict(self, texts):  # type: ignore[no-untyped-def]
        del texts
        return [2]

    def predict_proba(self, texts):  # type: ignore[no-untyped-def]
        del texts
        return [[0.1, 0.2, 0.7]]


class FakeRAG:
    def answer(
        self,
        question: str,
        *,
        top_k: int,
        sentiment_focus: str | None,
        generation_mode: str | None = None,
    ):
        del top_k, sentiment_focus, generation_mode
        return SimpleNamespace(
            question=question,
            answer="Found relevant reviews about slow updates.",
            sentiment_focus="negative",
            generation_mode="llm",
            llm_used=True,
            keyphrases=["последнее обновление", "медленно работает"],
            label_distribution={"negative": 2},
            contexts=[
                SimpleNamespace(
                    text="После обновления приложение стало медленным.",
                    label_id=0,
                    label="negative",
                    split="train",
                    score=0.91,
                )
            ],
        )


def test_health_endpoint_reports_missing_model(tmp_path) -> None:
    api_module.get_predictor.cache_clear()
    api_module.get_rag_pipeline.cache_clear()
    api_module.DEFAULT_MODEL_DIR = tmp_path / "missing-model"
    api_module.DEFAULT_RAG_INDEX_DIR = tmp_path / "missing-rag"
    client = TestClient(api_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_available"] is False
    assert payload["rag_index_available"] is False


def test_predict_endpoint_uses_loaded_predictor(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    api_module.get_predictor.cache_clear()
    api_module.DEFAULT_MODEL_DIR = model_dir
    monkeypatch.setattr(api_module, "load_predictor", lambda _: FakePredictor())
    client = TestClient(api_module.app)

    response = client.post("/predict", json={"text": "Очень полезное приложение"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "positive"
    assert payload["label_id"] == 2
    assert payload["scores"]["positive"] == 0.7


def test_predict_endpoint_uses_baseline_artifact_by_default(
    tmp_path, monkeypatch
) -> None:
    model_dir = tmp_path / "artifacts" / "baseline"
    model_dir.mkdir(parents=True)
    dump(FakeSklearnPipeline(), model_dir / "model.joblib")

    api_module.get_predictor.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        api_module,
        "DEFAULT_MODEL_DIR",
        api_module.Path("artifacts/baseline"),
    )
    client = TestClient(api_module.app)

    response = client.post("/predict", json={"text": "Очень полезное приложение"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"negative", "neutral", "positive"}
    assert set(payload["scores"]) == {"negative", "neutral", "positive"}


def test_ask_endpoint_uses_loaded_rag_pipeline(tmp_path, monkeypatch) -> None:
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    api_module.get_rag_pipeline.cache_clear()
    api_module.DEFAULT_RAG_INDEX_DIR = rag_dir
    monkeypatch.setattr(api_module, "load_review_rag", lambda _: FakeRAG())
    client = TestClient(api_module.app)

    response = client.post(
        "/ask",
        json={
            "question": "На что чаще всего жалуются после обновления?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sentiment_focus"] == "negative"
    assert payload["generation_mode"] == "llm"
    assert payload["llm_used"] is True
    assert payload["label_distribution"]["negative"] == 2
    assert payload["contexts"][0]["label"] == "negative"


def test_demo_page_is_available() -> None:
    client = TestClient(api_module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "NLP Pet Project Demo" in response.text
