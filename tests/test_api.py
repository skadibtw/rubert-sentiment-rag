from __future__ import annotations

import importlib
from dataclasses import dataclass

from fastapi.testclient import TestClient

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


def test_health_endpoint_reports_missing_model(tmp_path) -> None:
    api_module.get_predictor.cache_clear()
    api_module.DEFAULT_MODEL_DIR = tmp_path / "missing-model"
    client = TestClient(api_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_available"] is False


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


def test_demo_page_is_available() -> None:
    client = TestClient(api_module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "NLP Pet Project Demo" in response.text
