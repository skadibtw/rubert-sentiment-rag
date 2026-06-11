from __future__ import annotations

import json
from pathlib import Path

import torch
from joblib import dump

from src.inference.predictor import (
    PredictorConfig,
    SentimentPredictor,
    load_predictor,
)


class FakeSklearnPipeline:
    def predict(self, texts):  # type: ignore[no-untyped-def]
        del texts
        return [2]

    def predict_proba(self, texts):  # type: ignore[no-untyped-def]
        del texts
        return [[0.1, 0.2, 0.7]]


class FakeBinarySklearnPipeline:
    def predict(self, texts):  # type: ignore[no-untyped-def]
        del texts
        return [1]

    def predict_proba(self, texts):  # type: ignore[no-untyped-def]
        del texts
        return [[0.08, 0.92]]


class FakeTokenizer:
    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class FakeModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask):  # type: ignore[no-untyped-def]
        del input_ids, attention_mask
        return torch.tensor([[0.1, 0.4, 1.8]], dtype=torch.float32)


def test_sentiment_predictor_returns_label_and_scores() -> None:
    predictor = SentimentPredictor(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        config=PredictorConfig(
            model_name="fake-model",
            checkpoint_path=Path("checkpoint.pt"),
            tokenizer_dir=Path("tokenizer"),
            max_length=32,
            dropout=0.1,
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            device="cpu",
        ),
    )

    prediction = predictor.predict([" Отлично! "])[0]

    assert prediction.text == " Отлично! "
    assert prediction.label_id == 2
    assert prediction.label == "positive"
    assert set(prediction.scores) == {"negative", "neutral", "positive"}
    assert prediction.scores["positive"] > prediction.scores["neutral"]


def test_load_predictor_supports_sklearn_baseline(tmp_path: Path) -> None:
    dump(FakeSklearnPipeline(), tmp_path / "model.joblib")

    predictor = load_predictor(tmp_path)
    prediction = predictor.predict([" Отлично! "])[0]

    assert prediction.text == " Отлично! "
    assert prediction.label_id == 2
    assert prediction.label == "positive"
    assert prediction.scores["positive"] == 0.7


def test_load_predictor_reads_sklearn_label_metadata(tmp_path: Path) -> None:
    dump(FakeBinarySklearnPipeline(), tmp_path / "model.joblib")
    (tmp_path / "sklearn_metadata.json").write_text(
        json.dumps(
            {
                "label_mode": "binary_polarity",
                "id2label": {"0": "negative", "1": "positive"},
            }
        ),
        encoding="utf-8",
    )

    predictor = load_predictor(tmp_path)
    prediction = predictor.predict(["Отлично"])[0]

    assert prediction.label_id == 1
    assert prediction.label == "positive"
    assert prediction.scores == {"negative": 0.08, "positive": 0.92}
