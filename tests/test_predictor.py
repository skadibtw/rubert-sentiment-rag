from __future__ import annotations

from pathlib import Path

import torch

from src.inference.predictor import PredictorConfig, SentimentPredictor


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
