from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from joblib import load
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.dataset import clean_text
from src.models import BertSentimentClassifier
from src.training import detect_device


@dataclass(slots=True)
class PredictorConfig:
    model_name: str
    checkpoint_path: Path
    tokenizer_dir: Path
    max_length: int
    dropout: float
    id2label: dict[int, str]
    device: str


@dataclass(slots=True)
class SentimentPrediction:
    text: str
    label_id: int
    label: str
    scores: dict[str, float]


class SentimentPredictor:
    def __init__(
        self,
        *,
        model: BertSentimentClassifier,
        tokenizer: PreTrainedTokenizerBase,
        config: PredictorConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[SentimentPrediction]:
        cleaned_texts = [clean_text(text) for text in texts]
        encoded = self.tokenizer(
            cleaned_texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        probabilities = torch.softmax(logits, dim=-1).cpu()
        predictions = torch.argmax(probabilities, dim=-1).tolist()

        results = []
        for source_text, label_id, scores in zip(
            texts,
            predictions,
            probabilities.tolist(),
            strict=False,
        ):
            label_scores = {
                self.config.id2label[index]: round(score, 6)
                for index, score in enumerate(scores)
            }
            results.append(
                SentimentPrediction(
                    text=source_text,
                    label_id=label_id,
                    label=self.config.id2label[label_id],
                    scores=label_scores,
                )
            )
        return results


class SklearnSentimentPredictor:
    def __init__(
        self,
        *,
        pipeline: object,
        id2label: dict[int, str] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.id2label = id2label or {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, texts: list[str]) -> list[SentimentPrediction]:
        cleaned_texts = [clean_text(text) for text in texts]
        predictions = self.pipeline.predict(cleaned_texts)
        probabilities = self.pipeline.predict_proba(cleaned_texts)

        results = []
        for source_text, label_id, scores in zip(
            texts,
            predictions.tolist()
            if hasattr(predictions, "tolist")
            else list(predictions),
            probabilities.tolist()
            if hasattr(probabilities, "tolist")
            else list(probabilities),
            strict=False,
        ):
            label_id = int(label_id)
            label_scores = {
                self.id2label[index]: round(float(score), 6)
                for index, score in enumerate(scores)
            }
            results.append(
                SentimentPrediction(
                    text=source_text,
                    label_id=label_id,
                    label=self.id2label[label_id],
                    scores=label_scores,
                )
            )
        return results


def _load_metadata(model_dir: Path) -> dict[str, object]:
    metadata_path = model_dir / "model_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_sklearn_predictor(model_dir: Path) -> SklearnSentimentPredictor:
    pipeline = load(model_dir / "model.joblib")
    metadata_path = model_dir / "sklearn_metadata.json"
    id2label = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        id2label = {int(key): value for key, value in metadata["id2label"].items()}
    return SklearnSentimentPredictor(pipeline=pipeline, id2label=id2label)


def load_predictor(model_dir: Path) -> SentimentPredictor | SklearnSentimentPredictor:
    if (model_dir / "model.joblib").exists():
        return _load_sklearn_predictor(model_dir)

    metadata = _load_metadata(model_dir)
    checkpoint_path = model_dir / metadata["checkpoint_path"]
    tokenizer_dir = model_dir / metadata["tokenizer_dir"]
    id2label = {int(key): value for key, value in metadata["id2label"].items()}
    device = str(detect_device())

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = BertSentimentClassifier(
        model_name=str(metadata["model_name"]),
        num_labels=len(id2label),
        dropout=float(metadata["dropout"]),
        local_files_only=True,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    config = PredictorConfig(
        model_name=str(metadata["model_name"]),
        checkpoint_path=checkpoint_path,
        tokenizer_dir=tokenizer_dir,
        max_length=int(metadata["max_length"]),
        dropout=float(metadata["dropout"]),
        id2label=id2label,
        device=device,
    )
    return SentimentPredictor(model=model, tokenizer=tokenizer, config=config)
