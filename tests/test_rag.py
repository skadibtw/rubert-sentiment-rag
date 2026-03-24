from __future__ import annotations

from pathlib import Path

import numpy as np

from src.rag.pipeline import ReviewRAG, extract_keyphrases, infer_sentiment_focus


class FakeEncoder:
    def encode(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        embeddings = []
        for text in texts:
            lowered = text.casefold()
            if "обнов" in lowered or "медлен" in lowered:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return np.asarray(embeddings, dtype=np.float32)


def test_infer_sentiment_focus_detects_negative_queries() -> None:
    query = "На что чаще всего жалуются после обновления?"

    assert infer_sentiment_focus(query) == "negative"


def test_extract_keyphrases_returns_non_empty_list() -> None:
    phrases = extract_keyphrases(
        [
            "После обновления приложение стало медленно работать.",
            "Последнее обновление привело к проблемам со скоростью.",
        ]
    )

    assert phrases


def test_review_rag_returns_most_relevant_contexts() -> None:
    rag = ReviewRAG(
        index_dir=Path("artifacts/rag"),
        embedding_model="fake-model",
        encoder=FakeEncoder(),
        records=[
            {
                "text": "После обновления приложение тормозит и вылетает.",
                "label_id": 0,
                "label": "negative",
                "split": "train",
            },
            {
                "text": "Интерфейс стал удобнее и работает быстро.",
                "label_id": 2,
                "label": "positive",
                "split": "validation",
            },
        ],
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    response = rag.answer("Почему после обновления все стало медленнее?", top_k=1)

    assert response.sentiment_focus == "negative"
    assert response.contexts[0].label == "negative"
    assert "Found 1 relevant reviews" in response.answer
