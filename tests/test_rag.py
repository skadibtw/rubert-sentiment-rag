from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.rag.pipeline import (
    LLMConfig,
    ReviewRAG,
    extract_keyphrases,
    infer_sentiment_focus,
    resolve_generation_mode,
)


class FakeVectorStore:
    def similarity_search_with_score(  # type: ignore[no-untyped-def]
        self,
        query,
        *,
        k,
        filter=None,
    ):
        del query
        documents = [
            SimpleNamespace(
                page_content="После обновления приложение тормозит и вылетает.",
                metadata={
                    "label_id": 0,
                    "label": "negative",
                    "split": "train",
                },
            ),
            SimpleNamespace(
                page_content="Интерфейс стал удобнее и работает быстро.",
                metadata={
                    "label_id": 2,
                    "label": "positive",
                    "split": "validation",
                },
            ),
        ]
        if filter is not None:
            documents = [
                document
                for document in documents
                if document.metadata["label"] == filter["label"]
            ]
        return [
            (document, float(index)) for index, document in enumerate(documents[:k])
        ]


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


def test_resolve_generation_mode_falls_back_without_llm_credentials() -> None:
    config = LLMConfig(
        mode="auto", model="gpt-4o-mini", base_url="https://api.openai.com/v1"
    )

    assert resolve_generation_mode(None, config) == "extractive"


def test_review_rag_returns_most_relevant_contexts() -> None:
    rag = ReviewRAG(
        index_dir=Path("artifacts/rag"),
        embedding_model="fake-model",
        vector_store=FakeVectorStore(),
    )

    response = rag.answer("Почему после обновления все стало медленнее?", top_k=1)

    assert response.sentiment_focus == "negative"
    assert response.generation_mode == "extractive"
    assert response.llm_used is False
    assert response.contexts[0].label == "negative"
    assert "Found 1 relevant reviews" in response.answer
