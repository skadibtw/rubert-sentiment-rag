from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.dataset import (
    DEFAULT_CACHE_DIR,
    get_dataset,
    maybe_sample_dataset,
    prepare_text_classification_dataset,
)
from src.training import detect_device

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RAG_DIR = Path("artifacts/rag")
DEFAULT_TOP_K = 5
RUSSIAN_STOPWORDS = {
    "это",
    "как",
    "для",
    "что",
    "при",
    "если",
    "она",
    "они",
    "или",
    "уже",
    "просто",
    "после",
    "через",
    "когда",
    "который",
    "которая",
    "которые",
    "очень",
    "свои",
    "своих",
    "того",
    "только",
    "всего",
    "снова",
    "потому",
    "между",
    "будет",
    "чтобы",
    "можно",
    "нужно",
    "теперь",
    "раньше",
    "стало",
    "приложение",
    "приложении",
    "приложения",
    "программа",
    "программой",
    "сервис",
    "сервиса",
    "которое",
    "которых",
    "весь",
    "всем",
    "всех",
    "есть",
    "было",
    "были",
    "быть",
    "также",
    "пока",
    "потом",
    "либо",
    "лишь",
    "вроде",
    "тут",
    "там",
    "где",
    "куда",
    "отзыв",
    "отзывы",
    "отзыве",
    "мной",
    "меня",
    "тебя",
    "себя",
    "вами",
    "нами",
    "тоже",
    "себе",
    "этого",
    "этой",
    "этот",
    "этом",
    "много",
    "мало",
}


@dataclass(slots=True)
class RetrievedReview:
    text: str
    label_id: int
    label: str
    split: str
    score: float


@dataclass(slots=True)
class RAGResponse:
    question: str
    answer: str
    sentiment_focus: str | None
    keyphrases: list[str]
    label_distribution: dict[str, int]
    contexts: list[RetrievedReview]


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return str(detect_device())


def _load_encoder(model_name: str, device: str | None = None) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_resolve_device(device))


def _build_records(dataset: Any) -> list[dict[str, Any]]:
    prepared = prepare_text_classification_dataset(dataset, keep_label_text=True)
    records: list[dict[str, Any]] = []
    for split_name, split in prepared.items():
        label_texts = (
            split["label_text"] if "label_text" in split.column_names else None
        )
        for index, (text, label_id) in enumerate(
            zip(split["text"], split["label"], strict=False)
        ):
            label = (
                str(label_texts[index]) if label_texts is not None else str(label_id)
            )
            records.append(
                {
                    "text": text,
                    "label_id": int(label_id),
                    "label": label,
                    "split": split_name,
                }
            )
    return records


def _save_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _truncate(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def infer_sentiment_focus(question: str) -> str | None:
    lowered = question.casefold()
    negative_markers = {
        "негатив",
        "жалоб",
        "жалуют",
        "плохо",
        "проблем",
        "ошиб",
        "медлен",
        "не работает",
        "слом",
        "минус",
    }
    positive_markers = {
        "позитив",
        "плюс",
        "нрав",
        "хорош",
        "удоб",
        "быстро",
        "лучше",
        "довол",
    }
    neutral_markers = {"нейтрал", "без эмоций", "обычн", "средне"}

    if any(marker in lowered for marker in negative_markers):
        return "negative"
    if any(marker in lowered for marker in positive_markers):
        return "positive"
    if any(marker in lowered for marker in neutral_markers):
        return "neutral"
    return None


def extract_keyphrases(texts: list[str], top_n: int = 5) -> list[str]:
    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=1000,
        token_pattern=r"(?u)\b[\w-]{3,}\b",
        stop_words=list(RUSSIAN_STOPWORDS),
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(scores)[::-1]

    phrases: list[str] = []
    for index in order:
        phrase = str(terms[index]).strip()
        if not phrase or any(char.isdigit() for char in phrase):
            continue
        phrases.append(phrase)
        if len(phrases) == top_n:
            break
    return phrases


def compose_answer(
    question: str,
    contexts: list[RetrievedReview],
    sentiment_focus: str | None,
    keyphrases: list[str],
    label_distribution: dict[str, int],
) -> str:
    if not contexts:
        return (
            "No relevant reviews were found for this question. Try a broader query "
            "or rebuild the index with more data."
        )

    dominant_label = max(label_distribution, key=label_distribution.get)
    fragments = [
        f"Found {len(contexts)} relevant reviews for: {question}",
        (
            "Most retrieved reviews are "
            f"{dominant_label} ({label_distribution[dominant_label]}/{len(contexts)})."
        ),
    ]
    if sentiment_focus is not None:
        fragments.append(f"Sentiment focus applied: {sentiment_focus}.")
    if keyphrases:
        fragments.append("Recurring themes: " + ", ".join(keyphrases[:5]) + ".")
    example_snippets = "; ".join(_truncate(item.text, 120) for item in contexts[:2])
    fragments.append("Example evidence: " + example_snippets)
    return " ".join(fragments)


def build_review_index(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_RAG_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    sample_size: int | None = None,
    device: str | None = None,
) -> Path:
    dataset = get_dataset(cache_dir=cache_dir)
    dataset = maybe_sample_dataset(dataset, sample_size)
    records = _build_records(dataset)
    encoder = _load_encoder(model_name, device=device)
    texts = [record["text"] for record in records]
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    _save_records(output_dir / "reviews.jsonl", records)
    metadata = {
        "embedding_model": model_name,
        "size": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "sample_size": sample_size,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


class ReviewRAG:
    def __init__(
        self,
        *,
        index_dir: Path,
        records: list[dict[str, Any]],
        embeddings: np.ndarray,
        embedding_model: str,
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.records = records
        self.embeddings = embeddings.astype(np.float32)
        self.embedding_model = embedding_model
        self.device = device
        self._encoder = encoder

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = _load_encoder(self.embedding_model, device=self.device)
        return self._encoder

    def search(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        sentiment_focus: str | None = None,
    ) -> list[RetrievedReview]:
        resolved_focus = sentiment_focus or infer_sentiment_focus(question)
        candidate_indices = list(range(len(self.records)))
        if resolved_focus is not None:
            candidate_indices = [
                index
                for index, record in enumerate(self.records)
                if str(record["label"]).casefold() == resolved_focus.casefold()
            ]
            if not candidate_indices:
                candidate_indices = list(range(len(self.records)))

        query_embedding = self.encoder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        candidate_matrix = self.embeddings[candidate_indices]
        scores = candidate_matrix @ query_embedding

        limit = min(top_k, len(candidate_indices))
        best_local = np.argsort(scores)[::-1][:limit]
        results: list[RetrievedReview] = []
        for local_index in best_local:
            record_index = candidate_indices[int(local_index)]
            record = self.records[record_index]
            results.append(
                RetrievedReview(
                    text=str(record["text"]),
                    label_id=int(record["label_id"]),
                    label=str(record["label"]),
                    split=str(record["split"]),
                    score=float(scores[int(local_index)]),
                )
            )
        return results

    def answer(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        sentiment_focus: str | None = None,
    ) -> RAGResponse:
        resolved_focus = sentiment_focus or infer_sentiment_focus(question)
        contexts = self.search(
            question,
            top_k=top_k,
            sentiment_focus=resolved_focus,
        )
        label_distribution = dict(Counter(item.label for item in contexts))
        keyphrases = extract_keyphrases([item.text for item in contexts])
        answer = compose_answer(
            question,
            contexts,
            resolved_focus,
            keyphrases,
            label_distribution,
        )
        return RAGResponse(
            question=question,
            answer=answer,
            sentiment_focus=resolved_focus,
            keyphrases=keyphrases,
            label_distribution=label_distribution,
            contexts=contexts,
        )


def load_review_rag(
    index_dir: Path = DEFAULT_RAG_DIR,
    *,
    device: str | None = None,
    encoder: Any | None = None,
) -> ReviewRAG:
    metadata_path = index_dir / "metadata.json"
    reviews_path = index_dir / "reviews.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    if (
        not metadata_path.exists()
        or not reviews_path.exists()
        or not embeddings_path.exists()
    ):
        raise FileNotFoundError(
            f"RAG index is incomplete or missing in directory: {index_dir}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = _load_records(reviews_path)
    embeddings = np.load(embeddings_path)
    return ReviewRAG(
        index_dir=index_dir,
        records=records,
        embeddings=embeddings,
        embedding_model=str(metadata["embedding_model"]),
        device=device,
        encoder=encoder,
    )


def parse_build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local review retrieval index")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_build_args()
    output_dir = build_review_index(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        device=args.device,
    )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved RAG index to: {output_dir}")


if __name__ == "__main__":
    main()
