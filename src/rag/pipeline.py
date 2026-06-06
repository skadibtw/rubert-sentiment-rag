from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE_DIR = Path("data/cache/ru_reviews")
DEFAULT_RAG_DIR = Path("artifacts/rag")
DEFAULT_TOP_K = 5
DEFAULT_GENERATION_MODE = "auto"
DEFAULT_LLM_TIMEOUT = 45.0
DEFAULT_CHROMA_COLLECTION = "ru_reviews"
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
    generation_mode: str
    llm_used: bool
    keyphrases: list[str]
    label_distribution: dict[str, int]
    contexts: list[RetrievedReview]


@dataclass(slots=True)
class LLMConfig:
    mode: str = DEFAULT_GENERATION_MODE
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = DEFAULT_LLM_TIMEOUT


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    from src.training import detect_device

    return str(detect_device())


def _load_embeddings(model_name: str, device: str | None = None) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _resolve_device(device)},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_chroma(
    *,
    index_dir: Path,
    collection_name: str,
    embeddings: Any,
) -> Any:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(index_dir),
    )


def _record_to_document(record: dict[str, Any], index: int) -> Any:
    from langchain_core.documents import Document

    return Document(
        page_content=str(record["text"]),
        metadata={
            "label_id": int(record["label_id"]),
            "label": str(record["label"]),
            "split": str(record["split"]),
            "row_id": index,
        },
    )


def load_llm_config(mode: str | None = None) -> LLMConfig:
    return LLMConfig(
        mode=mode or os.getenv("RAG_GENERATION_MODE", DEFAULT_GENERATION_MODE),
        model=os.getenv("OPENAI_MODEL") or os.getenv("RAG_LLM_MODEL"),
        base_url=(
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("RAG_LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("RAG_LLM_API_KEY"),
        timeout=float(os.getenv("RAG_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT)),
    )


def is_llm_available(config: LLMConfig) -> bool:
    if not config.model or not config.base_url:
        return False
    if "api.openai.com" in config.base_url and not config.api_key:
        return False
    return True


def resolve_generation_mode(
    requested_mode: str | None,
    config: LLMConfig,
) -> str:
    mode = (requested_mode or config.mode or DEFAULT_GENERATION_MODE).casefold()
    if mode not in {"auto", "extractive", "llm"}:
        mode = DEFAULT_GENERATION_MODE
    if mode == "extractive":
        return "extractive"
    if mode == "llm":
        return "llm" if is_llm_available(config) else "extractive"
    return "llm" if is_llm_available(config) else "extractive"


def _build_records(dataset: Any) -> list[dict[str, Any]]:
    from src.data.dataset import prepare_text_classification_dataset

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


def compose_llm_prompt(
    question: str,
    contexts: list[RetrievedReview],
    sentiment_focus: str | None,
    keyphrases: list[str],
    label_distribution: dict[str, int],
) -> str:
    context_lines = []
    for index, item in enumerate(contexts, start=1):
        context_lines.append(
            (
                f"[{index}] label={item.label}; split={item.split}; "
                f"score={item.score:.4f}; text={item.text}"
            )
        )
    focus_line = sentiment_focus or "not specified"
    keyphrase_line = ", ".join(keyphrases) if keyphrases else "none"
    distribution_line = ", ".join(
        f"{label}: {count}" for label, count in label_distribution.items()
    )
    return "\n".join(
        [
            "Answer in Russian.",
            "Use only the evidence below.",
            "If the evidence is weak, say that the conclusion is approximate.",
            f"Question: {question}",
            f"Sentiment focus: {focus_line}",
            f"Key phrases: {keyphrase_line}",
            f"Label distribution: {distribution_line}",
            "Evidence:",
            *context_lines,
            (
                "Give a concise answer followed by 2-4 bullet-like findings "
                "in one paragraph."
            ),
        ]
    )


def generate_llm_answer(
    question: str,
    contexts: list[RetrievedReview],
    sentiment_focus: str | None,
    keyphrases: list[str],
    label_distribution: dict[str, int],
    config: LLMConfig,
) -> str:
    if not is_llm_available(config):
        raise RuntimeError("LLM generation is not configured")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You summarize user review corpora. Ground every answer in the "
                    "provided evidence and do not invent product facts."
                ),
            ),
            ("user", "{question_prompt}"),
        ]
    )
    llm = ChatOpenAI(
        model=str(config.model),
        api_key=config.api_key,
        base_url=str(config.base_url).rstrip("/"),
        temperature=0.2,
        timeout=config.timeout,
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "question_prompt": compose_llm_prompt(
                question,
                contexts,
                sentiment_focus,
                keyphrases,
                label_distribution,
            )
        }
    )
    return str(response.content).strip()


def build_review_index(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_RAG_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    sample_size: int | None = None,
    device: str | None = None,
) -> Path:
    from langchain_chroma import Chroma

    from src.data.dataset import get_dataset, maybe_sample_dataset

    dataset = get_dataset(cache_dir=cache_dir)
    dataset = maybe_sample_dataset(dataset, sample_size)
    records = _build_records(dataset)
    documents = [
        _record_to_document(record, index) for index, record in enumerate(records)
    ]
    embeddings = _load_embeddings(model_name, device=device)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    vector_store = Chroma(
        collection_name=DEFAULT_CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(output_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    chunk_size = max(1, batch_size)
    for start in range(0, len(documents), chunk_size):
        vector_store.add_documents(documents[start : start + chunk_size])
    _save_records(output_dir / "reviews.jsonl", records)
    metadata = {
        "vector_store": "chroma",
        "collection_name": DEFAULT_CHROMA_COLLECTION,
        "embedding_model": model_name,
        "size": len(records),
        "sample_size": sample_size,
        "batch_size": batch_size,
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
        embedding_model: str,
        vector_store: Any,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        device: str | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.device = device
        self.vector_store = vector_store
        self.llm_config = llm_config or load_llm_config()

    def search(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        sentiment_focus: str | None = None,
    ) -> list[RetrievedReview]:
        resolved_focus = sentiment_focus or infer_sentiment_focus(question)
        metadata_filter = None
        if resolved_focus is not None:
            metadata_filter = {"label": resolved_focus.casefold()}

        search_kwargs: dict[str, Any] = {"k": top_k}
        if metadata_filter is not None:
            search_kwargs["filter"] = metadata_filter
        raw_results = self.vector_store.similarity_search_with_score(
            question,
            **search_kwargs,
        )
        if not raw_results and metadata_filter is not None:
            raw_results = self.vector_store.similarity_search_with_score(
                question,
                k=top_k,
            )

        return [
            RetrievedReview(
                text=str(document.page_content),
                label_id=int(document.metadata["label_id"]),
                label=str(document.metadata["label"]),
                split=str(document.metadata["split"]),
                score=float(score),
            )
            for document, score in raw_results
        ]

    def answer(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        sentiment_focus: str | None = None,
        generation_mode: str | None = None,
    ) -> RAGResponse:
        resolved_focus = sentiment_focus or infer_sentiment_focus(question)
        contexts = self.search(
            question,
            top_k=top_k,
            sentiment_focus=resolved_focus,
        )
        label_distribution = dict(Counter(item.label for item in contexts))
        keyphrases = extract_keyphrases([item.text for item in contexts])
        resolved_mode = resolve_generation_mode(generation_mode, self.llm_config)
        llm_used = False
        if resolved_mode == "llm":
            try:
                answer = generate_llm_answer(
                    question,
                    contexts,
                    resolved_focus,
                    keyphrases,
                    label_distribution,
                    self.llm_config,
                )
                llm_used = True
            except Exception:
                resolved_mode = "extractive"
                answer = compose_answer(
                    question,
                    contexts,
                    resolved_focus,
                    keyphrases,
                    label_distribution,
                )
        else:
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
            generation_mode=resolved_mode,
            llm_used=llm_used,
            keyphrases=keyphrases,
            label_distribution=label_distribution,
            contexts=contexts,
        )


def load_review_rag(
    index_dir: Path = DEFAULT_RAG_DIR,
    *,
    device: str | None = None,
    llm_config: LLMConfig | None = None,
) -> ReviewRAG:
    metadata_path = index_dir / "metadata.json"
    reviews_path = index_dir / "reviews.jsonl"
    chroma_path = index_dir / "chroma.sqlite3"
    if (
        not metadata_path.exists()
        or not reviews_path.exists()
        or not chroma_path.exists()
    ):
        raise FileNotFoundError(
            f"RAG index is incomplete or missing in directory: {index_dir}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    collection_name = str(metadata.get("collection_name", DEFAULT_CHROMA_COLLECTION))
    embedding_model = str(metadata["embedding_model"])
    embeddings = _load_embeddings(embedding_model, device=device)
    vector_store = _load_chroma(
        index_dir=index_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return ReviewRAG(
        index_dir=index_dir,
        embedding_model=embedding_model,
        collection_name=collection_name,
        vector_store=vector_store,
        device=device,
        llm_config=llm_config,
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
