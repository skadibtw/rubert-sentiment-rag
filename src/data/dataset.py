from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


DATASET_NAME = "ai-forever/ru-reviews-classification"
DEFAULT_CACHE_DIR = Path("data/cache/ru_reviews")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = unescape(text)
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def get_dataset(
    cache_dir: Path = DEFAULT_CACHE_DIR, force_download: bool = False
) -> DatasetDict:
    if cache_dir.exists() and not force_download:
        dataset = load_from_disk(str(cache_dir))
        if not isinstance(dataset, DatasetDict):
            raise TypeError("Expected a DatasetDict in local cache")
        return dataset

    dataset = load_dataset(DATASET_NAME)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict from Hugging Face")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))
    return dataset


def maybe_sample_dataset(dataset: DatasetDict, sample_size: int | None) -> DatasetDict:
    if sample_size is None:
        return dataset

    sampled = {}
    for split_name, split in dataset.items():
        size = min(sample_size, len(split))
        sampled[split_name] = split.shuffle(seed=42).select(range(size))
    return DatasetDict(sampled)


def prepare_text_classification_dataset(
    dataset: DatasetDict,
    *,
    keep_label_text: bool = False,
) -> DatasetDict:
    columns_to_remove = ["id"]
    if not keep_label_text and "label_text" in dataset["train"].column_names:
        columns_to_remove.append("label_text")

    prepared = dataset.remove_columns(
        [
            column
            for column in columns_to_remove
            if column in dataset["train"].column_names
        ]
    )

    def clean_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        return {"text": [clean_text(text) for text in batch["text"]]}

    return prepared.map(clean_batch, batched=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache the reviews dataset"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local path for saved dataset",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download dataset and overwrite local cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_dataset(
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )
    print(f"Dataset ready at: {args.cache_dir}")
    print(dataset)


if __name__ == "__main__":
    main()
