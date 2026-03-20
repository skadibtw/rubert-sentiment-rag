from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    DEFAULT_CACHE_DIR,
    clean_text,
    get_dataset,
    maybe_sample_dataset,
)


@dataclass(slots=True)
class BaselineConfig:
    dataset_name: str = "ai-forever/ru-reviews-classification"
    cache_dir: Path = DEFAULT_CACHE_DIR
    text_column: str = "text"
    label_column: str = "label"
    max_features: int = 50000
    ngram_max: int = 2
    min_df: int = 3
    max_iter: int = 1000
    random_state: int = 42
    sample_size: int | None = None


def dataset_to_xy(
    split: Dataset, text_column: str, label_column: str
) -> tuple[list[str], list[int]]:
    texts = [clean_text(text) for text in split[text_column]]
    labels = [int(label) for label in split[label_column]]
    return texts, labels


def build_pipeline(config: BaselineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=config.max_features,
                    ngram_range=(1, config.ngram_max),
                    min_df=config.min_df,
                    strip_accents=None,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def run_baseline(config: BaselineConfig) -> dict[str, object]:
    dataset = get_dataset(cache_dir=config.cache_dir)
    dataset = maybe_sample_dataset(dataset, config.sample_size)
    train_split = dataset["train"]
    validation_split = dataset["validation"]
    test_split = dataset["test"]

    x_train, y_train = dataset_to_xy(
        train_split, config.text_column, config.label_column
    )
    x_validation, y_validation = dataset_to_xy(
        validation_split, config.text_column, config.label_column
    )
    x_test, y_test = dataset_to_xy(test_split, config.text_column, config.label_column)

    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)

    validation_predictions = pipeline.predict(x_validation)
    test_predictions = pipeline.predict(x_test)

    return {
        "pipeline": pipeline,
        "validation_accuracy": accuracy_score(y_validation, validation_predictions),
        "validation_f1_macro": f1_score(
            y_validation, validation_predictions, average="macro"
        ),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "test_f1_macro": f1_score(y_test, test_predictions, average="macro"),
        "test_report": classification_report(y_test, test_predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + logistic regression baseline"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Optional cap per split"
    )
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        cache_dir=args.cache_dir,
        sample_size=args.sample_size,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_iter=args.max_iter,
    )
    results = run_baseline(config)

    print("TF-IDF + LogisticRegression baseline")
    print(f"Validation accuracy: {results['validation_accuracy']:.4f}")
    print(f"Validation macro F1: {results['validation_f1_macro']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test macro F1: {results['test_f1_macro']:.4f}")
    print("\nClassification report:\n")
    print(results["test_report"])


if __name__ == "__main__":
    main()
