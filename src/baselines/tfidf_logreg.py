from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    DEFAULT_CACHE_DIR,
    clean_text,
    get_dataset,
    maybe_sample_dataset,
)
from src.evaluation.metrics import (
    build_error_table,
    classification_metrics,
    confusion_matrix_frame,
    save_evaluation_artifacts,
)

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass(slots=True)
class BaselineConfig:
    dataset_name: str = "ai-forever/ru-reviews-classification"
    cache_dir: Path = DEFAULT_CACHE_DIR
    text_column: str = "text"
    label_column: str = "label"
    feature_mode: str = "word_char"
    max_features: int = 100000
    ngram_max: int = 2
    min_df: int = 2
    char_max_features: int = 50000
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    regularization_c: float = 0.7
    max_iter: int = 1000
    random_state: int = 42
    sample_size: int | None = None
    output_dir: Path = Path("artifacts/baseline")


def dataset_to_xy(
    split: Dataset, text_column: str, label_column: str
) -> tuple[list[str], list[int]]:
    texts = [clean_text(text) for text in split[text_column]]
    labels = [int(label) for label in split[label_column]]
    return texts, labels


def build_vectorizer(config: BaselineConfig) -> TfidfVectorizer | FeatureUnion:
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=config.max_features,
        ngram_range=(1, config.ngram_max),
        min_df=config.min_df,
        strip_accents=None,
        sublinear_tf=True,
    )
    if config.feature_mode == "word":
        return word_vectorizer
    if config.feature_mode == "word_char":
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            max_features=config.char_max_features,
            ngram_range=(config.char_ngram_min, config.char_ngram_max),
            min_df=config.min_df,
            strip_accents=None,
            sublinear_tf=True,
        )
        return FeatureUnion(
            [
                ("word", word_vectorizer),
                ("char", char_vectorizer),
            ]
        )
    msg = "feature_mode must be one of: word, word_char"
    raise ValueError(msg)


def build_pipeline(config: BaselineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_vectorizer(config)),
            (
                "classifier",
                LogisticRegression(
                    C=config.regularization_c,
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
    validation_metrics = classification_metrics(
        y_validation, validation_predictions.tolist()
    )
    test_metrics = classification_metrics(y_test, test_predictions.tolist())
    confusion = confusion_matrix_frame(
        y_test,
        test_predictions.tolist(),
        labels=[ID2LABEL[index] for index in sorted(ID2LABEL)],
    )
    errors = build_error_table(
        x_test,
        y_test,
        test_predictions.tolist(),
        id2label=ID2LABEL,
    )
    save_evaluation_artifacts(
        config.output_dir,
        metrics=test_metrics,
        confusion=confusion,
        errors=errors,
    )
    dump(pipeline, config.output_dir / "model.joblib")

    return {
        "pipeline": pipeline,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_f1_macro": validation_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_report": test_metrics["report_text"],
        "output_dir": config.output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + logistic regression baseline"
    )
    defaults = BaselineConfig()
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Optional cap per split"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["word", "word_char"],
        default=defaults.feature_mode,
    )
    parser.add_argument("--max-features", type=int, default=defaults.max_features)
    parser.add_argument("--ngram-max", type=int, default=defaults.ngram_max)
    parser.add_argument("--min-df", type=int, default=defaults.min_df)
    parser.add_argument(
        "--char-max-features", type=int, default=defaults.char_max_features
    )
    parser.add_argument("--char-ngram-min", type=int, default=defaults.char_ngram_min)
    parser.add_argument("--char-ngram-max", type=int, default=defaults.char_ngram_max)
    parser.add_argument(
        "--regularization-c", type=float, default=defaults.regularization_c
    )
    parser.add_argument("--max-iter", type=int, default=defaults.max_iter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        cache_dir=args.cache_dir,
        sample_size=args.sample_size,
        feature_mode=args.feature_mode,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        char_max_features=args.char_max_features,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        regularization_c=args.regularization_c,
        max_iter=args.max_iter,
        output_dir=args.output_dir,
    )
    results = run_baseline(config)

    print("TF-IDF + LogisticRegression baseline")
    print(f"Validation accuracy: {results['validation_accuracy']:.4f}")
    print(f"Validation macro F1: {results['validation_f1_macro']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test macro F1: {results['test_f1_macro']:.4f}")
    print(f"Saved evaluation artifacts to: {results['output_dir']}")
    print("\nClassification report:\n")
    print(results["test_report"])


if __name__ == "__main__":
    main()
